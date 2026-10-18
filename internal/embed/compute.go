package embed

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/tryatlas/cli/internal/clistate"
)

// Backend is where memory search's vectors come from.
type Backend string

// The backends.
const (
	Off      Backend = "off"      // keywords only: no model is loaded
	CPU      Backend = "cpu"      // this package, on this machine's cores
	Endpoint Backend = "endpoint" // an OpenAI-compatible embeddings API: Ollama, TEI, vLLM, LM Studio
)

// Settings are this machine's choice for memory search, in the CLI's state
// directory (compute.json); ATLAS_EMBED, ATLAS_EMBED_MODEL and
// ATLAS_EMBED_THREADS override them.
type Settings struct {
	// Mode is auto (empty), off, cpu, ollama, or the http(s) URL of an
	// OpenAI-compatible API (the part before /embeddings).
	Mode string `json:"mode,omitempty"`
	// Model is the name an endpoint is asked for (default: all-minilm for
	// Ollama, else sentence-transformers/all-MiniLM-L6-v2).
	Model string `json:"model,omitempty"`
	// Threads caps the cores one embedding uses on the CPU; 0 is automatic.
	Threads int `json:"threads,omitempty"`
}

// Ollama, as it listens by default.
const (
	OllamaURL   = "http://127.0.0.1:11434/v1"
	OllamaModel = "all-minilm"
)

// ollamaURL is where Ollama is looked for; tests move it.
var ollamaURL = OllamaURL

func settingsPath() string { return filepath.Join(clistate.Base(), "compute.json") }

// SavedSettings reads compute.json alone; a missing file is auto.
func SavedSettings() Settings {
	var s Settings
	if b, err := os.ReadFile(settingsPath()); err == nil {
		_ = json.Unmarshal(b, &s)
	}
	return s
}

// LoadSettings is compute.json with the environment's overrides.
func LoadSettings() Settings {
	s := SavedSettings()
	if v := strings.TrimSpace(os.Getenv("ATLAS_EMBED")); v != "" {
		s.Mode = v
	}
	if v := strings.TrimSpace(os.Getenv("ATLAS_EMBED_MODEL")); v != "" {
		s.Model = v
	}
	if n, err := strconv.Atoi(os.Getenv("ATLAS_EMBED_THREADS")); err == nil && n >= 0 {
		s.Threads = n
	}
	return s
}

// SaveSettings writes compute.json and drops the plan this process holds.
func SaveSettings(s Settings) error {
	b, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(clistate.Base(), 0o700); err != nil {
		return err
	}
	if err := writeAtomic(settingsPath(), append(b, '\n')); err != nil {
		return fmt.Errorf("save the compute setting: %w", err)
	}
	Reset()
	return nil
}

func writeAtomic(path string, b []byte) error {
	tmp := fmt.Sprintf("%s.%d.tmp", path, os.Getpid())
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// ParseMode normalises a mode as Settings.Mode holds it.
func ParseMode(m string) (string, error) {
	switch v := strings.ToLower(strings.TrimSpace(m)); v {
	case "", "auto":
		return "", nil
	case "off", "keywords", "none":
		return "off", nil
	case "cpu", "local":
		return "cpu", nil
	case "ollama":
		return "ollama", nil
	}
	u, err := url.Parse(strings.TrimSpace(m))
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return "", fmt.Errorf("%q is not a mode: auto, off, cpu, ollama, or the http(s) URL of an OpenAI-compatible embeddings API", m)
	}
	return strings.TrimSuffix(strings.TrimSuffix(u.String(), "/"), "/embeddings"), nil
}

// Plan is how memory search embeds on this machine.
type Plan struct {
	Backend Backend
	// Mode is what it was chosen from: auto, off, cpu, ollama or a URL.
	Mode string
	// URL and Model are the endpoint's.
	URL, Model string
	// Threads caps the cores one embedding uses (0: every core);
	// Background is what a background pass runs on.
	Threads, Background int
	Machine             Machine
	// Note says why, for atlas-cli status.
	Note string
}

// Choose is the plan settings give on a machine before any endpoint is
// reached: auto is the CPU, or keywords only on a small machine.
func Choose(s Settings, m Machine) Plan {
	mode, err := ParseMode(s.Mode)
	p := Plan{Machine: m, Mode: "auto", Threads: s.Threads, Background: m.BackgroundThreads()}
	if s.Threads > 0 {
		p.Background = min(s.Threads, p.Background)
	}
	switch {
	case err != nil:
		p.Note = fmt.Sprintf("%q is not a mode, so automatic", s.Mode)
	case mode == "off":
		p.Mode, p.Backend, p.Note = mode, Off, "as set"
		return p
	case mode == "cpu":
		p.Mode, p.Backend, p.Note = mode, CPU, "as set"
		return p
	case mode == "ollama":
		p.Mode, p.Backend, p.URL, p.Model = mode, Endpoint, ollamaURL, cmpOr(s.Model, OllamaModel)
		return p
	case mode != "":
		p.Mode, p.Backend, p.URL, p.Model = mode, Endpoint, mode, cmpOr(s.Model, Repo)
		return p
	}
	if m.Small() {
		p.Backend = Off
		p.Note = strings.TrimPrefix(p.Note+"; ", "; ") + fmt.Sprintf("a small machine (%s); atlas-cli memory compute cpu overrides it", m)
		return p
	}
	p.Backend = CPU
	return p
}

func cmpOr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// Resolve is the plan in force: Choose, with an endpoint checked against
// the probes first. A set endpoint that cannot be reached, or that embeds
// otherwise, gives way to what auto picks; auto also takes a local Ollama
// that serves all-minilm and embeds as this package does.
func Resolve(ctx context.Context) Plan {
	s := LoadSettings()
	p := Choose(s, DetectMachine())
	if p.Backend == Endpoint {
		ep := newEndpoint(p.URL, p.Model)
		err := ep.reachable(ctx) // a recorded match says nothing about now
		if err == nil {
			err = checked(ctx, ep, false)
		}
		if err != nil {
			fallback := Choose(Settings{Threads: s.Threads}, p.Machine)
			fallback.Mode = p.Mode
			fallback.Note = fmt.Sprintf("%s: %v; %s instead", ep.label(), err, describeBackend(fallback.Backend))
			return fallback
		}
		p.Note = "verified against the app's model"
		return p
	}
	if p.Mode == "auto" {
		if name := localOllamaModel(ctx); name != "" {
			ep := newEndpoint(ollamaURL, name)
			var mm mismatchError
			switch err := checked(ctx, ep, false); {
			case err == nil:
				p.Backend, p.URL, p.Model = Endpoint, ollamaURL, name
				p.Note = "found a local Ollama serving " + name
			case errors.As(err, &mm):
				p.Note = strings.TrimPrefix(p.Note+"; ", "; ") + "the local Ollama's " + name + " is not used: " + err.Error()
			}
		}
	}
	return p
}

func describeBackend(b Backend) string {
	if b == CPU {
		return "this machine's CPU"
	}
	return "keywords only"
}

// Verification outcomes are kept per endpoint and model in
// compute-state.json: a match for a month, a mismatch for a day, an
// endpoint that could not be reached for two minutes (so a hung server
// stalls one process, not each). atlas-cli memory compute always checks
// afresh.
const (
	verifiedFor = 30 * 24 * time.Hour
	rejectedFor = 24 * time.Hour
	downFor     = 2 * time.Minute
)

type verdict struct {
	OK     bool      `json:"ok"`
	Cosine float32   `json:"cosine"`
	Down   string    `json:"down,omitempty"` // why it could not be reached
	At     time.Time `json:"at"`
}

func statePath() string { return filepath.Join(clistate.Base(), "compute-state.json") }

var stateMu sync.Mutex

// checked is nil when ep embeds as this package does, by a recorded
// verification or, when there is none or fresh is set, a new one.
func checked(ctx context.Context, ep *endpoint, fresh bool) error {
	key := ep.url + " " + ep.model
	if v, ok := readState()[key]; ok && !fresh {
		switch age := time.Since(v.At); {
		case v.OK && age < verifiedFor:
			return nil
		case v.Down != "" && age < downFor:
			return errors.New(v.Down)
		case !v.OK && v.Down == "" && age < rejectedFor:
			return mismatch(v.Cosine)
		}
	}
	cos, err := verify(ctx, ep)
	v := verdict{OK: err == nil, Cosine: cos, At: time.Now().UTC()}
	var mm mismatchError
	if err != nil && !errors.As(err, &mm) {
		v.Down = err.Error()
	}
	stateMu.Lock()
	defer stateMu.Unlock()
	state := readState()
	state[key] = v
	if b, merr := json.MarshalIndent(state, "", "  "); merr == nil && os.MkdirAll(clistate.Base(), 0o700) == nil {
		_ = writeAtomic(statePath(), append(b, '\n'))
	}
	return err
}

func readState() map[string]verdict {
	state := map[string]verdict{}
	if b, err := os.ReadFile(statePath()); err == nil {
		_ = json.Unmarshal(b, &state)
	}
	if state == nil { // a file holding null
		state = map[string]verdict{}
	}
	return state
}

// MinCosine is how close an endpoint's vector must be to this package's
// for every probe.
const MinCosine = 0.995

type mismatchError float32

func (m mismatchError) Error() string {
	return fmt.Sprintf("its vectors differ from the app's model (cosine %.3f, needs %.3f)", float32(m), MinCosine)
}

func mismatch(cos float32) error { return mismatchError(cos) }

// verify embeds the probes at ep and returns the worst cosine to this
// package's vectors; an error when one is below MinCosine.
func verify(ctx context.Context, ep *endpoint) (float32, error) {
	vs, err := ep.embed(ctx, probes)
	if err != nil {
		return 0, err
	}
	worst := float32(1)
	for i, want := range probeVectors {
		worst = min(worst, Cosine(vs[i], decodeProbe(want)))
	}
	if worst < MinCosine {
		return worst, mismatch(worst)
	}
	return worst, nil
}

// VerifyEndpoint checks url and model afresh, as atlas-cli memory compute
// sets them, and records the outcome.
func VerifyEndpoint(ctx context.Context, url, model string) error {
	return checked(ctx, newEndpoint(url, model), true)
}

// Source embeds text for memory search.
type Source interface {
	Embed(text string) []float32
}

// planTTL is how long a process keeps a resolved plan: a long-running
// listener follows a changed setting within it.
const planTTL = time.Minute

var active struct {
	mu   sync.Mutex
	plan Plan
	src  Source
	at   time.Time
}

// Current is the plan in force, resolved at most once a planTTL.
func Current() Plan {
	active.mu.Lock()
	defer active.mu.Unlock()
	refresh()
	return active.plan
}

// Active is the source the plan in force picks; nil when search is
// keywords only (the plan is off, or the model is not downloaded).
func Active() Source {
	active.mu.Lock()
	defer active.mu.Unlock()
	refresh()
	switch active.plan.Backend {
	case Endpoint:
		return active.src
	case CPU:
		e, err := Default()
		if err != nil {
			return nil
		}
		return e
	}
	return nil
}

// refresh resolves the plan when it is stale; active.mu is held.
func refresh() {
	if !active.at.IsZero() && time.Since(active.at) < planTTL {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	p := Resolve(ctx)
	active.plan, active.at, active.src = p, time.Now(), nil
	SetThreads(p.Threads)
	if p.Backend == Endpoint {
		active.src = &remote{ep: newEndpoint(p.URL, p.Model), fallback: !p.Machine.Small()}
	}
}

// Reset drops the plan this process holds.
func Reset() {
	active.mu.Lock()
	active.at = time.Time{}
	active.mu.Unlock()
}
