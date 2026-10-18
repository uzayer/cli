package embed

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// isolate gives a test its own CLI state directory and no overrides, and
// points Ollama at a closed port.
func isolate(t *testing.T) {
	t.Helper()
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	t.Setenv("ATLAS_EMBED", "")
	t.Setenv("ATLAS_EMBED_MODEL", "")
	t.Setenv("ATLAS_EMBED_THREADS", "")
	old := ollamaURL
	ollamaURL = "http://127.0.0.1:1/v1"
	t.Cleanup(func() { ollamaURL = old; Reset() })
}

// fakeEndpoint serves the probe vectors, each turned by tweak, answering
// in reverse order to check that indexes are honoured, and counts the
// requests.
func fakeEndpoint(t *testing.T, tweak func(i int, v []float32)) (*httptest.Server, *int) {
	t.Helper()
	var calls int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.NotFound(w, r)
			return
		}
		calls++
		var req struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		type item struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}
		var data []item
		for i := len(req.Input) - 1; i >= 0; i-- {
			v := make([]float32, Dim)
			for j, p := range probes {
				if p == req.Input[i] {
					v = decodeProbe(probeVectors[j])
					if tweak != nil {
						tweak(j, v)
					}
				}
			}
			for k := range v {
				v[k] *= 3 // unnormalised, as some servers answer
			}
			data = append(data, item{Index: i, Embedding: v})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

func TestParseMode(t *testing.T) {
	for in, want := range map[string]string{
		"": "", "auto": "", "OFF": "off", "keywords": "off", "cpu": "cpu", "ollama": "ollama",
		"http://gpu:8080/v1":                "http://gpu:8080/v1",
		"http://gpu:8080/v1/":               "http://gpu:8080/v1",
		"https://gpu.example/v1/embeddings": "https://gpu.example/v1",
	} {
		got, err := ParseMode(in)
		if err != nil || got != want {
			t.Errorf("ParseMode(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	for _, bad := range []string{"gpu", "ftp://x", "http://"} {
		if _, err := ParseMode(bad); err == nil {
			t.Errorf("ParseMode(%q) accepted", bad)
		}
	}
}

func TestChoose(t *testing.T) {
	big := Machine{CPUs: 8, Memory: 16 << 30}
	tests := []struct {
		name    string
		s       Settings
		m       Machine
		backend Backend
		bg      int
	}{
		{"auto on a laptop", Settings{}, big, CPU, 4},
		{"auto on one core", Settings{}, Machine{CPUs: 1, Memory: 4 << 30}, Off, 1},
		{"auto under 2 GB", Settings{}, Machine{CPUs: 4, Memory: 1 << 30}, Off, 2},
		{"auto, memory unknown", Settings{}, Machine{CPUs: 2}, CPU, 1},
		{"cpu on one core", Settings{Mode: "cpu"}, Machine{CPUs: 1, Memory: 1 << 30}, CPU, 1},
		{"threads cap the background", Settings{Threads: 2}, big, CPU, 2},
		{"off", Settings{Mode: "off"}, big, Off, 4},
		{"ollama", Settings{Mode: "ollama"}, big, Endpoint, 4},
		{"url", Settings{Mode: "http://gpu:8080/v1"}, big, Endpoint, 4},
		{"a bad mode is auto", Settings{Mode: "gpu"}, big, CPU, 4},
	}
	for _, tt := range tests {
		p := Choose(tt.s, tt.m)
		if p.Backend != tt.backend || p.Background != tt.bg {
			t.Errorf("%s: backend %s, background %d; want %s, %d (%+v)", tt.name, p.Backend, p.Background, tt.backend, tt.bg, p)
		}
	}
	if p := Choose(Settings{Mode: "ollama"}, big); p.Model != OllamaModel {
		t.Errorf("ollama model %q", p.Model)
	}
	if p := Choose(Settings{Mode: "http://gpu/v1"}, big); p.Model != Repo {
		t.Errorf("endpoint model %q", p.Model)
	}
	if p := Choose(Settings{}, Machine{CPUs: 1}); !strings.Contains(p.Note, "a small machine (1 core)") {
		t.Errorf("note %q", p.Note)
	}
	if p := Choose(Settings{Mode: "warp"}, Machine{CPUs: 1}); !strings.HasPrefix(p.Note, `"warp" is not a mode, so automatic; a small machine`) {
		t.Errorf("a bad mode on a small machine: note %q", p.Note)
	}
}

func TestSettingsOverrides(t *testing.T) {
	isolate(t)
	if err := SaveSettings(Settings{Mode: "cpu", Threads: 3}); err != nil {
		t.Fatal(err)
	}
	if s := LoadSettings(); s.Mode != "cpu" || s.Threads != 3 {
		t.Fatalf("saved %+v", s)
	}
	t.Setenv("ATLAS_EMBED", "off")
	t.Setenv("ATLAS_EMBED_THREADS", "1")
	if s := LoadSettings(); s.Mode != "off" || s.Threads != 1 {
		t.Fatalf("overridden %+v", s)
	}
	if s := SavedSettings(); s.Mode != "cpu" {
		t.Fatalf("the file changed: %+v", s)
	}
}

func TestResolveUsesAMatchingEndpoint(t *testing.T) {
	isolate(t)
	srv, calls := fakeEndpoint(t, nil)
	t.Setenv("ATLAS_EMBED", srv.URL+"/v1")
	p := Resolve(context.Background())
	if p.Backend != Endpoint || p.URL != srv.URL+"/v1" {
		t.Fatalf("plan %+v", p)
	}
	if *calls != 1 {
		t.Fatalf("probed %d times", *calls)
	}
	// The verdict is kept: the next resolve does not probe again.
	Resolve(context.Background())
	if *calls != 1 {
		t.Fatalf("probed again: %d times", *calls)
	}
	src := &remote{ep: newEndpoint(p.URL, p.Model)}
	v := src.Embed(probes[1])
	if c := Cosine(v, decodeProbe(probeVectors[1])); c < 0.9999 {
		t.Fatalf("cosine %v", c)
	}
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	if n < 0.999 || n > 1.001 {
		t.Fatalf("not a unit vector: |v|² = %v", n)
	}
}

func TestResolveRejectsAnotherModel(t *testing.T) {
	isolate(t)
	// An endpoint that truncates the long probe sooner embeds it otherwise.
	srv, calls := fakeEndpoint(t, func(i int, v []float32) {
		if i == 3 {
			for k := range v[:Dim/4] {
				v[k] = -v[k]
			}
		}
	})
	t.Setenv("ATLAS_EMBED", srv.URL+"/v1")
	p := Resolve(context.Background())
	if p.Backend != CPU || !strings.Contains(p.Note, "differ from the app's model") {
		t.Fatalf("plan %+v", p)
	}
	Resolve(context.Background())
	if *calls != 1 {
		t.Fatalf("a rejected endpoint was probed %d times within a day", *calls)
	}
	// On a small machine the fallback is keywords only.
	if fb := Choose(Settings{}, Machine{CPUs: 1}); fb.Backend != Off {
		t.Fatalf("fallback %+v", fb)
	}
}

// A set endpoint that stopped after it was verified is not claimed as the
// source: the record of its match does not say it is up.
func TestResolveStoppedEndpoint(t *testing.T) {
	isolate(t)
	srv, _ := fakeEndpoint(t, nil)
	if err := VerifyEndpoint(context.Background(), srv.URL+"/v1", Repo); err != nil {
		t.Fatal(err)
	}
	t.Setenv("ATLAS_EMBED", srv.URL+"/v1")
	if p := Resolve(context.Background()); p.Backend != Endpoint {
		t.Fatalf("plan %+v", p)
	}
	srv.Close()
	if p := Resolve(context.Background()); p.Backend == Endpoint || !strings.Contains(p.Note, "cannot reach it") {
		t.Fatalf("a stopped endpoint: plan %+v", p)
	}
}

// A local Ollama whose all-minilm differs is named in the note, so its
// owner knows why it is not used.
func TestAutoSaysWhyOllamaIsNotUsed(t *testing.T) {
	isolate(t)
	wrong, _ := fakeEndpoint(t, func(_ int, v []float32) {
		for k := range v {
			v[k] = -v[k]
		}
	})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			_, _ = w.Write([]byte(`{"models":[{"name":"all-minilm"}]}`))
			return
		}
		wrong.Config.Handler.ServeHTTP(w, r)
	}))
	defer srv.Close()
	ollamaURL = srv.URL + "/v1"
	if p := Resolve(context.Background()); p.Backend == Endpoint || !strings.Contains(p.Note, "the local Ollama's all-minilm is not used") {
		t.Fatalf("plan %+v", p)
	}
}

// A refused connection is found by the dial alone, and costs nothing to
// try again: no verdict is kept for it.
func TestResolveRefusedEndpointIsNotRecorded(t *testing.T) {
	isolate(t)
	t.Setenv("ATLAS_EMBED", "http://127.0.0.1:1/v1")
	p := Resolve(context.Background())
	if p.Backend == Endpoint || !strings.Contains(p.Note, "cannot reach it") {
		t.Fatalf("plan %+v", p)
	}
	if _, err := os.Stat(filepath.Join(os.Getenv("XDG_STATE_HOME"), "atlas-cli", "compute-state.json")); err == nil {
		t.Fatal("a refused endpoint left a verdict")
	}
}

// A hung endpoint is remembered as down for a while, so the next process
// does not wait for it again; atlas-cli memory compute checks afresh.
func TestDownEndpointIsRememberedBriefly(t *testing.T) {
	isolate(t)
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits++
		http.Error(w, "warming up", http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	t.Setenv("ATLAS_EMBED", srv.URL)
	Resolve(context.Background())
	Resolve(context.Background())
	if hits != 1 {
		t.Fatalf("a down endpoint was probed %d times within two minutes", hits)
	}
	if err := VerifyEndpoint(context.Background(), srv.URL, Repo); err == nil || hits != 2 {
		t.Fatalf("an explicit check used the record: %v, %d probes", err, hits)
	}
}

// A rejected endpoint that was fixed is accepted by an explicit check
// within the day the rejection is kept.
func TestExplicitCheckRetriesARejection(t *testing.T) {
	isolate(t)
	bad := true
	good, _ := fakeEndpoint(t, nil)
	wrong, _ := fakeEndpoint(t, func(_ int, v []float32) {
		for k := range v {
			v[k] = -v[k]
		}
	})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if bad {
			wrong.Config.Handler.ServeHTTP(w, r)
			return
		}
		good.Config.Handler.ServeHTTP(w, r)
	}))
	defer srv.Close()
	if err := VerifyEndpoint(context.Background(), srv.URL+"/v1", Repo); err == nil {
		t.Fatal("a wrong endpoint was accepted")
	}
	bad = false
	if err := VerifyEndpoint(context.Background(), srv.URL+"/v1", Repo); err != nil {
		t.Fatalf("the fixed endpoint was refused: %v", err)
	}
	t.Setenv("ATLAS_EMBED", srv.URL+"/v1")
	if p := Resolve(context.Background()); p.Backend != Endpoint {
		t.Fatalf("plan %+v", p)
	}
}

func TestAutoFindsLocalOllama(t *testing.T) {
	isolate(t)
	emb, _ := fakeEndpoint(t, nil)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			_, _ = w.Write([]byte(`{"models":[{"name":"llama3:8b"},{"name":"all-minilm:33m"},{"name":"all-minilm:latest"}]}`))
			return
		}
		emb.Config.Handler.ServeHTTP(w, r)
	}))
	defer srv.Close()
	ollamaURL = srv.URL + "/v1"
	p := Resolve(context.Background())
	if p.Backend != Endpoint || p.Model != "all-minilm:latest" || p.Mode != "auto" {
		t.Fatalf("plan %+v", p)
	}
	// Stopped, it is not used, though its match is on record.
	srv.Close()
	if p := Resolve(context.Background()); p.Backend == Endpoint {
		t.Fatalf("a stopped Ollama: plan %+v", p)
	}
	// Set off, it is not used.
	t.Setenv("ATLAS_EMBED", "off")
	if p := Resolve(context.Background()); p.Backend != Off {
		t.Fatalf("plan %+v", p)
	}
}

func TestRemoteFallsBack(t *testing.T) {
	isolate(t)
	t.Setenv("ATLAS_MODELS_DIR", t.TempDir()) // no model here
	Forget()
	r := &remote{ep: newEndpoint("http://127.0.0.1:1/v1", Repo), fallback: true}
	if v := r.Embed("anything"); v != nil {
		t.Fatalf("no endpoint and no model, got a vector")
	}
	if !r.down.Load() {
		t.Fatal("a failed endpoint is tried again in the same process")
	}
	if v := r.Embed("  "); len(v) != Dim {
		t.Fatal("an empty text is the zero vector")
	}
}

func TestEndpointRejectsOtherDimensions(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"data":[{"index":0,"embedding":[0.1,0.2,0.3]}]}`))
	}))
	defer srv.Close()
	_, err := newEndpoint(srv.URL, "m").embed(context.Background(), []string{"x"})
	if err == nil || !strings.Contains(err.Error(), "3 dimensions") {
		t.Fatalf("err %v", err)
	}
}

func TestProbeVectorsAreCurrent(t *testing.T) {
	if !Available() {
		t.Skip("needs the model (ATLAS_MODELS_DIR)")
	}
	e, err := Default()
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range probes {
		if c := Cosine(e.Embed(p), decodeProbe(probeVectors[i])); c < 0.9999 {
			t.Errorf("probe %d: cosine %v", i, c)
		}
	}
}
