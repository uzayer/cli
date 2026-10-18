package embed

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
	"time"
)

// endpoint is an OpenAI-compatible embeddings API: POST {url}/embeddings
// with {"model", "input": [texts]}, answered with {"data": [{"index",
// "embedding"}]}. Ollama (/v1), Hugging Face TEI, vLLM and LM Studio all
// speak it.
type endpoint struct {
	url, model string
	client     *http.Client
}

// endpointTimeout bounds one request; a GPU answers in milliseconds.
const endpointTimeout = 30 * time.Second

func newEndpoint(u, model string) *endpoint {
	tr := &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		DialContext:         (&net.Dialer{Timeout: 2 * time.Second}).DialContext,
		MaxIdleConnsPerHost: 2,
	}
	return &endpoint{url: strings.TrimSuffix(u, "/"), model: model,
		client: &http.Client{Timeout: endpointTimeout, Transport: tr}}
}

// label names the endpoint for a person: the model at its host.
func (e *endpoint) label() string {
	host := e.url
	if u, err := url.Parse(e.url); err == nil && u.Host != "" {
		host = u.Host
	}
	return e.model + " at " + host
}

// reachable dials the endpoint's host: a stopped server is found in well
// under a millisecond on this machine, a lost one within the dial timeout.
func (e *endpoint) reachable(ctx context.Context) error {
	u, err := url.Parse(e.url)
	if err != nil {
		return err
	}
	host := u.Host
	if u.Port() == "" {
		port := "80"
		if u.Scheme == "https" {
			port = "443"
		}
		host = net.JoinHostPort(u.Hostname(), port)
	}
	c, err := (&net.Dialer{Timeout: 2 * time.Second}).DialContext(ctx, "tcp", host)
	if err != nil {
		return fmt.Errorf("cannot reach it (%w)", err)
	}
	return c.Close()
}

// embed returns texts' unit vectors, in order.
func (e *endpoint) embed(ctx context.Context, texts []string) ([][]float32, error) {
	body, err := json.Marshal(map[string]any{"model": e.model, "input": texts})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.url+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("User-Agent", "atlas-cli")
	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("cannot reach it (%w)", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 300))
		return nil, fmt.Errorf("it answered %s: %s", resp.Status, strings.TrimSpace(string(msg)))
	}
	var out struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 64<<20)).Decode(&out); err != nil {
		return nil, fmt.Errorf("its answer is not an embeddings response: %w", err)
	}
	if len(out.Data) != len(texts) {
		return nil, fmt.Errorf("it returned %d vectors for %d texts", len(out.Data), len(texts))
	}
	vs := make([][]float32, len(texts))
	for _, d := range out.Data {
		if d.Index < 0 || d.Index >= len(texts) || vs[d.Index] != nil {
			return nil, fmt.Errorf("it returned a vector for text %d", d.Index)
		}
		if len(d.Embedding) != Dim {
			return nil, fmt.Errorf("its vectors have %d dimensions, not %d: another model", len(d.Embedding), Dim)
		}
		vs[d.Index] = normalise(d.Embedding)
	}
	return vs, nil
}

func normalise(v []float32) []float32 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	n = math.Max(math.Sqrt(n), 1e-12)
	for i := range v {
		v[i] = float32(float64(v[i]) / n)
	}
	return v
}

// remote is an endpoint as a Source. When it fails, the rest of the
// process embeds on this machine's CPU, if the machine is not small and the
// model is here, else not at all (search falls back to keywords).
type remote struct {
	ep       *endpoint
	down     atomic.Bool
	fallback bool
}

func (r *remote) Embed(text string) []float32 {
	if strings.TrimSpace(text) == "" {
		return make([]float32, Dim)
	}
	if !r.down.Load() {
		ctx, cancel := context.WithTimeout(context.Background(), endpointTimeout)
		vs, err := r.ep.embed(ctx, []string{text})
		cancel()
		if err == nil {
			return vs[0]
		}
		r.down.Store(true)
	}
	if !r.fallback {
		return nil
	}
	e, err := Default()
	if err != nil {
		return nil
	}
	return e.Embed(text)
}

// localOllamaModel is the all-minilm model a local Ollama serves, "" when
// there is no Ollama or no such model. A refused connection costs well
// under a millisecond.
func localOllamaModel(ctx context.Context) string {
	ctx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimSuffix(ollamaURL, "/v1")+"/api/tags", nil)
	if err != nil {
		return ""
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()
	var tags struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if resp.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&tags) != nil {
		return ""
	}
	for _, m := range tags.Models {
		// all-minilm:latest and :22m are L6-v2; :33m is L12, which the
		// probes would reject anyway.
		if m.Name == OllamaModel || strings.HasPrefix(m.Name, OllamaModel+":") && !strings.HasSuffix(m.Name, ":33m") {
			return m.Name
		}
	}
	return ""
}

func decodeProbe(s string) []float32 {
	b, _ := base64.StdEncoding.DecodeString(s)
	return Decode(b)
}
