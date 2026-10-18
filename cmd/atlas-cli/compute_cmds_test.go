package main

import (
	"os"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/unix"

	"github.com/tryatlas/cli/internal/clistate"
	"github.com/tryatlas/cli/internal/embed"
)

func TestDescribePlan(t *testing.T) {
	t.Setenv("ATLAS_MODELS_DIR", t.TempDir())
	m := embed.Machine{CPUs: 8, Memory: 16 << 30}
	for _, tt := range []struct {
		p    embed.Plan
		want string
	}{
		{embed.Plan{Backend: embed.Off, Note: "as set"}, "keywords only · as set"},
		{embed.Plan{Backend: embed.CPU, Machine: m}, "keywords only · the model is not downloaded"},
		{embed.Plan{Backend: embed.Endpoint, URL: "http://gpu:8080/v1", Model: "m", Note: "verified"}, "meaning and keywords · m at gpu:8080 · verified"},
	} {
		if got := describePlan(tt.p); !strings.HasPrefix(got, tt.want) {
			t.Errorf("describePlan(%+v) = %q, want %q…", tt.p, got, tt.want)
		}
	}
}

func TestLoopback(t *testing.T) {
	for u, want := range map[string]bool{
		"http://127.0.0.1:11434/v1": true, "http://localhost:8080": true, "http://[::1]:80/v1": true,
		"http://gpu-box:8080/v1": false, "https://10.0.0.5/v1": false,
	} {
		if loopback(u) != want {
			t.Errorf("loopback(%q) = %v", u, !want)
		}
	}
}

// A pass in progress is seen through the Project's index lock, and the
// pending mark outlives it.
func TestIndexRunningAndPending(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	root := t.TempDir()
	if indexRunning(root) || indexPending(root) {
		t.Fatal("a fresh Project has a pass or a mark")
	}
	f, err := os.OpenFile(clistate.Path(root, "index.lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB); err != nil {
		t.Fatal(err)
	}
	if !indexRunning(root) {
		t.Fatal("a held lock is not a running pass")
	}
	_ = os.WriteFile(clistate.Path(root, indexPendingFile), nil, 0o600)
	if !indexPending(root) {
		t.Fatal("the mark is not seen")
	}
	_ = unix.Flock(int(f.Fd()), unix.LOCK_UN)
	if indexRunning(root) {
		t.Fatal("a released lock is a running pass")
	}
}

func TestComputeSetsAndShows(t *testing.T) {
	t.Setenv("XDG_STATE_HOME", t.TempDir())
	t.Setenv("ATLAS_EMBED", "")
	t.Cleanup(embed.Reset)
	var out, errb strings.Builder
	if code := run([]string{"memory", "compute", "off"}, &out, &errb); code != 0 {
		t.Fatalf("exit %d: %s", code, errb.String())
	}
	if !strings.Contains(out.String(), "Mode:      off") || !strings.Contains(out.String(), "keywords only") {
		t.Fatalf("output %q", out.String())
	}
	if s := embed.SavedSettings(); s.Mode != "off" {
		t.Fatalf("saved %+v", s)
	}
	out.Reset()
	if code := run([]string{"memory", "compute", "--threads", "2"}, &out, &errb); code != 0 {
		t.Fatalf("exit %d: %s", code, errb.String())
	}
	if s := embed.SavedSettings(); s.Mode != "off" || s.Threads != 2 {
		t.Fatalf("threads alone changed the mode: %+v", s)
	}
	if code := run([]string{"memory", "compute", "gpu"}, &out, &errb); code == 0 {
		t.Fatal("a bad mode was accepted")
	}
	if code := run([]string{"memory", "compute", "http://127.0.0.1:1/v1"}, &out, &errb); code == 0 || !strings.Contains(errb.String(), "not saved") {
		t.Fatalf("an unreachable endpoint was saved (%d): %s", code, errb.String())
	}
	if s := embed.SavedSettings(); s.Mode != "off" {
		t.Fatalf("saved %+v", s)
	}
}

// With shared memory off a background run has nothing to index; it still
// consumes the mark, so it ends instead of passing again and again.
func TestBackgroundIndexWithMemoryOffEnds(t *testing.T) {
	root := gitRepo(t)
	if code, _, e := runCmd(t, "init", "--local"); code != 0 {
		t.Fatal(e)
	}
	if code, _, e := runCmd(t, "memory", "disable"); code != 0 {
		t.Fatal(e)
	}
	t.Setenv("ATLAS_EMBED", "off")
	t.Cleanup(embed.Reset)
	_ = os.WriteFile(clistate.Path(root, indexPendingFile), nil, 0o600)
	done := make(chan int, 1)
	go func() {
		code, _, _ := runCmd(t, "memory", "index", "--background")
		done <- code
	}()
	select {
	case code := <-done:
		if code != 0 {
			t.Fatalf("exit %d", code)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("the background run did not end")
	}
	if indexPending(root) {
		t.Fatal("the mark was left")
	}
}
