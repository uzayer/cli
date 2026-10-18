package main

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"time"

	"golang.org/x/sys/unix"

	"github.com/tryatlas/cli/internal/clistate"
	"github.com/tryatlas/cli/internal/embed"
)

// backgroundNice is the nice value of a detached pass: below the agent and
// the shell, so a small server stays responsive while it embeds.
const backgroundNice = 10

// backgroundWait bounds how long a detached pass queues behind another
// Project's; its pending mark stays, so the next pass does its work.
const backgroundWait = 10 * time.Minute

// background readies a detached pass (atlas-cli memory index --background,
// an extraction's reindex) to share the machine: a lower priority, the
// cores the compute plan leaves a background pass, and one pass at a time
// across every Project here. ok is false when the turn did not come.
func background(ctx context.Context) (release func(), ok bool) {
	yield()
	if err := os.MkdirAll(clistate.Base(), 0o700); err != nil {
		return func() {}, true
	}
	f, err := os.OpenFile(filepath.Join(clistate.Base(), "embed.lock"), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return func() {}, true // no lock file: run unqueued rather than not at all
	}
	ctx, cancel := context.WithTimeout(ctx, backgroundWait)
	defer cancel()
	for unix.Flock(int(f.Fd()), unix.LOCK_EX|unix.LOCK_NB) != nil {
		select {
		case <-ctx.Done():
			_ = f.Close()
			return func() {}, false
		case <-time.After(250 * time.Millisecond):
		}
	}
	return func() {
		_ = unix.Flock(int(f.Fd()), unix.LOCK_UN)
		_ = f.Close()
	}, true
}

// yield lowers this process's priority and caps its cores at what the
// compute plan leaves a background pass.
func yield() {
	lowerPriority()
	if p := embed.Current(); p.Backend == embed.CPU {
		runtime.GOMAXPROCS(p.Background)
	}
}

// lowerPriority renices this process. On Linux a nice value is a thread's,
// so every thread the runtime has started is reniced; threads it starts
// later inherit it.
func lowerPriority() {
	if runtime.GOOS == "linux" {
		if ents, err := os.ReadDir("/proc/self/task"); err == nil {
			for _, e := range ents {
				if tid, err := strconv.Atoi(e.Name()); err == nil {
					_ = unix.Setpriority(unix.PRIO_PROCESS, tid, backgroundNice)
				}
			}
			return
		}
	}
	_ = unix.Setpriority(unix.PRIO_PROCESS, 0, backgroundNice)
}
