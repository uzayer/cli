package embed

import (
	"os"
	"path/filepath"
	"testing"
)

func write(t *testing.T, path, s string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestMeminfoTotal(t *testing.T) {
	p := filepath.Join(t.TempDir(), "meminfo")
	write(t, p, "MemTotal:        2014860 kB\nMemFree:          112000 kB\n")
	if got := meminfoTotal(p); got != 2014860<<10 {
		t.Fatalf("got %d", got)
	}
	if meminfoTotal(filepath.Join(t.TempDir(), "none")) != 0 {
		t.Fatal("a missing file is unknown")
	}
}

func TestCgroupMemoryLimit(t *testing.T) {
	t.Run("v2 at the process's path", func(t *testing.T) {
		d := t.TempDir()
		write(t, filepath.Join(d, "self"), "0::/system.slice/atlas.service\n")
		write(t, filepath.Join(d, "fs/system.slice/atlas.service/memory.max"), "1073741824\n")
		if got := cgroupMemoryLimit(filepath.Join(d, "self"), filepath.Join(d, "fs")); got != 1<<30 {
			t.Fatalf("got %d", got)
		}
	})
	t.Run("v2 unlimited in a container", func(t *testing.T) {
		d := t.TempDir()
		write(t, filepath.Join(d, "self"), "0::/\n")
		write(t, filepath.Join(d, "fs/memory.max"), "max\n")
		if got := cgroupMemoryLimit(filepath.Join(d, "self"), filepath.Join(d, "fs")); got != 0 {
			t.Fatalf("got %d", got)
		}
	})
	t.Run("v1", func(t *testing.T) {
		d := t.TempDir()
		write(t, filepath.Join(d, "self"), "12:cpu,cpuacct:/docker/abc\n11:memory:/docker/abc\n")
		write(t, filepath.Join(d, "fs/memory/docker/abc/memory.limit_in_bytes"), "536870912\n")
		if got := cgroupMemoryLimit(filepath.Join(d, "self"), filepath.Join(d, "fs")); got != 512<<20 {
			t.Fatalf("got %d", got)
		}
	})
	t.Run("v1 unlimited", func(t *testing.T) {
		d := t.TempDir()
		write(t, filepath.Join(d, "self"), "11:memory:/\n")
		write(t, filepath.Join(d, "fs/memory/memory.limit_in_bytes"), "9223372036854771712\n")
		if got := cgroupMemoryLimit(filepath.Join(d, "self"), filepath.Join(d, "fs")); got != 0 {
			t.Fatalf("got %d", got)
		}
	})
}

func TestMachine(t *testing.T) {
	for _, tt := range []struct {
		m     Machine
		small bool
		bg    int
		s     string
	}{
		{Machine{CPUs: 1, Memory: 8 << 30}, true, 1, "1 core, 8.0 GB"},
		{Machine{CPUs: 2, Memory: 1 << 30}, true, 1, "2 cores, 1.0 GB"},
		{Machine{CPUs: 2, Memory: 2 << 30}, false, 1, "2 cores, 2.0 GB"},
		{Machine{CPUs: 16}, false, 8, "16 cores"},
	} {
		if tt.m.Small() != tt.small || tt.m.BackgroundThreads() != tt.bg || tt.m.String() != tt.s {
			t.Errorf("%+v: small %v, background %d, %q", tt.m, tt.m.Small(), tt.m.BackgroundThreads(), tt.m.String())
		}
	}
	if m := DetectMachine(); m.CPUs < 1 {
		t.Fatalf("detected %+v", m)
	}
}
