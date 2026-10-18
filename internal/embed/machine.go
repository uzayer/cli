package embed

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
)

// Machine is what memory search sizes itself to.
type Machine struct {
	// CPUs is the cores this process may use: GOMAXPROCS as the process
	// started, which follows a container's CPU limit.
	CPUs int
	// Memory is physical memory in bytes, or a container's limit where that
	// is lower; 0 when unknown.
	Memory uint64
}

var startProcs = runtime.GOMAXPROCS(0)

// DetectMachine reads this machine.
func DetectMachine() Machine {
	return Machine{CPUs: startProcs, Memory: physicalMemory()}
}

// Small machines (one core, or under 2 GiB) search on keywords unless told
// otherwise: a forward pass would hold a core for seconds per document.
const (
	smallCPUs   = 2
	smallMemory = 2 << 30
)

// Small reports a machine below either line.
func (m Machine) Small() bool {
	return m.CPUs < smallCPUs || m.Memory > 0 && m.Memory < smallMemory
}

// BackgroundThreads is what a background pass embeds on: one core on two
// or fewer, else half of them.
func (m Machine) BackgroundThreads() int {
	if m.CPUs <= 2 {
		return 1
	}
	return m.CPUs / 2
}

func (m Machine) String() string {
	cores := "1 core"
	if m.CPUs != 1 {
		cores = fmt.Sprintf("%d cores", m.CPUs)
	}
	if m.Memory == 0 {
		return cores
	}
	return fmt.Sprintf("%s, %.1f GB", cores, float64(m.Memory)/(1<<30))
}

// meminfoTotal is MemTotal from /proc/meminfo, in bytes.
func meminfoTotal(path string) uint64 {
	f, err := os.Open(path) //nolint:gosec // G304: a fixed /proc path
	if err != nil {
		return 0
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if rest, ok := strings.CutPrefix(sc.Text(), "MemTotal:"); ok {
			kb, err := strconv.ParseUint(strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(rest), "kB")), 10, 64)
			if err == nil {
				return kb << 10
			}
		}
	}
	return 0
}

// cgroupMemoryLimit is the memory limit of this process's cgroup (v2
// memory.max, else v1 memory.limit_in_bytes); 0 when there is none.
func cgroupMemoryLimit(self, root string) uint64 {
	b, _ := os.ReadFile(self) //nolint:gosec // G304: a fixed /proc path
	var paths []string
	for _, l := range bytes.Split(b, []byte("\n")) {
		parts := strings.SplitN(string(l), ":", 3)
		if len(parts) != 3 {
			continue
		}
		switch {
		case parts[0] == "0" && parts[1] == "":
			paths = append(paths, filepath.Join(root, parts[2], "memory.max"))
		case strings.Contains(","+parts[1]+",", ",memory,"):
			paths = append(paths, filepath.Join(root, "memory", parts[2], "memory.limit_in_bytes"))
		}
	}
	// Inside a container the cgroup is often mounted as the root itself.
	paths = append(paths, filepath.Join(root, "memory.max"), filepath.Join(root, "memory", "memory.limit_in_bytes"))
	for _, p := range paths {
		b, err := os.ReadFile(p) //nolint:gosec // G304: a cgroup file
		if err != nil {
			continue
		}
		s := strings.TrimSpace(string(b))
		if s == "max" {
			return 0
		}
		n, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			continue
		}
		if n >= 1<<60 { // v1's "unlimited" is a huge page-aligned number
			return 0
		}
		return n
	}
	return 0
}
