package embed

// physicalMemory is MemTotal, or the cgroup's limit where that is lower.
func physicalMemory() uint64 {
	m := meminfoTotal("/proc/meminfo")
	if limit := cgroupMemoryLimit("/proc/self/cgroup", "/sys/fs/cgroup"); limit > 0 && (m == 0 || limit < m) {
		m = limit
	}
	return m
}
