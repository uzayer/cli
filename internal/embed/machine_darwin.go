package embed

import "golang.org/x/sys/unix"

func physicalMemory() uint64 {
	n, _ := unix.SysctlUint64("hw.memsize")
	return n
}
