//go:build !darwin && !linux

package embed

func physicalMemory() uint64 { return 0 }
