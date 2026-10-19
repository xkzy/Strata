//go:build !linux

package autotune

// pinnedMemoryLimit is unknown on this platform; 0 means "not established", and the solver does not treat it as
// unlimited (it simply cannot prove a pinning budget, so it leaves pinning at the runtime default).
func pinnedMemoryLimit() uint64 { return 0 }
