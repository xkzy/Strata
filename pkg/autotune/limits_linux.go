//go:build linux

package autotune

import "syscall"

// rlimitMemlock is RLIMIT_MEMLOCK on Linux (resource 8 on every architecture). The syscall package does not
// export the name, so the number is used with this note rather than pulling in a new dependency.
const rlimitMemlock = 8

// pinnedMemoryLimit is RLIMIT_MEMLOCK in bytes (0: unlimited or a limit too large to matter). Pinned host
// staging and GPU mapped buffers draw on it; a plan that needs more than this is infeasible.
func pinnedMemoryLimit() uint64 {
	var r syscall.Rlimit
	if err := syscall.Getrlimit(rlimitMemlock, &r); err != nil {
		return 0
	}
	if r.Cur == ^uint64(0) || r.Cur > 1<<40 {
		return 0
	}
	return r.Cur
}
