package telemetry

import "testing"

// BenchmarkSampleOnce measures one full sample: /proc reads, disk, and the nvidia-smi probes. The sampler runs at
// 1 Hz in the server, so a sample that takes longer than its period delays every later reading (and the CPU/VRAM
// values the runtime controller decides on). Run with -benchtime 20x on a host with a GPU.
func BenchmarkSampleOnce(b *testing.B) {
	tc := NewTelemetryCollector()
	tc.sampleOnce() // warm the CPU delta and the shared probes
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc.sampleOnce()
	}
}

// BenchmarkSampleOnceCached measures the same sample once the GPU probes are inside their reuse window, which is
// what the 1 Hz loop pays on most ticks.
func BenchmarkSampleOnceCached(b *testing.B) {
	tc := NewTelemetryCollector()
	tc.sampleOnce()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc.sampleOnce()
	}
}
