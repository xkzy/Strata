package resourcemanager

import (
	"context"
	"testing"
	"time"
)

func BenchmarkControllerObserve(b *testing.B) {
	c := newTestController()
	c.Admit(context.Background(), 100)
	at := time.Unix(20000, 0)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		at = at.Add(time.Second)
		c.Observe(healthy(at, 1))
	}
}

func BenchmarkControllerAdmitRelease(b *testing.B) {
	cfg := ConfigForPolicy(PolicyMaxThroughput)
	cfg.Enabled = true
	c := NewController(cfg)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a := c.Admit(context.Background(), 128)
		if !a.Admitted {
			b.Fatal(a.Reason)
		}
		c.Release(a.Ticket, time.Now())
	}
}

func BenchmarkControllerQueueHandoff(b *testing.B) {
	cfg := ConfigForPolicy(PolicyBalanced)
	cfg.Enabled = true
	cfg.StartConcurrency, cfg.MinConcurrency, cfg.MaxConcurrency = 1, 1, 2
	cfg.StartQueue, cfg.MinQueue, cfg.MaxQueue = 64, 1, 64
	c := NewController(cfg)
	a := c.Admit(context.Background(), 10)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		go c.Release(a.Ticket, time.Now())
		next := c.Admit(context.Background(), 10)
		if !next.Admitted {
			b.Fatal(next.Reason)
		}
		a = next
	}
	b.StopTimer()
	c.Release(a.Ticket, time.Now())
}
