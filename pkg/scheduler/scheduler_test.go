package scheduler

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func ctxFor(tenant string, prio int) context.Context {
	return WithTicket(context.Background(), Ticket{Tenant: tenant, Priority: prio})
}

// waitQueued blocks until n requests wait.
func waitQueued(t *testing.T, s *Scheduler, n int) {
	t.Helper()
	for i := 0; i < 500; i++ {
		s.mu.Lock()
		q := len(s.queue)
		s.mu.Unlock()
		if q == n {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("queue never reached %d", n)
}

// enqueue starts an Acquire that records its name in order when it gets the slot and then releases at once.
func enqueue(s *Scheduler, ctx context.Context, name string, order *[]string, mu *sync.Mutex, wg *sync.WaitGroup) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		rel, err := s.Acquire(ctx)
		if err != nil {
			return
		}
		mu.Lock()
		*order = append(*order, name)
		mu.Unlock()
		rel()
	}()
}

func TestPriorityThenFairnessThenArrival(t *testing.T) {
	s := New(1, 8)
	hold, err := s.Acquire(ctxFor("x", 5))
	if err != nil {
		t.Fatal(err)
	}
	var order []string
	var mu sync.Mutex
	var wg sync.WaitGroup
	// arrival order: a1, a2 (tenant a), b1 (tenant b), c1 urgent; a has already been served more than b
	s.served["a"] = 3
	enqueue(s, ctxFor("a", 5), "a1", &order, &mu, &wg)
	waitQueued(t, s, 1)
	enqueue(s, ctxFor("a", 5), "a2", &order, &mu, &wg)
	waitQueued(t, s, 2)
	enqueue(s, ctxFor("b", 5), "b1", &order, &mu, &wg)
	waitQueued(t, s, 3)
	enqueue(s, ctxFor("c", 9), "c1", &order, &mu, &wg)
	waitQueued(t, s, 4)
	hold()
	wg.Wait()
	want := []string{"c1", "b1", "a1", "a2"} // urgent first, then the less-served tenant, then arrival
	if len(order) != 4 {
		t.Fatalf("order = %v", order)
	}
	for i := range want {
		if order[i] != want[i] {
			t.Fatalf("order = %v, want %v", order, want)
		}
	}
}

func TestAgingLetsALowPriorityRequestThrough(t *testing.T) {
	s := New(1, 8)
	now := time.Unix(1000, 0)
	s.now = func() time.Time { return now }
	hold, _ := s.Acquire(ctxFor("x", 5))
	low := &waiter{ticket: Ticket{Tenant: "low", Priority: 0}, since: now, granted: make(chan struct{})}
	s.queue = append(s.queue, low)
	now = now.Add(5 * agingStep) // the low one has waited long enough to reach priority 5
	fresh := &waiter{ticket: Ticket{Tenant: "high", Priority: 5}, since: now, granted: make(chan struct{})}
	s.queue = append(s.queue, fresh)
	hold()
	select {
	case <-low.granted:
	default:
		t.Fatal("the aged request was not served first")
	}
}

func TestQueueFullRefusesWithRetryAfter(t *testing.T) {
	s := New(1, 2)
	hold, _ := s.Acquire(ctxFor("x", 5))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	for i := 0; i < 2; i++ {
		go s.Acquire(ctx)
	}
	waitQueued(t, s, 2)
	if ra, full := s.Full(); !full || ra < 1 {
		t.Fatalf("Full = %d %v", ra, full)
	}
	_, err := s.Acquire(ctxFor("y", 5))
	var qf *ErrQueueFull
	if !errors.As(err, &qf) || qf.Depth != 2 || qf.RetryAfter < 1 {
		t.Fatalf("err = %v", err)
	}
	if s.Stats()["rejected"] != 1 {
		t.Errorf("stats = %v", s.Stats())
	}
	hold()
}

func TestCancelWhileWaitingLeavesTheQueue(t *testing.T) {
	s := New(1, 4)
	hold, _ := s.Acquire(ctxFor("x", 5))
	ctx, cancel := context.WithCancel(ctxFor("y", 5))
	done := make(chan error, 1)
	go func() { _, err := s.Acquire(ctx); done <- err }()
	waitQueued(t, s, 1)
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v", err)
	}
	waitQueued(t, s, 0)
	hold()
	rel, err := s.Acquire(ctxFor("z", 5)) // the slot is free again
	if err != nil {
		t.Fatal(err)
	}
	rel()
	if st := s.Stats(); st["running"] != 0 || st["queued"] != 0 {
		t.Errorf("stats = %v", st)
	}
}

func TestSlotsRunTogetherAndReleaseIsIdempotent(t *testing.T) {
	s := New(2, 4)
	a, _ := s.Acquire(ctxFor("a", 5))
	b, _ := s.Acquire(ctxFor("b", 5))
	if s.Stats()["running"] != 2 {
		t.Fatalf("stats = %v", s.Stats())
	}
	a()
	a() // a second release must not free b's slot
	if s.Stats()["running"] != 1 {
		t.Fatalf("stats = %v", s.Stats())
	}
	b()
}
