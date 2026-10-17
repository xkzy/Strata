// Package scheduler decides which waiting request gets the engine next. The engine runs a fixed number of requests at
// once (one session, or the --batch slots), so the rest wait here: in a bounded queue, highest priority first, then the
// tenant that has been served least, then the oldest. A request that waits long gains priority so a low-priority one
// is never starved, and a full queue refuses new requests at once with an estimate of when to come back.
package scheduler

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
)

const (
	// DefaultPriority is a request that sent no priority.
	DefaultPriority = 5
	// MaxPriority is the highest priority a caller can ask for.
	MaxPriority = 10
	// agingStep: a waiting request gains one priority point for every agingStep it waits.
	agingStep = 20 * time.Second
	// DefaultQueueLimit is how many requests may wait (besides the ones running).
	DefaultQueueLimit = 32
)

// Ticket says who a request is and how urgent it is.
type Ticket struct {
	Tenant   string
	Priority int // 0..MaxPriority, higher first
}

type ticketKey struct{}

// WithTicket attaches t to ctx; Acquire reads it. A context without one is the "default" tenant at DefaultPriority.
func WithTicket(ctx context.Context, t Ticket) context.Context {
	if t.Priority < 0 {
		t.Priority = 0
	}
	if t.Priority > MaxPriority {
		t.Priority = MaxPriority
	}
	return context.WithValue(ctx, ticketKey{}, t)
}

func ticketOf(ctx context.Context) Ticket {
	if t, ok := ctx.Value(ticketKey{}).(Ticket); ok {
		return t
	}
	return Ticket{Tenant: "default", Priority: DefaultPriority}
}

// ErrQueueFull is returned when the queue is at its limit. RetryAfter is in seconds.
type ErrQueueFull struct {
	Depth      int
	RetryAfter int
}

func (e *ErrQueueFull) Error() string {
	return fmt.Sprintf("the engine is busy and %d requests are already waiting; try again in about %d s", e.Depth, e.RetryAfter)
}

type waiter struct {
	ticket  Ticket
	since   time.Time
	granted chan struct{}
}

// Scheduler hands out a fixed number of run slots.
type Scheduler struct {
	mu      sync.Mutex
	slots   int
	limit   int
	running int
	queue   []*waiter
	served  map[string]int // per tenant, halved when it grows large so old history fades
	now     func() time.Time

	svcEWMA  float64 // seconds a request holds a slot
	admitted int
	rejected int
	waitSum  float64
	waitMax  float64
}

// New makes a scheduler with `slots` requests running at once and `queueLimit` waiting (0: DefaultQueueLimit).
func New(slots, queueLimit int) *Scheduler {
	if slots < 1 {
		slots = 1
	}
	if queueLimit <= 0 {
		queueLimit = DefaultQueueLimit
	}
	return &Scheduler{slots: slots, limit: queueLimit, served: map[string]int{}, now: time.Now}
}

func (s *Scheduler) retryAfterLocked(depth int) int {
	per := s.svcEWMA
	if per <= 0 {
		per = 5
	}
	sec := int(math.Ceil(float64(depth+1) * per / float64(s.slots)))
	if sec < 1 {
		sec = 1
	}
	if sec > 120 {
		sec = 120
	}
	return sec
}

// Full reports whether a request arriving now would be refused, and when to retry.
func (s *Scheduler) Full() (retryAfter int, full bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.running < s.slots || len(s.queue) < s.limit {
		return 0, false
	}
	return s.retryAfterLocked(len(s.queue)), true
}

// Acquire waits for a slot. release must be called when the request is done. It returns *ErrQueueFull at once when
// the queue is full, or the context's error if the caller leaves while waiting.
func (s *Scheduler) Acquire(ctx context.Context) (release func(), err error) {
	t := ticketOf(ctx)
	s.mu.Lock()
	if s.running < s.slots && len(s.queue) == 0 {
		s.running++
		s.admitLocked(t, 0)
		s.mu.Unlock()
		return s.releaser(), nil
	}
	if len(s.queue) >= s.limit {
		s.rejected++
		e := &ErrQueueFull{Depth: len(s.queue), RetryAfter: s.retryAfterLocked(len(s.queue))}
		s.mu.Unlock()
		return nil, e
	}
	w := &waiter{ticket: t, since: s.now(), granted: make(chan struct{})}
	s.queue = append(s.queue, w)
	s.mu.Unlock()

	select {
	case <-w.granted:
		return s.releaser(), nil
	case <-ctx.Done():
		s.mu.Lock()
		for i, q := range s.queue {
			if q == w {
				s.queue = append(s.queue[:i], s.queue[i+1:]...)
				s.mu.Unlock()
				return nil, context.Cause(ctx)
			}
		}
		s.mu.Unlock()
		// granted just as the caller left: give the slot back
		<-w.granted
		s.releaser()()
		return nil, context.Cause(ctx)
	}
}

func (s *Scheduler) admitLocked(t Ticket, waited float64) {
	s.admitted++
	s.waitSum += waited
	if waited > s.waitMax {
		s.waitMax = waited
	}
	s.served[t.Tenant]++
	if s.served[t.Tenant] > 1024 {
		for k := range s.served {
			s.served[k] /= 2
		}
	}
}

func (s *Scheduler) releaser() func() {
	start := s.now()
	var once sync.Once
	return func() {
		once.Do(func() {
			s.mu.Lock()
			defer s.mu.Unlock()
			d := s.now().Sub(start).Seconds()
			if s.svcEWMA == 0 {
				s.svcEWMA = d
			} else {
				s.svcEWMA = 0.8*s.svcEWMA + 0.2*d
			}
			s.running--
			s.dispatchLocked()
		})
	}
}

// effective is the priority plus what the wait has earned.
func (s *Scheduler) effective(w *waiter, now time.Time) int {
	return w.ticket.Priority + int(now.Sub(w.since)/agingStep)
}

func (s *Scheduler) dispatchLocked() {
	for s.running < s.slots && len(s.queue) > 0 {
		now := s.now()
		best := 0
		for i := 1; i < len(s.queue); i++ {
			a, b := s.queue[i], s.queue[best]
			ea, eb := s.effective(a, now), s.effective(b, now)
			switch {
			case ea != eb:
				if ea > eb {
					best = i
				}
			case s.served[a.ticket.Tenant] != s.served[b.ticket.Tenant]:
				if s.served[a.ticket.Tenant] < s.served[b.ticket.Tenant] {
					best = i
				}
			case a.since.Before(b.since):
				best = i
			}
		}
		w := s.queue[best]
		s.queue = append(s.queue[:best], s.queue[best+1:]...)
		s.running++
		s.admitLocked(w.ticket, now.Sub(w.since).Seconds())
		close(w.granted)
	}
}

// Stats is the scheduler's state for /metrics.
func (s *Scheduler) Stats() map[string]interface{} {
	s.mu.Lock()
	defer s.mu.Unlock()
	avg := 0.0
	if s.admitted > 0 {
		avg = s.waitSum / float64(s.admitted)
	}
	waiting := map[string]int{}
	for _, w := range s.queue {
		waiting[strings.TrimSpace(w.ticket.Tenant)]++
	}
	return map[string]interface{}{
		"slots": s.slots, "running": s.running, "queued": len(s.queue), "queue_limit": s.limit,
		"admitted": s.admitted, "rejected": s.rejected,
		"avg_wait_s": math.Round(avg*100) / 100, "max_wait_s": math.Round(s.waitMax*100) / 100,
		"avg_run_s": math.Round(s.svcEWMA*100) / 100, "waiting_by_tenant": waiting,
	}
}
