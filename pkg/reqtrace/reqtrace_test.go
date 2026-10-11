package reqtrace

import (
	"context"
	"testing"
)

type memSink struct{ got []string }

func (m *memSink) Add(source, msg string) { m.got = append(m.got, source+": "+msg) }

func TestAddReachesTheSinkInTheContext(t *testing.T) {
	s := &memSink{}
	ctx := With(context.Background(), s)
	Add(ctx, "engine", "GEN sent: %d tokens", 5)
	if len(s.got) != 1 || s.got[0] != "engine: GEN sent: 5 tokens" {
		t.Fatalf("got %v", s.got)
	}
}

func TestAddWithoutASinkDoesNothing(t *testing.T) {
	Add(context.Background(), "engine", "x %d", 1) // must not panic
	Add(nil, "engine", "x")                         //nolint:staticcheck // a nil context is tolerated
}

func TestTheSinkSurvivesAContextWithoutCancel(t *testing.T) {
	s := &memSink{}
	parent, cancel := context.WithCancel(With(context.Background(), s))
	child := context.WithoutCancel(parent)
	cancel()
	Add(child, "rtclient", "still here")
	if len(s.got) != 1 {
		t.Fatalf("the sink must outlive the request's cancellation: %v", s.got)
	}
}
