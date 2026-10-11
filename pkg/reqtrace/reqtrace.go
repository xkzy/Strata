// Package reqtrace lets any layer add a timestamped debug event to the request it is working for.
//
// The server puts a Sink into the request's context; the sidecar client, the engine IPC and the runtime's own trace add
// events to it, and the Monitor shows them. Without a Sink in the context every call does nothing.
package reqtrace

import (
	"context"
	"fmt"
)

// Sink receives the events of one request.
type Sink interface {
	Add(source, msg string)
}

type key struct{}

// With returns ctx carrying s.
func With(ctx context.Context, s Sink) context.Context { return context.WithValue(ctx, key{}, s) }

// From returns the Sink in ctx, or nil.
func From(ctx context.Context) Sink {
	if ctx == nil {
		return nil
	}
	s, _ := ctx.Value(key{}).(Sink)
	return s
}

// Add records one event for the request in ctx; source names the layer ("server", "rtclient", "engine", "runtime").
func Add(ctx context.Context, source, format string, a ...any) {
	if s := From(ctx); s != nil {
		s.Add(source, fmt.Sprintf(format, a...))
	}
}
