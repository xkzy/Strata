package server

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// The Monitor tab polls /metrics once a second while it is open, so this is a hot path for an open browser. The
// monitor is filled to its cap first: that is the worst case, and the one an operator watching a busy server has.
func benchmarkMetrics(b *testing.B, query string) {
	srv := NewStrataServer(ServerConfig{Port: 0})
	for i := 0; i < monitorKeep; i++ {
		r := srv.mon.begin(1000+i, 256)
		r.token()
		r.finish("stop", 100, 0)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec := httptest.NewRecorder()
		srv.handleMetrics(rec, httptest.NewRequest(http.MethodGet, "/metrics"+query, nil))
	}
}

func BenchmarkMetricsHandler(b *testing.B)            { benchmarkMetrics(b, "") }
func BenchmarkMetricsHandlerAllRequests(b *testing.B) { benchmarkMetrics(b, "?requests=all") }
