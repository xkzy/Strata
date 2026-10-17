package server

import (
	"errors"
	"net"
	"net/http"
	"strconv"

	"strata/pkg/multitenant"
	"strata/pkg/scheduler"
)

// requestTicket says who a request is for the engine's queue: the caller's API key if it sends one, else the tenant /
// user headers, else its address - so one noisy client does not push the others back. X-Strata-Priority (0-10, default
// 5) orders what waits; it is the caller's own word, which is fine on a loopback server and meaningless for strangers,
// so a server open to others should keep the default and rely on the per-caller fairness.
func requestTicket(r *http.Request) scheduler.Ticket {
	sc := multitenant.FromHeaders(r.Header, nil)
	tenant := sc.Credential
	if tenant == "" && (r.Header.Get("X-Tenant-ID") != "" || r.Header.Get("X-User-ID") != "") {
		tenant = sc.TenantID + "/" + sc.UserID
	}
	if tenant == "" {
		tenant, _, _ = net.SplitHostPort(r.RemoteAddr)
		if tenant == "" {
			tenant = r.RemoteAddr
		}
	}
	prio := scheduler.DefaultPriority
	if v := r.Header.Get("X-Strata-Priority"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			prio = n
		}
	}
	return scheduler.Ticket{Tenant: tenant, Priority: prio}
}

// withTicket puts the request's ticket in its context, where the engine's queue reads it.
func withTicket(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		next.ServeHTTP(w, r.WithContext(scheduler.WithTicket(r.Context(), requestTicket(r))))
	})
}

// unavailableErr is the API error for engineUnavailable's answer. A 429 (the queue is full) carries when to come back.
func (s *StrataServer) unavailableErr(status int, msg string) *apiError {
	e := &apiError{Status: status, Type: "engine_unavailable", Msg: msg}
	if status == http.StatusTooManyRequests {
		e.Type = "rate_limit_exceeded"
		if s.EngineIPC != nil {
			e.RetryAfter, _ = s.EngineIPC.Scheduler().Full()
		}
	}
	return e
}

// queueFullError turns the queue's refusal (it can come after the pre-check, when the queue filled in between) into the
// same 429 answer, and reports whether err was one.
func queueFullError(err error) (*apiError, bool) {
	var qf *scheduler.ErrQueueFull
	if !errors.As(err, &qf) {
		return nil, false
	}
	return &apiError{Status: http.StatusTooManyRequests, Type: "rate_limit_exceeded", Msg: qf.Error(), RetryAfter: qf.RetryAfter}, true
}
