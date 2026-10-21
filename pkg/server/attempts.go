// pkg/server/attempts.go - why each generation of a request ended, read from the runtime's trace. The runtime can end a
// generation itself and start another (loop recovery); the engine only reports "cancel", so the Recent requests list
// would show a bare "Stopped" for each of them. The cause is in the runtime trace and goes onto the generation's record.
package server

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"strata/pkg/reqtrace"
	"strata/pkg/rtclient"
)

var attemptEndRE = regexp.MustCompile(`^attempt (\d+) ends: (\d+) tokens, (.+)$`)

// attemptNotes maps an attempt number (1-based, the order of the engine generations) to a one-line reason, for the
// attempts the runtime ended or re-ran because of something it detected. A plain finish has no entry.
func attemptNotes(trace []rtclient.TraceEvent) map[int]string {
	notes := map[int]string{}
	pending := ""
	for _, t := range trace {
		switch t.Kind {
		case "recovery":
			pending = strings.TrimRight(strings.TrimSpace(t.Detail), ".")
		case "generation":
			m := attemptEndRE.FindStringSubmatch(t.Detail)
			if m == nil {
				continue
			}
			n, _ := strconv.Atoi(m[1])
			switch {
			case strings.HasPrefix(m[3], "restart requested") && pending != "":
				notes[n] = fmt.Sprintf("runtime retry after attempt %d: %s", n, pending)
			case strings.HasPrefix(m[3], "restart requested"):
				notes[n] = fmt.Sprintf("runtime restarted the generation after attempt %d", n)
			case pending != "":
				notes[n] = "runtime: " + pending
			}
			pending = ""
		}
	}
	return notes
}

// recordIDOf is the API monitor record id of the HTTP request ctx belongs to ("" when there is none).
func recordIDOf(ctx context.Context) string {
	if k, ok := reqtrace.From(ctx).(recSink); ok && k.rec != nil {
		return k.rec.ID
	}
	return ""
}

// noteAttempts puts attemptNotes on the generations of this request, in the order they ran.
func (b *rtBackend) noteAttempts(trace []rtclient.TraceEvent) {
	b.mu.Lock()
	gens := append([]*monitoredRequest(nil), b.gens...)
	b.mu.Unlock()
	for n, note := range attemptNotes(trace) {
		if n >= 1 && n <= len(gens) {
			gens[n-1].setNote(note)
		}
	}
}
