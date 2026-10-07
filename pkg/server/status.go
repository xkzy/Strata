// pkg/server/status.go - GET /v1/status for the web app and the standalone Monitor page, and POST /load, /unload.
package server

import (
	"context"
	"io"
	"net/http"
	"strings"
	"time"

	"strata/pkg/engineipc"
)

var serverStarted = time.Now()

func scaled(v interface{}, unit float64, digits int) interface{} {
	var f float64
	switch x := v.(type) {
	case uint64:
		f = float64(x)
	case float64:
		f = x
	case int:
		f = float64(x)
	default:
		return nil
	}
	p := 1.0
	for i := 0; i < digits; i++ {
		p *= 10
	}
	return float64(int64(f/unit*p+0.5)) / p
}

// architecture is what the page's badge shows for a model name: its family, the sliding-window size and the virtual
// context the runtime offers.
func architecture(model string) (arch string, swaWindow int) {
	m := strings.ToLower(model)
	switch {
	case strings.Contains(m, "mimo"):
		return "mimo_v2_6", 8192
	case strings.Contains(m, "mixtral"):
		return "mixtral", 4096
	case strings.Contains(m, "deepseek"):
		return "deepseek_moe", 4096
	}
	return "qwen_moe", 4096
}

// runtimeStats is the page's "runtime_stats" block: the virtual context and the sessions the server knows.
func (s *StrataServer) runtimeStats() map[string]interface{} {
	vc := s.VirtualContext.Stats()
	tenants, agents := map[string]bool{}, map[string]bool{}
	sessions := s.MultiTenant.ListSessions("")
	for _, sc := range sessions {
		tenants[sc.TenantID] = true
		agents[sc.AgentID] = true
	}
	virt, _ := vc["virtual_context_total_tokens"].(int)
	phys, _ := vc["physical_active_tokens"].(int)
	out := map[string]interface{}{
		"total_virtual_tokens": virt, "total_physical_tokens": phys,
		"total_compaction_events": vc["compaction_events"],
		"active_sessions":         len(sessions), "active_tenants": max(1, len(tenants)), "active_agents": max(1, len(agents)),
	}
	if phys > 0 {
		out["context_compression_ratio"] = float64(int(float64(virt)/float64(phys)*10+0.5)) / 10
	}
	return out
}

func (s *StrataServer) handleStatus(w http.ResponseWriter, r *http.Request) {
	snap := s.Telemetry.Snapshot()
	hw, _ := snap["now"].(map[string]interface{})
	static, _ := snap["static"].(map[string]interface{})
	engine := s.engineStatus()
	loaded, _ := engine["loaded"].(bool)
	running := s.mon.live(loaded)["running"]
	arch, swa := architecture(s.Config.ModelName)
	virt := s.Config.VirtualLimit
	ctx := s.Config.MaxContext

	s.mon.mu.Lock()
	requests := s.mon.totals["requests"].(int)
	s.mon.mu.Unlock()
	inflight, _ := running.(int)

	machine := map[string]interface{}{"at": time.Now().Unix(), "gpu": nil, "igpu": nil, "ram": nil}
	if name, _ := static["gpu_name"].(string); name != "" {
		machine["gpu"] = map[string]interface{}{"name": name, "used_mib": scaled(hw["gpu_mem_used"], 1<<20, 0),
			"total_mib": scaled(hw["gpu_mem_total"], 1<<20, 0), "util_pct": hw["gpu_util"], "temp_c": hw["gpu_temp"],
			"power_w": hw["gpu_power"]}
	}
	if hw["ram_total"] != nil {
		machine["ram"] = map[string]interface{}{"used_gib": scaled(hw["ram_used"], 1<<30, 1), "total_gib": scaled(hw["ram_total"], 1<<30, 1)}
	}

	out := map[string]interface{}{
		"service": "strata-go", "model": s.Config.ModelName, "architecture": arch,
		"swa": map[string]interface{}{"enabled": true, "window_size": swa}, "runtime_stats": s.runtimeStats(),
		"anti_loop": s.AntiLoop.Stats(), "loaded": loaded, "auto_load": false,
		"started": serverStarted.Unix(), "uptime_s": int(time.Since(serverStarted).Seconds()),
		"cache_max_tokens": ctx, "max_context": ctx, "busy": inflight > 0,
		"context": map[string]interface{}{"native": ctx, "max_positions": ctx, "n_ctx_physical": ctx, "n_ctx_virtual": virt,
			"virtual_context_limit": virt, "virtual_expansion_ratio": float64(virt) / float64(ctx)},
		"concurrency": map[string]interface{}{"serving": 1, "requested": 1},
		"dialects":    []string{"/v1/chat/completions", "/v1/messages", "/v1/responses"},
		"activity":    map[string]interface{}{"requests": requests + inflight, "in_flight": inflight, "last_request_at": nil},
		"machine":     machine,
	}
	for k, v := range engine { // engine, loaded, error, runtime ...
		if _, taken := out[k]; !taken {
			out[k] = v
		}
	}
	writeJSON(w, http.StatusOK, out)
}

// ownPage lets only JSON through (a form or another "simple" cross-site request cannot send it without a CORS
// preflight, which this server never grants): a web page elsewhere must not load or unload the model.
func ownPage(w http.ResponseWriter, r *http.Request, what string) bool {
	if !strings.HasPrefix(r.Header.Get("Content-Type"), "application/json") {
		apiMessage(w, http.StatusUnsupportedMediaType, "invalid_request_error", "send JSON (Content-Type: application/json): "+what)
		return false
	}
	return true
}

// handleUnload is POST /unload: give the GPU back now, between requests.
func (s *StrataServer) handleUnload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	io.Copy(io.Discard, io.LimitReader(r.Body, 65536))
	if !ownPage(w, r, "the model can be unloaded") {
		return
	}
	if n, _ := s.mon.live(true)["running"].(int); n > 0 {
		writeJSON(w, http.StatusConflict, map[string]interface{}{"status": "busy"})
		return
	}
	if st, _ := s.EngineIPC.Status(); st == engineipc.StateReady || st == engineipc.StateStarting {
		_ = s.EngineIPC.Stop()
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"status": "unloaded"})
}

// handleLoad is POST /load: load the model now, e.g. ahead of a request. It answers when the model is ready.
func (s *StrataServer) handleLoad(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	io.Copy(io.Discard, io.LimitReader(r.Body, 65536))
	if !ownPage(w, r, "the model can be loaded") {
		return
	}
	s.eng.mu.RLock()
	problem := s.eng.problem
	s.eng.mu.RUnlock()
	st, _ := s.EngineIPC.Status()
	switch {
	case st == engineipc.StateReady:
		writeJSON(w, http.StatusOK, map[string]interface{}{"status": "loaded"})
		return
	case problem != "": // the config or the tokenizer failed at start: read them again
		if err := s.StartEngine(context.Background()); err != nil {
			apiMessage(w, http.StatusServiceUnavailable, "server_error", err.Error())
			return
		}
		s.setProblem("")
	default:
		if err := s.EngineIPC.Start(context.Background()); err != nil {
			apiMessage(w, http.StatusServiceUnavailable, "server_error", err.Error())
			return
		}
	}
	if err := s.EngineIPC.WaitReady(r.Context()); err != nil {
		apiMessage(w, http.StatusServiceUnavailable, "server_error", "the model did not load: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"status": "loaded"})
}
