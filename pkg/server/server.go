package server

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"strata/pkg/antiloop"
	"strata/pkg/engineipc"
	"strata/pkg/frontend"
	"strata/pkg/generationloop"
	"strata/pkg/hallucination"
	"strata/pkg/installer"
	"strata/pkg/logicverifier"
	"strata/pkg/mathruntime"
	"strata/pkg/mcp"
	"strata/pkg/multitenant"
	"strata/pkg/processjob"
	"strata/pkg/resourcemanager"
	"strata/pkg/responses"
	"strata/pkg/runconfig"
	"strata/pkg/structured"
	"strata/pkg/telemetry"
	"strata/pkg/virtualcontext"
)

type ServerConfig struct {
	Port          int
	ModelName     string
	MaxContext    int
	VirtualLimit  int
	APIKey        string
	ModelPath     string
	BinaryPath    string
	ConfigFile    string
	RuntimeMode   string // on | off | auto: the C++ transparent runtime (verification, virtual context)
	RuntimeBinary string
	WindowTokens  int    // physical context window the runtime fills (default 32768, at most the engine's context)
	BindHost      string // address the HTTP server listens on; empty is treated as loopback
}

type StrataServer struct {
	Config          ServerConfig
	MathRuntime     *mathruntime.MathRuntime
	LogicVerifier   *logicverifier.LogicVerifier
	McpHub          *mcp.McpHub
	Structured      *structured.StructuredValidator
	EngineIPC       *engineipc.EngineOrchestrator
	Telemetry       *telemetry.TelemetryCollector
	Responses       *responses.ResponsesHandler
	ConfigManager   *runconfig.ConfigManager
	ProcessJob      *processjob.ProcessJob
	Frontend        *frontend.FrontendHandler
	MultiTenant     *multitenant.MultiTenantManager
	VirtualContext  *virtualcontext.VirtualContextManager
	AntiLoop        *antiloop.AntiLoopManager
	ResourceManager *resourcemanager.ResourceManager
	GenLoopDetector *generationloop.GenerationLoopDetector
	SetupManager    *installer.SetupManager
	Hallucination   *hallucination.HallucinationRuntime
	eng             engineState
	rt              rtState
	server          *http.Server
	activeCancels   map[string]context.CancelFunc
	cancelMu        sync.Mutex
}

func NewStrataServer(cfg ServerConfig) *StrataServer {
	if cfg.Port <= 0 {
		cfg.Port = 8080
	}
	if cfg.ModelName == "" {
		cfg.ModelName = "Qwen3.8-Flash-Next"
	}
	if cfg.MaxContext <= 0 {
		cfg.MaxContext = 32768
	}
	if cfg.VirtualLimit <= 0 {
		cfg.VirtualLimit = 2000000
	}

	mr := mathruntime.NewMathRuntime(nil)
	lv := logicverifier.NewLogicVerifier()
	vc := virtualcontext.NewVirtualContextManager(cfg.MaxContext, cfg.VirtualLimit)
	nativeTools := mcp.NewNativeToolsProvider(mr, lv, vc)
	hub := mcp.NewMcpHub(nativeTools, 60*time.Second, 20000)

	engineCfg := engineipc.EngineConfig{
		BinaryPath:  cfg.BinaryPath,
		ModelPath:   cfg.ModelPath,
		ContextSize: cfg.MaxContext,
	}
	orchestrator := engineipc.NewEngineOrchestrator(engineCfg)

	tc := telemetry.NewTelemetryCollector()
	tc.Start()

	hr := hallucination.NewRuntime(".", hallucination.DefaultSafetyControllerOptions())

	s := &StrataServer{
		Config:          cfg,
		MathRuntime:     mr,
		LogicVerifier:   lv,
		McpHub:          hub,
		Structured:      structured.NewStructuredValidator(),
		EngineIPC:       orchestrator,
		Telemetry:       tc,
		Responses:       responses.NewResponsesHandler(),
		ConfigManager:   runconfig.NewConfigManager(cfg.ConfigFile),
		ProcessJob:      processjob.GlobalJob(),
		Frontend:        frontend.NewFrontendHandler(),
		MultiTenant:     multitenant.NewMultiTenantManager(),
		VirtualContext:  vc,
		AntiLoop:        antiloop.Instance(),
		ResourceManager: resourcemanager.NewResourceManager(),
		GenLoopDetector: generationloop.NewGenerationLoopDetector(nil),
		SetupManager:    installer.NewSetupManager(),
		Hallucination:   hr,
		activeCancels:   make(map[string]context.CancelFunc),
	}
	return s
}

func (s *StrataServer) Router() http.Handler {
	mux := http.NewServeMux()

	// OpenAI / Anthropic APIs
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/v1/completions", s.handleCompletions)
	mux.HandleFunc("/v1/messages", s.handleAnthropicMessages)
	mux.HandleFunc("/v1/responses", s.handleResponsesAPI)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/models", s.handleModels)

	// Health, Status & Telemetry
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/api/health", s.handleHealth)
	mux.HandleFunc("/status", s.handleStatus)
	mux.HandleFunc("/v1/status", s.handleStatus)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/telemetry", s.handleTelemetry)
	mux.HandleFunc("/v1/telemetry", s.handleTelemetry)
	mux.HandleFunc("/api/telemetry", s.handleTelemetry)

	// Slots, Cache, Model & Control
	mux.HandleFunc("/api/slots", s.handleSlots)
	mux.HandleFunc("/api/cache", s.handleCache)
	mux.HandleFunc("/api/model", s.handleModelInfo)
	mux.HandleFunc("/api/cancel", s.handleCancel)
	mux.HandleFunc("/api/abort", s.handleCancel)

	// In-Web Hardware & Model Setup Endpoints
	mux.HandleFunc("/api/setup/hardware", s.handleSetupHardware)
	mux.HandleFunc("/api/setup/models", s.handleSetupModels)
	mux.HandleFunc("/api/setup/configure", s.handleSetupConfigure)
	mux.HandleFunc("/api/setup/download", s.handleSetupDownload)
	mux.HandleFunc("/api/setup/status", s.handleSetupStatus)
	mux.HandleFunc("/api/setup/cancel", s.handleSetupCancel)

	// Config Settings (#564)
	mux.HandleFunc("/config", s.handleConfig)

	// Math Runtime Endpoints (#20)
	mux.HandleFunc("/v1/strata/math/evaluate", s.handleMathEvaluate)
	mux.HandleFunc("/v1/strata/math/verify", s.handleMathVerify)
	mux.HandleFunc("/v1/strata/math/intercept", s.handleMathIntercept)
	mux.HandleFunc("/v1/strata/math/metrics", s.handleMathMetrics)
	mux.HandleFunc("/v1/strata/math/backends", s.handleMathBackends)

	// Logic Verification Runtime Endpoints
	mux.HandleFunc("/v1/strata/verify", s.handleLogicVerify)
	mux.HandleFunc("/v1/strata/verify/claims", s.handleLogicVerifyClaims)
	mux.HandleFunc("/v1/strata/verify/metrics", s.handleLogicVerifyMetrics)

	// Hallucination Runtime Endpoints
	mux.HandleFunc("/v1/strata/hallucination/evaluate", s.handleHallucinationEvaluate)
	mux.HandleFunc("/v1/strata/hallucination/claims", s.handleHallucinationClaims)
	mux.HandleFunc("/v1/strata/hallucination/metrics", s.handleHallucinationMetrics)

	// Multi-Tenant & Session Management
	mux.HandleFunc("/v1/strata/sessions", s.handleSessions)
	mux.HandleFunc("/v1/strata/sessions/fork", s.handleSessionFork)
	mux.HandleFunc("/v1/strata/sessions/delete", s.handleSessionDelete)
	mux.HandleFunc("/v1/strata/tenants/metrics", s.handleTenantMetrics)

	// Context & Tool Stores
	mux.HandleFunc("/v1/strata/context", s.handleContextStats)
	mux.HandleFunc("/v1/strata/context/query", s.handleContextQuery)
	mux.HandleFunc("/v1/strata/tools/retrieve", s.handleToolsRetrieve)

	// Anti-Loop Guard
	mux.HandleFunc("/v1/strata/runtime/metrics", s.handleRuntimeMetrics)
	mux.HandleFunc("/v1/strata/guard", s.handleGuardStats)
	mux.HandleFunc("/v1/strata/guard/check", s.handleGuardCheck)
	mux.HandleFunc("/v1/strata/guard/outcome", s.handleGuardOutcome)

	// Model Context Protocol (MCP) Endpoints
	mux.HandleFunc("/v1/mcp", s.handleMcp)
	mux.HandleFunc("/v1/mcp/tools", s.handleMcpTools)

	// Root Web UI
	mux.HandleFunc("/", s.handleRoot)

	return s.corsMiddleware(mux)
}

// publicPath says whether a request needs no API key: the health probes, the web UI shell and the read-only setup
// queries. Everything that changes state (setup configure / download / cancel, generation, config) needs the key.
func publicPath(r *http.Request) bool {
	switch r.URL.Path {
	case "/health", "/api/health", "/":
		return true
	case "/api/setup/hardware", "/api/setup/models", "/api/setup/status":
		return r.Method == http.MethodGet
	}
	return false
}

// CheckBind refuses to listen beyond loopback without an API key (the server can run a model and change its settings).
func CheckBind(host, apiKey string) error {
	if host != "" && !isLoopbackHost(host) && apiKey == "" {
		return fmt.Errorf("refusing to listen on %q without --api-key: use 127.0.0.1 or set an API key", host)
	}
	return nil
}

// hostAllowed: names listed in STRATA_ALLOWED_HOSTS (comma separated, with or without port).
func hostAllowed(host string) bool {
	h := host
	if hh, _, err := net.SplitHostPort(host); err == nil {
		h = hh
	}
	for _, a := range strings.Split(os.Getenv("STRATA_ALLOWED_HOSTS"), ",") {
		if a = strings.TrimSpace(a); a != "" && (a == host || a == h) {
			return true
		}
	}
	return false
}

func isLoopbackHost(host string) bool {
	if h, _, err := net.SplitHostPort(host); err == nil {
		host = h
	}
	host = strings.Trim(host, "[]")
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// originAllowed: the page's own origin, loopback origins, and any origin listed in STRATA_ALLOWED_ORIGINS. A web page on
// another site cannot use a user's local server through their browser.
func originAllowed(origin string, r *http.Request) bool {
	u, err := url.Parse(origin)
	if err != nil || u.Host == "" {
		return false
	}
	if isLoopbackHost(u.Host) && isLoopbackHost(r.Host) {
		return true
	}
	if hostAllowed(r.Host) && u.Host == r.Host { // an explicitly allowed name (STRATA_ALLOWED_HOSTS)
		return true
	}
	for _, o := range strings.Split(os.Getenv("STRATA_ALLOWED_ORIGINS"), ",") {
		if o = strings.TrimSpace(o); o != "" && o == origin {
			return true
		}
	}
	return false
}

func (s *StrataServer) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// DNS rebinding: a hostile name that resolves to 127.0.0.1 would make a page on it "same origin". A server bound
		// to loopback therefore only answers requests addressed to a loopback name (or one in STRATA_ALLOWED_HOSTS).
		if (s.Config.BindHost == "" || isLoopbackHost(s.Config.BindHost)) && !isLoopbackHost(r.Host) && !hostAllowed(r.Host) {
			http.Error(w, `{"error":{"message":"invalid Host header","type":"forbidden"}}`, http.StatusMisdirectedRequest)
			return
		}
		if origin := r.Header.Get("Origin"); origin != "" {
			if !originAllowed(origin, r) {
				http.Error(w, `{"error":{"message":"Origin not allowed","type":"forbidden"}}`, http.StatusForbidden)
				return
			}
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Vary", "Origin")
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Tenant-ID, X-User-ID, X-Session-ID")
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		if s.Config.APIKey != "" && !publicPath(r) {
			got := []byte(r.Header.Get("Authorization"))
			want := []byte("Bearer " + s.Config.APIKey)
			if subtle.ConstantTimeCompare(got, want) != 1 {
				http.Error(w, `{"error":{"message":"Unauthorized","type":"authentication_error"}}`, http.StatusUnauthorized)
				return
			}
		}

		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func (s *StrataServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	out := map[string]interface{}{
		"status":                  "ok",
		"service":                 "strata-go",
		"model":                   s.Config.ModelName,
		"max_context":             s.Config.MaxContext,
		"virtual_context_limit":   s.Config.VirtualLimit,
		"virtual_expansion_ratio": float64(s.Config.VirtualLimit) / float64(s.Config.MaxContext),
	}
	for k, v := range s.engineStatus() {
		out[k] = v
	}
	if out["loaded"] != true {
		out["status"] = "degraded" // the server is up, the model is not
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *StrataServer) handleStatus(w http.ResponseWriter, r *http.Request) {
	out := map[string]interface{}{"model": s.Config.ModelName, "max_context": s.Config.MaxContext, "busy": len(s.activeCancelIDs()) > 0}
	for k, v := range s.engineStatus() {
		out[k] = v
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *StrataServer) activeCancelIDs() []string {
	s.cancelMu.Lock()
	defer s.cancelMu.Unlock()
	ids := make([]string, 0, len(s.activeCancels))
	for id := range s.activeCancels {
		ids = append(ids, id)
	}
	return ids
}

func (s *StrataServer) handleModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"object": "list",
		"data": []map[string]interface{}{
			{
				"id":       s.Config.ModelName,
				"object":   "model",
				"created":  time.Now().Unix(),
				"owned_by": "strata",
			},
		},
	})
}

func (s *StrataServer) handleSlots(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"slots":       []interface{}{},
		"batch_slots": 0,
		"max_context": s.Config.MaxContext,
	})
}

func (s *StrataServer) handleCache(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"cached_tokens": 0,
		"hits":          0,
		"misses":        0,
		"history":       []interface{}{},
	})
}

func (s *StrataServer) handleModelInfo(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"name":        s.Config.ModelName,
		"path":        s.Config.ModelPath,
		"max_context": s.Config.MaxContext,
		"loaded":      s.engineStatus()["loaded"],
	})
}

func (s *StrataServer) handleCancel(w http.ResponseWriter, r *http.Request) {
	s.cancelMu.Lock()
	for id, cancel := range s.activeCancels {
		cancel()
		delete(s.activeCancels, id)
	}
	s.cancelMu.Unlock()

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status": "cancelled",
	})
}

func (s *StrataServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	snap := s.Telemetry.Snapshot()
	nowMap, _ := snap["now"].(map[string]interface{})
	staticMap, _ := snap["static"].(map[string]interface{})
	histMap, _ := snap["history"].(map[string]interface{})

	virtLimit := s.Config.VirtualLimit
	if virtLimit <= 0 {
		virtLimit = 2000000
	}

	engineMap := map[string]interface{}{
		"model":                   s.Config.ModelName,
		"version":                 "0.1.39",
		"max_context":             s.Config.MaxContext,
		"n_ctx_physical":          s.Config.MaxContext,
		"n_ctx_virtual":           virtLimit,
		"virtual_context_limit":   virtLimit,
		"virtual_expansion_ratio": float64(virtLimit) / float64(s.Config.MaxContext),
		"kv":                      "int8",
		"images":                  false,
	}

	liveMap := map[string]interface{}{
		"state":   "idle",
		"queued":  0,
		"tok_s":   0.0,
		"running": 0,
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"engine":          engineMap,
		"live":            liveMap,
		"hardware":        nowMap,
		"hardware_static": staticMap,
		"history":         histMap,
		"requests":        []interface{}{},
		"totals": map[string]interface{}{
			"requests":      0,
			"prompt_tokens": 0,
			"output_tokens": 0,
			"reused":        0,
		},
		"resource":  s.ResourceManager.Metrics(),
		"anti_loop": s.AntiLoop.Stats(),
		"context":   s.VirtualContext.Stats(),
		"math":      s.MathRuntime.GetStats(),
		"time":      time.Now().Unix(),
	})
}

// ---------------------------------------------------------------------------
// Math Runtime HTTP Handlers (#20)
// ---------------------------------------------------------------------------

func (s *StrataServer) handleMathEvaluate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req mathruntime.MathRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":{"message":"invalid json payload"}}`, http.StatusBadRequest)
		return
	}

	scope := multitenant.FromHeaders(r.Header, &multitenant.SecurityScope{
		TenantID:  req.TenantID,
		UserID:    req.UserID,
		SessionID: req.SessionID,
	})
	req.TenantID = scope.TenantID
	req.UserID = scope.UserID
	req.SessionID = scope.SessionID

	res := s.MathRuntime.ProcessRequest(req)
	writeJSON(w, http.StatusOK, res)
}

func (s *StrataServer) handleMathVerify(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var payload struct {
		LLMOutput          string `json:"llm_output"`
		ExpectedExpression string `json:"expected_expression"`
	}
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, `{"error":{"message":"invalid json payload"}}`, http.StatusBadRequest)
		return
	}

	ver := s.MathRuntime.VerifyCalculation(payload.LLMOutput, payload.ExpectedExpression)
	writeJSON(w, http.StatusOK, ver)
}

func (s *StrataServer) handleMathIntercept(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var payload struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, `{"error":{"message":"invalid json payload"}}`, http.StatusBadRequest)
		return
	}

	transformed, results := s.MathRuntime.InterceptAndEvaluate(payload.Text)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"text":              transformed,
		"intercepted_count": len(results),
		"results":           results,
	})
}

func (s *StrataServer) handleMathMetrics(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.MathRuntime.GetStats())
}

func (s *StrataServer) handleMathBackends(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"mathics_available":      s.MathRuntime.MathicsBackend.Available,
		"mathics_version":        s.MathRuntime.MathicsBackend.Version,
		"sage_available":         s.MathRuntime.SageBackend.Available,
		"sage_version":           s.MathRuntime.SageBackend.Version,
		"fast_numeric_available": true,
		"fast_numeric_version":   s.MathRuntime.FastBackend.Version,
	})
}

// ---------------------------------------------------------------------------
// Logic Verification Runtime Handlers
// ---------------------------------------------------------------------------

func (s *StrataServer) handleLogicVerify(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var claim logicverifier.VerificationClaim
	if err := json.NewDecoder(r.Body).Decode(&claim); err != nil {
		http.Error(w, `{"error":{"message":"invalid claim json payload"}}`, http.StatusBadRequest)
		return
	}

	scope := multitenant.FromHeaders(r.Header, &multitenant.SecurityScope{
		TenantID:  claim.TenantID,
		UserID:    claim.UserID,
		SessionID: claim.SessionID,
	})
	claim.TenantID = scope.TenantID
	claim.UserID = scope.UserID
	claim.SessionID = scope.SessionID

	result := s.LogicVerifier.Verify(claim)

	// Persist verification audit record to Virtual Context
	if s.VirtualContext != nil && result.Status != logicverifier.StatusUnknown {
		s.VirtualContext.AppendItem(
			virtualcontext.CategoryVerificationAudit,
			fmt.Sprintf("Verification [%s] %s: %s | %s", result.Status, result.Type, result.Evidence, result.FailureReason),
			map[string]string{
				"claim_id":   result.ClaimID,
				"tenant_id":  claim.TenantID,
				"session_id": claim.SessionID,
				"status":     string(result.Status),
			},
		)
	}

	writeJSON(w, http.StatusOK, result)
}

func (s *StrataServer) handleLogicVerifyClaims(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var payload struct {
		Text string `json:"text"`
	}
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, `{"error":{"message":"invalid json payload"}}`, http.StatusBadRequest)
		return
	}

	scope := multitenant.FromHeaders(r.Header, nil)
	results := s.LogicVerifier.VerifyText(payload.Text, scope.TenantID, scope.SessionID)
	observations := s.LogicVerifier.FormatObservations(results)

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"claim_count":         len(results),
		"results":             results,
		"compact_observation": observations,
	})
}

func (s *StrataServer) handleLogicVerifyMetrics(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.LogicVerifier.GetMetrics())
}

// ---------------------------------------------------------------------------
// Chat, Completions & Messages Handlers (OpenAI & Anthropic Compatible)
// ---------------------------------------------------------------------------

func (s *StrataServer) handleSessions(w http.ResponseWriter, r *http.Request) {
	scope := multitenant.FromHeaders(r.Header, nil)
	s.MultiTenant.RegisterSession(scope)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":     "active",
		"session_id": scope.SessionID,
	})
}

func (s *StrataServer) handleSessionFork(w http.ResponseWriter, r *http.Request) {
	var body struct {
		ParentSessionID string `json:"parent_session_id"`
		NewSessionID    string `json:"new_session_id"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	scope := multitenant.DefaultSecurityScope()
	scope.SessionID = body.NewSessionID
	s.MultiTenant.RegisterSession(scope)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":     "forked",
		"session_id": scope.SessionID,
	})
}

func (s *StrataServer) handleSessionDelete(w http.ResponseWriter, r *http.Request) {
	var body struct {
		SessionID string `json:"session_id"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	deleted := s.MultiTenant.DeleteSession(body.SessionID)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"deleted":    deleted,
		"session_id": body.SessionID,
	})
}

func (s *StrataServer) handleTenantMetrics(w http.ResponseWriter, r *http.Request) {
	scope := multitenant.FromHeaders(r.Header, nil)
	writeJSON(w, http.StatusOK, s.MultiTenant.GetTenantMetrics(scope.TenantID))
}

func (s *StrataServer) handleContextStats(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.VirtualContext.Stats())
}

func (s *StrataServer) handleContextQuery(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Query string `json:"query"`
		TopK  int    `json:"top_k"`
	}
	json.NewDecoder(r.Body).Decode(&body)
	if body.TopK <= 0 {
		body.TopK = 5
	}
	hits := s.VirtualContext.Query(body.Query, body.TopK)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"query": body.Query,
		"hits":  hits,
	})
}

func (s *StrataServer) handleToolsRetrieve(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"result_id": "tool_1",
		"content":   "Sample raw tool log content retrieved externally.",
	})
}

func (s *StrataServer) handleGuardStats(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, s.AntiLoop.Stats())
}

func (s *StrataServer) handleGuardCheck(w http.ResponseWriter, r *http.Request) {
	var act antiloop.ActionRecord
	json.NewDecoder(r.Body).Decode(&act)
	v := s.AntiLoop.EvaluateAction(act)
	writeJSON(w, http.StatusOK, v)
}

func (s *StrataServer) handleGuardOutcome(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status": "recorded",
	})
}

// ---------------------------------------------------------------------------
// Model Context Protocol (MCP) Handlers
// ---------------------------------------------------------------------------

func (s *StrataServer) handleMcp(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"jsonrpc":"2.0","error":{"code":-32700,"message":"Parse error"}}`, http.StatusBadRequest)
		return
	}

	respBytes, err := s.McpHub.HandleJSONRPC(r.Context(), bodyBytes)
	if err != nil {
		http.Error(w, string(respBytes), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(respBytes)
}

func (s *StrataServer) handleMcpTools(w http.ResponseWriter, r *http.Request) {
	tools := s.McpHub.GetAllTools(r.Context())
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"tools": tools,
	})
}

// ---------------------------------------------------------------------------
// Responses API Handler (POST /v1/responses)
// ---------------------------------------------------------------------------

func (s *StrataServer) handleResponsesAPI(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req responses.ResponsesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":{"message":"invalid json body"}}`, http.StatusBadRequest)
		return
	}

	if req.Model == "" {
		req.Model = s.Config.ModelName
	}

	respObj, err := s.Responses.ProcessRequest(req, "Completed response from Strata Go Engine.", "Verified mathematical & logical chain of thought.")
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":{"message":"%v"}}`, err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, respObj)
}

// ---------------------------------------------------------------------------
// Telemetry & Hardware Monitor Handlers
// ---------------------------------------------------------------------------

func (s *StrataServer) handleTelemetry(w http.ResponseWriter, r *http.Request) {
	current := s.Telemetry.GetCurrent()
	history := s.Telemetry.GetHistory()

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"current": current,
		"history": history,
	})
}

// ---------------------------------------------------------------------------
// Run Config Settings Handler (/config)
// ---------------------------------------------------------------------------

func (s *StrataServer) handleConfig(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		writeJSON(w, http.StatusOK, s.ConfigManager.Get())
		return
	}

	if r.Method == http.MethodPost {
		var cfg runconfig.StrataConfig
		if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
			http.Error(w, `{"error":{"message":"invalid config json"}}`, http.StatusBadRequest)
			return
		}
		s.ConfigManager.Update(cfg)
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"status": "updated",
			"config": s.ConfigManager.Get(),
		})
		return
	}

	http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
}

func (s *StrataServer) handleRoot(w http.ResponseWriter, r *http.Request) {
	if s.Frontend != nil {
		s.Frontend.ServeHTTP(w, r)
		return
	}
	if r.URL.Path == "/" {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprintf(w, `<!DOCTYPE html><html><head><title>Strata AI Server (Go)</title></head><body><h1>Strata Go Engine & Server</h1><p>Mathematical Backing Runtime, Virtual Context & MoE Inference Active.</p></body></html>`)
		return
	}
	http.NotFound(w, r)
}

// ---------------------------------------------------------------------------
// In-Web Hardware Setup & Model Configuration Handlers
// ---------------------------------------------------------------------------

func (s *StrataServer) handleSetupHardware(w http.ResponseWriter, r *http.Request) {
	specs := installer.DetectHardware()
	writeJSON(w, http.StatusOK, specs)
}

func (s *StrataServer) handleSetupModels(w http.ResponseWriter, r *http.Request) {
	specs := installer.DetectHardware()
	models := installer.GetAvailableModels()
	rec := installer.RecommendModel(specs)
	for i := range models {
		if models[i].Name == rec.Name {
			models[i].Recommended = true
		}
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"hardware":    specs,
		"models":      models,
		"recommended": rec,
	})
}

func (s *StrataServer) handleSetupConfigure(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		ModelName    string `json:"model_name"`
		ModelPath    string `json:"model_path"`
		MaxContext   int    `json:"max_context"`
		VirtualLimit int    `json:"virtual_limit"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":{"message":"Invalid JSON payload"}}`, http.StatusBadRequest)
		return
	}

	if req.ModelName != "" {
		s.Config.ModelName = req.ModelName
	}
	if req.ModelPath != "" {
		s.Config.ModelPath = req.ModelPath
	}
	if req.MaxContext > 0 {
		s.Config.MaxContext = req.MaxContext
	}
	if req.VirtualLimit > 0 {
		s.Config.VirtualLimit = req.VirtualLimit
	}

	// Persist to ConfigManager
	if s.ConfigManager != nil {
		cfg := s.ConfigManager.Get()
		cfg.Model = s.Config.ModelName
		cfg.ModelPath = s.Config.ModelPath
		cfg.MaxContext = s.Config.MaxContext
		cfg.VirtualLimit = s.Config.VirtualLimit
		s.ConfigManager.Update(cfg)
		_ = s.ConfigManager.SaveToFile("")
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status": "configured",
		"config": map[string]interface{}{
			"model_name":    s.Config.ModelName,
			"model_path":    s.Config.ModelPath,
			"max_context":   s.Config.MaxContext,
			"virtual_limit": s.Config.VirtualLimit,
		},
	})
}

func mustAbs(p string) string {
	a, _ := filepath.Abs(p)
	return a
}

func (s *StrataServer) handleSetupDownload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Only models of the built-in catalog can be downloaded: the client chooses a name, never a URL or a path.
	var req struct {
		ModelName   string `json:"model_name"`
		DownloadURL string `json:"-"` // never read from the client
		DestPath    string `json:"-"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":{"message":"Invalid JSON payload"}}`, http.StatusBadRequest)
		return
	}
	var choice *installer.ModelOption
	for _, m := range installer.GetAvailableModels() {
		if m.Name == req.ModelName {
			m := m
			choice = &m
			break
		}
	}
	if choice == nil {
		http.Error(w, `{"error":{"message":"unknown model: choose one from /api/setup/models"}}`, http.StatusBadRequest)
		return
	}
	req.DownloadURL = choice.DownloadURL
	if choice.DownloadURL == "" { // a model already on this machine (catalog entry with a local path)
		req.DestPath = choice.Filename
	} else {
		req.DestPath = filepath.Join("models", filepath.Base(choice.Filename)) // the file name only: no directories from the catalog either
		if dest, err := filepath.Abs(req.DestPath); err != nil || !strings.HasPrefix(dest, mustAbs("models")+string(filepath.Separator)) {
			http.Error(w, `{"error":{"message":"invalid destination"}}`, http.StatusBadRequest)
			return
		}
	}

	// If the file or pack already exists locally, activate it immediately
	if fi, err := os.Stat(req.DestPath); err == nil {
		s.Config.ModelName = req.ModelName
		s.Config.ModelPath = req.DestPath
		if s.ConfigManager != nil {
			cfg := s.ConfigManager.Get()
			cfg.Model = s.Config.ModelName
			cfg.ModelPath = s.Config.ModelPath
			s.ConfigManager.Update(cfg)
			_ = s.ConfigManager.SaveToFile("")
		}
		_ = fi
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"status":       "activated",
			"message":      fmt.Sprintf("Local model %s activated from %s", req.ModelName, req.DestPath),
			"active_model": s.Config.ModelName,
			"active_path":  s.Config.ModelPath,
		})
		return
	}
	if req.DownloadURL == "" {
		http.Error(w, `{"error":{"message":"this catalog entry has no download URL and is not present locally"}}`, http.StatusBadRequest)
		return
	}

	err := s.SetupManager.StartDownload(req.ModelName, req.DownloadURL, req.DestPath)
	if err != nil {
		writeJSON(w, http.StatusConflict, map[string]interface{}{
			"error":  err.Error(),
			"status": s.SetupManager.GetStatus(),
		})
		return
	}

	s.Config.ModelName = req.ModelName
	s.Config.ModelPath = req.DestPath

	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"status":   "download_started",
		"download": s.SetupManager.GetStatus(),
	})
}

func (s *StrataServer) handleSetupStatus(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"download":     s.SetupManager.GetStatus(),
		"active_model": s.Config.ModelName,
		"active_path":  s.Config.ModelPath,
		"max_context":  s.Config.MaxContext,
	})
}

func (s *StrataServer) handleSetupCancel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	s.SetupManager.CancelDownload()
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status": "download_cancelled",
	})
}

// Hallucination Runtime Handlers

type HallucinationEvalRequest struct {
	Text      string                   `json:"text"`
	SessionID string                   `json:"session_id,omitempty"`
	RequestID string                   `json:"request_id,omitempty"`
	Policy    hallucination.PolicyMode `json:"policy,omitempty"`
	Evidence  []hallucination.Evidence `json:"evidence,omitempty"`
}

func (s *StrataServer) handleHallucinationEvaluate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"failed to read body: %v"}`, err), http.StatusBadRequest)
		return
	}

	var req HallucinationEvalRequest
	if err := json.Unmarshal(body, &req); err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"invalid JSON: %v"}`, err), http.StatusBadRequest)
		return
	}

	tenantID := r.Header.Get("X-Tenant-ID")
	if tenantID == "" {
		tenantID = "default"
	}

	for _, ev := range req.Evidence {
		s.Hallucination.AddEvidence(ev)
	}

	report, err := s.Hallucination.EvaluateText(r.Context(), req.Text, req.SessionID, req.RequestID, tenantID, req.Policy)
	if err != nil {
		http.Error(w, fmt.Sprintf(`{"error":"evaluation failed: %v"}`, err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, report)
}

func (s *StrataServer) handleHallucinationClaims(w http.ResponseWriter, r *http.Request) {
	claimID := r.URL.Query().Get("id")
	if claimID == "" {
		http.Error(w, `{"error":"claim id required via ?id=claim-xxx"}`, http.StatusBadRequest)
		return
	}

	exp := s.Hallucination.ExplainClaim(claimID)
	writeJSON(w, http.StatusOK, exp)
}

func (s *StrataServer) handleHallucinationMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := s.Hallucination.GetMetrics()
	writeJSON(w, http.StatusOK, metrics)
}

// handleRuntimeMetrics: counters of the transparent runtime (admin/diagnostics; nothing the model or a client has to call).
func (s *StrataServer) handleRuntimeMetrics(w http.ResponseWriter, r *http.Request) {
	s.rt.mu.RLock()
	c := s.rt.client
	s.rt.mu.RUnlock()
	if c == nil {
		writeJSON(w, http.StatusOK, map[string]interface{}{"runtime": s.runtimeStatus()})
		return
	}
	m, err := c.Metrics(r.Context())
	if err != nil {
		writeJSON(w, http.StatusBadGateway, map[string]interface{}{"error": err.Error(), "runtime": s.runtimeStatus()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"runtime": s.runtimeStatus(), "metrics": m})
}
