package server

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"io"
	"log"
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
	"strata/pkg/modelschema"
	"strata/pkg/multitenant"
	"strata/pkg/processjob"
	"strata/pkg/resourcemanager"
	"strata/pkg/responses"
	"strata/pkg/runconfig"
	"strata/pkg/session"
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
	GuardPolicy   string // reply guard: off | monitor | regenerate (default; STRATA_GUARD and the run config's guard_policy also set it)
	ModelPath     string
	BinaryPath    string
	ConfigFile    string
	RuntimeMode   string // on | off | auto: the C++ transparent runtime (verification, virtual context)
	RuntimeBinary string
	WindowTokens  int    // physical context window the runtime fills (default 32768, at most the engine's context)
	BindHost      string // address the HTTP server listens on; empty is treated as loopback
	APIMonitor    bool   // keep the last requests' prompts and answers for the standalone Monitor page (opt-in)
	SessionDir    string // directory for persistent chat session storage
}

type StrataServer struct {
	Config          ServerConfig
	Registry        *modelschema.ModelRegistry
	MathRuntime     *mathruntime.MathRuntime
	LogicVerifier   *logicverifier.LogicVerifier
	Structured      *structured.StructuredValidator
	EngineIPC       *engineipc.EngineOrchestrator
	Telemetry       *telemetry.TelemetryCollector
	Responses       *responses.ResponsesHandler
	ConfigManager   *runconfig.ConfigManager
	ProcessJob      *processjob.ProcessJob
	Frontend        *frontend.FrontendHandler
	MultiTenant     *multitenant.MultiTenantManager
	VirtualContext  *virtualcontext.VirtualContextManager
	Memory          *virtualcontext.ConversationMemory // per-session recall for the tool interception
	SessionStore    *session.Store                     // persistent disk-backed web chat sessions
	started         time.Time
	AntiLoop        *antiloop.AntiLoopManager
	ResourceManager *resourcemanager.ResourceManager
	GenLoopDetector *generationloop.GenerationLoopDetector
	SetupManager    *installer.SetupManager
	Hallucination   *hallucination.HallucinationRuntime
	eng             engineState
	shared          sharedState
	mon             requestMonitor
	apiMon          apiMonitorState
	cfgMu           sync.Mutex // one run-config write at a time
	rt              rtState
	server          *http.Server
	activeCancels   map[string]context.CancelFunc
	cancelMu        sync.Mutex
}

func NewStrataServer(cfg ServerConfig) *StrataServer {
	cm := runconfig.NewConfigManager(cfg.ConfigFile)
	if cm != nil {
		saved := cm.Get()
		activeModel, activePath := saved.GetActiveModel()
		if cfg.ModelPath == "" && activePath != "" {
			cfg.ModelPath = activePath
		}
		if cfg.ConfigFile == "" && saved.LastConfigFile != "" {
			cfg.ConfigFile = saved.LastConfigFile
		}
		if (cfg.ModelName == "" || cfg.ModelName == "Qwen3.8-Flash-Next") && saved.LastModel != "" {
			cfg.ModelName = saved.LastModel
		} else if (cfg.ModelName == "" || cfg.ModelName == "Qwen3.8-Flash-Next") && activeModel != "" {
			cfg.ModelName = activeModel
		}
		if cfg.MaxContext <= 0 && saved.MaxContext > 0 {
			cfg.MaxContext = saved.MaxContext
		}
		if cfg.VirtualLimit <= 0 && saved.VirtualLimit > 0 {
			cfg.VirtualLimit = saved.VirtualLimit
		}
		if cfg.BindHost == "" && saved.Host != "" {
			cfg.BindHost = saved.Host
		}
		if (cfg.Port <= 0 || cfg.Port == 8080) && saved.Port > 0 {
			cfg.Port = saved.Port
		}
		if cfg.BinaryPath == "" && saved.EngineBinary != "" {
			cfg.BinaryPath = saved.EngineBinary
		}
	}

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

	engineCfg := engineipc.EngineConfig{
		BinaryPath:  cfg.BinaryPath,
		ModelPath:   cfg.ModelPath,
		ContextSize: cfg.MaxContext,
	}
	orchestrator := engineipc.NewEngineOrchestrator(engineCfg)

	tc := telemetry.NewTelemetryCollector()
	tc.Start()

	hr := hallucination.NewRuntime(".", hallucination.DefaultSafetyControllerOptions())
	reg := modelschema.NewRegistry()
	primaryID := cfg.ModelName
	if primaryID == "" {
		primaryID = "Qwen3.8-Flash-Next"
	}
	_ = reg.RegisterRuntime(modelschema.ModelRuntimeProfile{
		ID:           primaryID,
		ModelPath:    cfg.ModelPath,
		EngineBinary: cfg.BinaryPath,
		MaxContext:   cfg.MaxContext,
		VirtualLimit: cfg.VirtualLimit,
		Enabled:      true,
		Status:       "ready",
	})
	if cm != nil {
		for _, m := range cm.Get().AllModels() {
			_ = reg.RegisterRuntime(modelschema.ModelRuntimeProfile{
				ID:                   m.ID,
				ModelPath:            m.ModelPath,
				EngineBinary:         m.EngineBinary,
				MaxContext:           m.MaxContext,
				VirtualLimit:         m.VirtualLimit,
				Abliterate:           m.Abliterate,
				AbliterationStrength: m.AbliterationStrength,
				AbliterationVector:   m.AbliterationVector,
				Enabled:              m.Enabled,
				Status:               "ready",
			})
		}
	}

	sessDir := cfg.SessionDir
	if sessDir == "" {
		sessDir = filepath.Join(installer.GetDataDir(), "sessions")
	}
	sessStore, err := session.NewStore(sessDir)
	if err != nil {
		sessStore, _ = session.NewStore(filepath.Join(os.TempDir(), "strata-sessions"))
	}

	s := &StrataServer{
		Config:          cfg,
		Registry:        reg,
		MathRuntime:     mr,
		LogicVerifier:   lv,
		Structured:      structured.NewStructuredValidator(),
		EngineIPC:       orchestrator,
		Telemetry:       tc,
		Responses:       responses.NewResponsesHandler(),
		ConfigManager:   cm,
		ProcessJob:      processjob.GlobalJob(),
		Frontend:        frontend.NewFrontendHandler(),
		MultiTenant:     multitenant.NewMultiTenantManager(),
		VirtualContext:  vc,
		Memory:          virtualcontext.NewConversationMemory(0, 0),
		SessionStore:    sessStore,
		started:         time.Now(),
		AntiLoop:        antiloop.Instance(),
		ResourceManager: resourcemanager.NewResourceManager(),
		GenLoopDetector: generationloop.NewGenerationLoopDetector(nil),
		SetupManager:    installer.NewSetupManager(),
		Hallucination:   hr,
		activeCancels:   make(map[string]context.CancelFunc),
	}
	s.mon.init()
	tc.SetTokSource(s.mon.currentTokS)
	return s
}

func (s *StrataServer) Router() http.Handler {
	mux := http.NewServeMux()

	// OpenAI / Anthropic APIs
	mux.HandleFunc("/v1/chat/completions", s.recordAPI(s.handleChatCompletions))
	mux.HandleFunc("/chat/completions", s.recordAPI(s.handleChatCompletions))
	mux.HandleFunc("/v1/completions", s.recordAPI(s.handleCompletions))
	mux.HandleFunc("/completions", s.recordAPI(s.handleCompletions))
	mux.HandleFunc("/v1/messages", s.recordAPI(s.handleAnthropicMessages))
	mux.HandleFunc("/messages", s.recordAPI(s.handleAnthropicMessages))
	mux.HandleFunc("/v1/responses", s.recordAPI(s.handleResponsesAPI))
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/models/", s.handleModels)
	mux.HandleFunc("/models", s.handleModels)
	mux.HandleFunc("/models/", s.handleModels)
	mux.HandleFunc("/api/models/switch", s.handleModelSwitch)

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
	mux.HandleFunc("/api/setup/resolve", s.handleSetupResolve)
	mux.HandleFunc("/api/setup/configure", s.handleSetupConfigure)
	mux.HandleFunc("/api/setup/download", s.handleSetupDownload)
	mux.HandleFunc("/api/setup/status", s.handleSetupStatus)
	mux.HandleFunc("/api/setup/cancel", s.handleSetupCancel)
	mux.HandleFunc("/api/setup/delete", s.handleSetupDelete)

	// Config Settings (#564)
	mux.HandleFunc("/config", s.handleRunConfig)
	mux.HandleFunc("/settings", s.handleSettings)

	// The standalone Monitor page
	mux.HandleFunc("/api/requests", s.handleAPIRequests)
	mux.HandleFunc("/load", s.handleLoad)
	mux.HandleFunc("/unload", s.handleUnload)

	// Math Runtime Endpoints (#20)
	mux.HandleFunc("/v1/strata/math/evaluate", s.handleMathEvaluate)
	mux.HandleFunc("/v1/strata/math/verify", s.handleMathVerify)
	mux.HandleFunc("/v1/strata/math/intercept", s.handleMathIntercept)
	mux.HandleFunc("/v1/strata/math/metrics", s.handleMathMetrics)
	mux.HandleFunc("/v1/strata/math/backends", s.handleMathBackends)
	mux.HandleFunc("/v1/strata/math/theorems", s.handleMathTheorems)
	mux.HandleFunc("/v1/strata/math/theorems/verify", s.handleMathVerifyTheorem)

	// Unified CAS Endpoints
	mux.HandleFunc("/v1/strata/cas/solve", s.handleCASSolve)
	mux.HandleFunc("/v1/strata/cas/constants", s.handleCASConstants)
	mux.HandleFunc("/v1/strata/cas/formulas", s.handleCASFormulas)

	// Logic Verification Runtime Endpoints
	mux.HandleFunc("/v1/strata/verify", s.handleLogicVerify)
	mux.HandleFunc("/v1/strata/verify/claims", s.handleLogicVerifyClaims)
	mux.HandleFunc("/v1/strata/verify/metrics", s.handleLogicVerifyMetrics)

	// Hallucination Runtime Endpoints
	mux.HandleFunc("/v1/strata/hallucination/evaluate", s.handleHallucinationEvaluate)
	mux.HandleFunc("/v1/strata/hallucination/claims", s.handleHallucinationClaims)
	mux.HandleFunc("/v1/strata/hallucination/metrics", s.handleHallucinationMetrics)
	mux.HandleFunc("/v1/strata/guard/metrics", s.handleGuardMetrics)

	// Multi-Tenant & Web Chat Session Management
	mux.HandleFunc("/api/sessions", s.handleSessions)
	mux.HandleFunc("/api/sessions/", s.handleSessionDetail)
	mux.HandleFunc("/api/strata/sessions/delete", s.handleSessionDelete)
	mux.HandleFunc("/v1/strata/sessions", s.handleSessions)
	mux.HandleFunc("/v1/strata/sessions/fork", s.handleSessionFork)
	mux.HandleFunc("/v1/strata/sessions/delete", s.handleSessionDelete)
	mux.HandleFunc("/v1/strata/tenants/metrics", s.handleTenantMetrics)

	// Context & Tool Stores
	mux.HandleFunc("/v1/strata/context", s.handleContextStats)
	mux.HandleFunc("/v1/strata/context/query", s.handleContextQuery)
	mux.HandleFunc("/v1/strata/memory", s.handleMemory)
	mux.HandleFunc("/v1/strata/tools/retrieve", s.handleToolsRetrieve)

	// Anti-Loop Guard
	mux.HandleFunc("/v1/strata/runtime/metrics", s.handleRuntimeMetrics)
	mux.HandleFunc("/v1/strata/guard", s.handleGuardStats)
	mux.HandleFunc("/v1/strata/guard/check", s.handleGuardCheck)
	mux.HandleFunc("/v1/strata/guard/outcome", s.handleGuardOutcome)

	// Root Web UI
	mux.HandleFunc("/", s.handleRoot)

	return s.corsMiddleware(mux)
}

// publicPath says whether a request needs no API key: the health probes, the web UI shell and the read-only setup
// queries. Everything that changes state (setup configure / download / cancel, generation, config) needs the key.
// authorized reports whether the request carries the API key; with no key configured every request is (the server then
// listens on loopback only, see CheckBind).
func (s *StrataServer) authorized(r *http.Request) bool {
	if s.Config.APIKey == "" {
		return true
	}
	gotAuth := []byte(r.Header.Get("Authorization"))
	wantAuth := []byte("Bearer " + s.Config.APIKey)
	gotKey := []byte(r.Header.Get("X-Api-Key"))
	if len(gotKey) == 0 {
		gotKey = []byte(r.Header.Get("x-api-key"))
	}
	wantKey := []byte(s.Config.APIKey)
	authOK := subtle.ConstantTimeCompare(gotAuth, wantAuth) == 1
	keyOK := subtle.ConstantTimeCompare(gotKey, wantKey) == 1
	return authOK || keyOK
}

// publicHealthFields are the parts of the status that carry file paths and the engine's own log text. The public
// routes (/health, /api/setup/status) leave them out unless the caller sent the API key.
var publicHealthFields = []string{"error", "engine_config", "template_warning", "last_config_file"}

func publicPath(r *http.Request) bool {
	// the page and its scripts, styles and fonts: a browser fetches them before the user can type a key
	if strings.HasPrefix(r.URL.Path, "/web/") || strings.HasPrefix(r.URL.Path, "/fonts/") {
		return r.Method == http.MethodGet
	}
	switch r.URL.Path {
	case "/health", "/api/health", "/", "/monitor", "/monitor.html", "/api-monitor", "/favicon.ico":
		return true
	case "/api/setup/hardware", "/api/setup/models", "/api/setup/status", "/api/setup/resolve":
		return true
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
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Tenant-ID, X-User-ID, X-Session-ID")
		}
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		if s.Config.APIKey != "" && !publicPath(r) && !s.authorized(r) {
			http.Error(w, `{"error":{"message":"Unauthorized","type":"authentication_error"}}`, http.StatusUnauthorized)
			return
		}

		if r.Method == http.MethodPost || r.Method == http.MethodPut || r.Method == http.MethodPatch {
			if limit := bodyLimit(r.URL.Path); limit > 0 {
				r.Body = http.MaxBytesReader(w, r.Body, limit)
			}
		}
		next.ServeHTTP(w, r)
	})
}

// bodyLimit is how much request body an endpoint may read into memory (0: it applies its own limit). The model APIs
// carry whole conversations and cap themselves at 64 MiB; everything else (math, theorems, CAS, verification, the guard,
// sessions, setup, settings) takes small JSON, so a larger body is refused instead of read: without a limit, one
// request could use as much memory as its sender cared to send.
func bodyLimit(path string) int64 {
	switch {
	case path == "/v1/chat/completions", path == "/chat/completions", path == "/v1/completions", path == "/completions", path == "/v1/messages", path == "/messages",
		path == "/v1/messages/count_tokens", strings.HasPrefix(path, "/v1/responses"):
		return 0
	case strings.HasPrefix(path, "/api/sessions"), strings.HasPrefix(path, "/v1/strata/sessions"):
		return 32 << 20
	}
	return 1 << 20
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

	seenModels := make(map[string]bool)
	modelsList := make([]map[string]interface{}, 0)

	addModel := func(id, name string) {
		if id == "" {
			return
		}
		if name == "" {
			name = id
		}
		if !seenModels[id] {
			seenModels[id] = true
			modelsList = append(modelsList, map[string]interface{}{
				"id":   id,
				"name": name,
			})
		}
	}

	if s.Config.ModelName != "" {
		addModel(s.Config.ModelName, s.Config.ModelName)
	}
	if s.Registry != nil {
		for _, m := range s.Registry.ListManifests() {
			addModel(m.ID, m.Name)
		}
	}
	if s.ConfigManager != nil {
		for id, m := range s.ConfigManager.Get().AllModels() {
			name := m.Name
			if name == "" {
				name = id
			}
			addModel(id, name)
		}
	}
	if matches, err := filepath.Glob("strata-*.json"); err == nil {
		for _, m := range matches {
			if sp, err := readEngineSpec(m); err == nil && sp.ModelName != "" {
				addModel(sp.ModelName, sp.ModelName)
			}
		}
	}
	out["models"] = modelsList

	if s.ConfigManager != nil {
		cfg := s.ConfigManager.Get()
		out["abliterate"] = cfg.Abliterate
		out["abliteration_strength"] = cfg.AbliterationStrength
		if cfg.LastConfigFile != "" {
			out["last_config_file"] = cfg.LastConfigFile
		}
	}
	for k, v := range s.engineStatus() {
		out[k] = v
	}
	if out["loaded"] != true {
		out["status"] = "degraded" // the server is up, the model is not
	}
	if !s.authorized(r) { // /health needs no key, so it must not hand out paths and engine log lines
		for _, k := range publicHealthFields {
			delete(out, k)
		}
		if rt, ok := out["runtime"].(map[string]interface{}); ok {
			delete(rt, "error")
			delete(rt, "note")
		}
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

func (s *StrataServer) modelEntry(id string) map[string]interface{} {
	ctx := s.Config.MaxContext
	if ctx <= 0 {
		ctx = 32768
	}
	virt := s.Config.VirtualLimit
	if virt < ctx {
		virt = 2000000
	}
	created := s.started.Unix()
	if created <= 0 {
		created = time.Now().Unix()
	}
	if s.Registry != nil {
		return s.Registry.OpenAIEntry(id, created, ctx, virt)
	}
	return map[string]interface{}{
		"id":                      id,
		"object":                  "model",
		"created":                 created,
		"owned_by":                "strata",
		"context_window":          virt,
		"max_context_length":      virt,
		"context_length":          virt,
		"max_tokens":              virt,
		"max_input_tokens":        virt,
		"max_output_tokens":       16384,
		"physical_context_window": ctx,
		"physical_context_length": ctx,
		"virtual_context_limit":   virt,
		"virtual_context_tokens":  virt,
		"permission": []map[string]interface{}{
			{
				"id":                   "modelperm-" + id,
				"object":               "model_permission",
				"created":              created,
				"allow_create_engine":  false,
				"allow_sampling":       true,
				"allow_logprobs":       true,
				"allow_search_indices": false,
				"allow_view":           true,
				"allow_fine_tuning":    false,
				"organization":         "*",
				"group":                nil,
				"is_blocking":          false,
			},
		},
	}
}

func (s *StrataServer) handleModels(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/v1/models")
	path = strings.TrimPrefix(path, "/models")
	path = strings.TrimPrefix(path, "/")

	if path != "" {
		writeJSON(w, http.StatusOK, s.modelEntry(path))
		return
	}

	var data []map[string]interface{}
	if s.Registry != nil {
		data = s.Registry.OpenAIList(s.started.Unix(), s.Config.VirtualLimit)
	} else {
		primaryID := s.Config.ModelName
		if primaryID == "" {
			primaryID = "qwen3.8-flash-next"
		}
		data = append(data, s.modelEntry(primaryID))
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"object": "list",
		"data":   data,
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
	info := map[string]interface{}{
		"name":        s.Config.ModelName,
		"path":        s.Config.ModelPath,
		"max_context": s.Config.MaxContext,
		"loaded":      s.engineStatus()["loaded"],
	}
	if s.ConfigManager != nil {
		cfg := s.ConfigManager.Get()
		info["abliterate"] = cfg.Abliterate
		info["abliteration_strength"] = cfg.AbliterationStrength
		if cfg.LastConfigFile != "" {
			info["last_config_file"] = cfg.LastConfigFile
		}
	}
	writeJSON(w, http.StatusOK, info)
}

func (s *StrataServer) handleModelSwitch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Model                string   `json:"model"`
		Config               string   `json:"config"`
		Abliterate           *bool    `json:"abliterate,omitempty"`
		AbliterationStrength *float64 `json:"abliteration_strength,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error":{"message":"Invalid JSON payload"}}`, http.StatusBadRequest)
		return
	}

	targetModel := strings.TrimSpace(req.Model)
	if targetModel == "" && req.Config == "" {
		http.Error(w, `{"error":{"message":"Model name or config path is required"}}`, http.StatusBadRequest)
		return
	}

	s.cfgMu.Lock()
	defer s.cfgMu.Unlock()

	// 1. Stop current engine if running
	if s.EngineIPC != nil {
		_ = s.EngineIPC.Stop()
	}

	// 2. Update abliteration settings if provided
	if s.ConfigManager != nil {
		cfg := s.ConfigManager.Get()
		if req.Abliterate != nil {
			cfg.Abliterate = *req.Abliterate
		}
		if req.AbliterationStrength != nil {
			cfg.AbliterationStrength = *req.AbliterationStrength
		}
		s.ConfigManager.Update(cfg)
	}

	// 3. Resolve target config file and model name
	targetConfig := req.Config
	if targetConfig != "" {
		if realPath, sp, err := resolveEngineConfigFile(targetConfig); err == nil {
			targetConfig = realPath
			if sp.ModelName != "" {
				targetModel = sp.ModelName
			}
		} else {
			if fi, serr := os.Stat(targetConfig); serr == nil && fi.IsDir() {
				s.Config.ModelPath = targetConfig
			}
			targetConfig = ""
		}
	}

	if targetConfig == "" {
		matches, _ := filepath.Glob("strata-*.json")
		cleanTarget := strings.ToLower(strings.ReplaceAll(strings.ReplaceAll(targetModel, " ", "_"), "/", "_"))
		cleanTarget = strings.TrimSuffix(cleanTarget, "_(local_pack)")
		cleanTarget = strings.ReplaceAll(cleanTarget, "-", "_")

		for _, m := range matches {
			if sp, err := readEngineSpec(m); err == nil {
				pack := argValue(sp.Args, "--pack")
				cleanM := strings.ToLower(strings.ReplaceAll(m, "-", "_"))
				if strings.EqualFold(sp.ModelName, targetModel) ||
					(cleanTarget != "" && (strings.Contains(cleanM, cleanTarget) || strings.Contains(cleanTarget, strings.TrimSuffix(cleanM, ".json")))) ||
					(pack != "" && (pack == req.Config || strings.EqualFold(filepath.Base(pack), filepath.Base(req.Config)))) {
					targetConfig = m
					if sp.ModelName != "" {
						targetModel = sp.ModelName
					}
					break
				}
			}
		}

		if targetConfig == "" && s.ConfigManager != nil {
			cfg := s.ConfigManager.Get()
			if entry, ok := cfg.GetModelEntry(targetModel); ok && entry.ModelPath != "" {
				s.Config.ModelPath = entry.ModelPath
				if entry.MaxContext > 0 {
					s.Config.MaxContext = entry.MaxContext
				}
			}
		}
	}

	if targetModel != "" {
		s.Config.ModelName = targetModel
	}
	s.Config.ConfigFile = targetConfig

	// Persist last session config
	if s.ConfigManager != nil {
		cfg := s.ConfigManager.Get()
		cfg.LastConfigFile = targetConfig
		cfg.LastModel = s.Config.ModelName
		s.ConfigManager.Update(cfg)
		_ = s.ConfigManager.SaveToFile("")
	}

	// 4. Start the new engine
	startErr := s.StartEngine(context.Background())
	resp := map[string]interface{}{
		"status": "switching",
		"model":  s.Config.ModelName,
		"config": s.Config.ConfigFile,
	}
	if startErr != nil {
		resp["warning"] = startErr.Error()
	}

	writeJSON(w, http.StatusOK, resp)
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

	engStatus := s.engineStatus()
	loaded, _ := engStatus["loaded"].(bool)
	liveMap := s.mon.live(loaded)
	engState, _ := engStatus["engine"].(string)
	overlayEngineState(liveMap, engState)
	if liveMap["state"] == "error" {
		liveMap["detail"], _ = engStatus["error"].(string)
	}
	requests, kept, totals := s.mon.view(r.URL.Query().Get("requests") == "all")
	var engInfo map[string]string
	if s.EngineIPC != nil {
		engInfo = s.EngineIPC.Info()
	}
	mergeEngineInfo(engineMap, engInfo)
	var kvTotals engineipc.KVTotals
	if s.EngineIPC != nil {
		requests = annotateKVCost(requests, s.EngineIPC.KVLedger().Events())
		kvTotals = s.EngineIPC.KVLedger().Totals()
	}
	cc := conversationCacheView(engInfo, requests, totals)
	cc["ledger"] = kvLedgerView(kvTotals)
	if s.EngineIPC != nil {
		cc["parks"], cc["restores"] = kvTotals.Parks, kvTotals.Restores
		if kvTotals.HasParked { // the engine's own count after its latest park or restore
			cc["parked"], cc["bytes"] = int(kvTotals.Parked), int(kvTotals.ParkedBytes)
		}
		sum := 0 // the monitor records no reuse; the listed requests carry what the engine's restores gave back
		for _, r := range requests {
			sum += asInt(r["reused"])
		}
		if sum > asInt(cc["reused_tokens"]) {
			cc["reused_tokens"] = sum
		}
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"conversation_cache": cc,
		"engine":             engineMap,
		"live":               liveMap,
		"hardware":           nowMap,
		"hardware_static":    staticMap,
		"history":            histMap,
		"requests":           requests,
		"requests_kept":      kept,
		"totals":             totals,
		"resource":           s.ResourceManager.Metrics(),
		"anti_loop":          s.AntiLoop.Stats(),
		"context":            s.VirtualContext.Stats(),
		"math":               s.MathRuntime.GetStats(),
		"time":               time.Now().Unix(),
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
		"cas_available":          s.MathRuntime.CasBackend.Available,
		"cas_version":            s.MathRuntime.CasBackend.Version,
		"mathics_available":      s.MathRuntime.MathicsBackend.Available,
		"mathics_version":        s.MathRuntime.MathicsBackend.Version,
		"sage_available":         s.MathRuntime.SageBackend.Available,
		"sage_version":           s.MathRuntime.SageBackend.Version,
		"fast_numeric_available": true,
		"fast_numeric_version":   s.MathRuntime.FastBackend.Version,
	})
}

func (s *StrataServer) handleMathTheorems(w http.ResponseWriter, r *http.Request) {
	reg := mathruntime.GetTheoremRegistry()
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"count":    len(reg.ListAll()),
		"theorems": reg.ListAll(),
	})
}

func (s *StrataServer) handleMathVerifyTheorem(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req struct {
		Name        string   `json:"name"`
		Args        []string `json:"args"`
		Assumptions string   `json:"assumptions"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid JSON body", http.StatusBadRequest)
		return
	}
	reg := mathruntime.GetTheoremRegistry()
	res := reg.VerifyTheorem(req.Name, req.Args, req.Assumptions)
	writeJSON(w, http.StatusOK, res)
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
				"scope":      scopeKey(scope), // QueryScoped shows a record only to the scope that made it
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
	if body.TopK > 50 {
		body.TopK = 50
	}
	// only the caller's own records (the scope from the headers; callers that send none share the default scope)
	hits := s.VirtualContext.QueryScoped(scopeKey(multitenant.FromHeaders(r.Header, nil)), body.Query, body.TopK)
	if hits == nil {
		hits = []map[string]interface{}{}
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"query": body.Query,
		"hits":  hits,
	})
}

// handleMemory is DELETE /v1/strata/memory: forget what the automatic recall kept of the caller's session
// (the scope of the X-Tenant-ID / X-User-ID / X-Session-ID headers and the credential).
func (s *StrataServer) handleMemory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		w.Header().Set("Allow", "DELETE")
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	n := 0
	if s.Memory != nil {
		n = s.Memory.Forget(scopeKey(multitenant.FromHeaders(r.Header, nil)))
	}
	writeJSON(w, http.StatusOK, map[string]int{"forgotten": n})
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

func (s *StrataServer) handleRoot(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/monitor", "/monitor.html", "/api-monitor", "/web/monitor.html", "/web/monitor.js":
		if !s.apiMonitorOn() { // the standalone page exists only when the monitor is on (it keeps prompts and answers)
			http.NotFound(w, r)
			return
		}
	}
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
		"data_dir":    installer.GetDataDir(),
		"models_dir":  installer.GetModelsDir(),
		"packs_dir":   installer.GetPacksDir(),
	})
}

func (s *StrataServer) handleSetupConfigure(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		ModelName            string   `json:"model_name"`
		ModelPath            string   `json:"model_path"`
		MaxContext           int      `json:"max_context"`
		VirtualLimit         int      `json:"virtual_limit"`
		PoolWorkers          int      `json:"pool_workers"`
		PcieFrac             *float64 `json:"pcie_frac"`
		GPUs                 string   `json:"gpus"`
		Abliterate           *bool    `json:"abliterate,omitempty"`
		AbliterationStrength *float64 `json:"abliteration_strength,omitempty"`
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
		if req.PoolWorkers > 0 {
			cfg.PoolWorkers = req.PoolWorkers
		}
		if req.PcieFrac != nil {
			cfg.PcieFrac = *req.PcieFrac
		}
		if req.GPUs != "" {
			cfg.GPUs = req.GPUs
		}
		if req.Abliterate != nil {
			cfg.Abliterate = *req.Abliterate
		}
		if req.AbliterationStrength != nil {
			cfg.AbliterationStrength = *req.AbliterationStrength
		}
		s.ConfigManager.Update(cfg)
		s.saveConfig()
	}

	resConfig := map[string]interface{}{
		"model_name":    s.Config.ModelName,
		"model_path":    s.Config.ModelPath,
		"max_context":   s.Config.MaxContext,
		"virtual_limit": s.Config.VirtualLimit,
		"pool_workers":  req.PoolWorkers,
		"pcie_frac":     req.PcieFrac,
		"gpus":          req.GPUs,
	}
	if s.ConfigManager != nil {
		cfg := s.ConfigManager.Get()
		resConfig["abliterate"] = cfg.Abliterate
		resConfig["abliteration_strength"] = cfg.AbliterationStrength
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status": "configured",
		"config": resConfig,
	})
}

func (s *StrataServer) handleSetupResolve(w http.ResponseWriter, r *http.Request) {
	input := r.URL.Query().Get("q")
	if input == "" && r.Method == http.MethodPost {
		var req struct {
			Input string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err == nil {
			input = req.Input
		}
	}
	if strings.TrimSpace(input) == "" {
		http.Error(w, `{"error":{"message":"input query or JSON body is required"}}`, http.StatusBadRequest)
		return
	}

	opt, err := installer.ResolveHuggingFaceInput(input)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, opt)
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

	// Only models of the built-in catalog or valid Hugging Face repos/URLs can be downloaded.
	var req struct {
		ModelName   string `json:"model_name"`
		DownloadURL string `json:"-"` // never read directly from client body for untrusted paths
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
		if res, err := installer.ResolveHuggingFaceInput(req.ModelName); err == nil {
			choice = res
		} else {
			http.Error(w, `{"error":{"message":"unknown model: choose one from /api/setup/models or provide a valid Hugging Face URL/repo"}}`, http.StatusBadRequest)
			return
		}
	}
	req.DownloadURL = choice.DownloadURL
	if choice.DownloadURL == "" { // a model already on this machine (catalog entry with a local path)
		req.DestPath = choice.Filename
	} else {
		req.DestPath = filepath.Join(installer.GetModelsDir(), filepath.Base(choice.Filename))
		if !installer.IsSafeModelPath(req.DestPath) {
			http.Error(w, `{"error":{"message":"invalid destination"}}`, http.StatusBadRequest)
			return
		}
	}

	// If the file or pack already exists locally and has all shards, activate it immediately
	existingGB := installer.TotalShardsSizeGB(req.DestPath)
	if installer.AllShardsExist(req.DestPath) && (choice.SizeGB == 0 || existingGB >= choice.SizeGB*0.95) {
		s.Config.ModelName = req.ModelName
		s.Config.ModelPath = req.DestPath
		if s.ConfigManager != nil {
			cfg := s.ConfigManager.Get()
			cfg.Model = s.Config.ModelName
			cfg.ModelPath = s.Config.ModelPath
			s.ConfigManager.Update(cfg)
			s.saveConfig()
		}
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

	err := s.SetupManager.StartDownloadWithOptions(installer.DownloadOptions{
		ModelName: req.ModelName,
		URL:       req.DownloadURL,
		DestPath:  req.DestPath,
		SizeGB:    choice.SizeGB,
		Shards:    choice.Shards,
	})
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error":  map[string]interface{}{"message": err.Error()},
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
	dl, activePath := s.SetupManager.GetStatus(), s.Config.ModelPath
	if !s.authorized(r) { // a public route: no file paths or error text without the key
		dl.DestPath, dl.Error, activePath = "", "", ""
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"download":     dl,
		"active_model": s.Config.ModelName,
		"active_path":  activePath,
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

type SetupDeleteRequest struct {
	ModelName string `json:"model_name"`
	Filename  string `json:"filename"`
	Path      string `json:"path"`
}

func (s *StrataServer) handleSetupDelete(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost && r.Method != http.MethodDelete {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req SetupDeleteRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && err != io.EOF {
		http.Error(w, fmt.Sprintf(`{"error":{"message":"invalid JSON: %v"}}`, err), http.StatusBadRequest)
		return
	}

	target := req.Path
	if target == "" {
		target = req.Filename
	}

	var matchedName string = req.ModelName
	if target == "" && req.ModelName != "" {
		for _, m := range installer.GetAvailableModels() {
			if strings.EqualFold(m.Name, req.ModelName) || strings.EqualFold(m.RepoID, req.ModelName) {
				target = m.Filename
				matchedName = m.Name
				break
			}
		}
	}

	if target == "" {
		http.Error(w, `{"error":{"message":"model target path or name required"}}`, http.StatusBadRequest)
		return
	}

	// If download is currently in progress for this model or file, cancel it first
	status := s.SetupManager.GetStatus()
	if status.Status == installer.StatusDownloading {
		isTargetDownload := false
		if req.ModelName != "" && strings.EqualFold(status.ModelName, req.ModelName) {
			isTargetDownload = true
		}
		if target != "" && (strings.Contains(status.DestPath, target) || strings.Contains(target, filepath.Base(status.DestPath))) {
			isTargetDownload = true
		}
		if isTargetDownload {
			s.SetupManager.CancelDownload()
			time.Sleep(150 * time.Millisecond)
		}
	}

	freedBytes, deletedFiles, err := installer.DeleteModelFiles(target)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": map[string]interface{}{
				"message": err.Error(),
			},
		})
		return
	}

	wasActive := false
	if s.Config.ModelName == matchedName || s.Config.ModelPath == target || (target != "" && strings.Contains(s.Config.ModelPath, filepath.Base(target))) {
		wasActive = true
		s.Config.ModelName = ""
		s.Config.ModelPath = ""
		if s.ConfigManager != nil {
			cfg := s.ConfigManager.Get()
			cfg.Model = ""
			cfg.ModelPath = ""
			s.ConfigManager.Update(cfg)
			s.saveConfig()
		}
	}

	freedGB := float64(freedBytes) / (1024 * 1024 * 1024)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":        "deleted",
		"model_name":    matchedName,
		"deleted_files": deletedFiles,
		"freed_bytes":   freedBytes,
		"freed_gb":      float64(int(freedGB*100+0.5)) / 100.0,
		"was_active":    wasActive,
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

// saveConfig writes the run config to disk. A failure (full disk, read-only folder) is logged: the change still holds
// until the server stops, but it must not look saved.
func (s *StrataServer) saveConfig() {
	if err := s.ConfigManager.SaveToFile(""); err != nil {
		log.Printf("strata: cannot save the run config, the change is lost at the next restart: %v", err)
	}
}
