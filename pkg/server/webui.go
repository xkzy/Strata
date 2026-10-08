// pkg/server/webui.go - the endpoints the web app (pkg/frontend/web) reads besides the model APIs: /config (the Settings
// view of the run config), /settings (the Chat settings other apps share).
package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"strata/pkg/runconfig"
)

type sharedState struct {
	mu       sync.RWMutex
	loadOnce sync.Once
	values   map[string]interface{}
}

// runConfigPath is the strata-<model>.json the engine was started from (empty: the server runs without one).
func (s *StrataServer) runConfigPath() string {
	s.eng.mu.RLock()
	defer s.eng.mu.RUnlock()
	if s.eng.source != "" {
		return s.eng.source
	}
	return s.Config.ConfigFile
}

func (s *StrataServer) sharedPath() string {
	p := s.runConfigPath()
	if p == "" {
		return ""
	}
	return strings.TrimSuffix(p, filepath.Ext(p)) + ".shared-settings.json"
}

// cleanSharedDefaults keeps only the known keys, each checked; an error names a bad one.
func cleanSharedDefaults(d map[string]interface{}) (map[string]interface{}, error) {
	out := map[string]interface{}{}
	for key, value := range d {
		if value == nil || value == "" {
			continue
		}
		f, isNum := value.(float64)
		whole := isNum && f == float64(int64(f))
		switch key {
		case "reasoning_effort":
			if v, _ := value.(string); v != "none" && v != "low" && v != "medium" && v != "high" {
				return nil, fmt.Errorf("reasoning_effort: none, low, medium or high")
			}
		case "temperature":
			if !isNum || f < 0 || f > 2 {
				return nil, fmt.Errorf("temperature: 0..2")
			}
		case "top_p":
			if !isNum || f <= 0 || f > 1 {
				return nil, fmt.Errorf("top_p: 0 < top_p <= 1")
			}
		case "top_k":
			if !whole || f < 1 || f > 64 {
				return nil, fmt.Errorf("top_k: an integer 1..64")
			}
		case "seed", "max_tokens":
			if !whole || f <= 0 {
				return nil, fmt.Errorf("%s: a positive integer", key)
			}
		case "experimental_speed_projection":
			if _, ok := value.(bool); !ok {
				return nil, fmt.Errorf("experimental_speed_projection: true or false")
			}
		default:
			return nil, fmt.Errorf("unknown setting %q", key)
		}
		out[key] = value
	}
	return out, nil
}

func (s *StrataServer) loadShared() {
	s.shared.loadOnce.Do(func() {
		p := s.sharedPath()
		if p == "" {
			return
		}
		raw, err := os.ReadFile(p)
		if err != nil {
			return
		}
		var m map[string]interface{}
		if json.Unmarshal(raw, &m) != nil {
			return
		}
		if clean, err := cleanSharedDefaults(m); err == nil {
			s.shared.mu.Lock()
			s.shared.values = clean
			s.shared.mu.Unlock()
		}
	})
}

func (s *StrataServer) sharedDefaults() map[string]interface{} {
	s.loadShared()
	s.shared.mu.RLock()
	defer s.shared.mu.RUnlock()
	out := make(map[string]interface{}, len(s.shared.values))
	for k, v := range s.shared.values {
		out[k] = v
	}
	return out
}

// setShared replaces the shared Chat settings (nil or empty: clients use their own again) and keeps them next to the
// run config so a restart keeps them.
func (s *StrataServer) setShared(d map[string]interface{}) (map[string]interface{}, error) {
	clean, err := cleanSharedDefaults(d)
	if err != nil {
		return nil, err
	}
	s.loadShared()
	s.shared.mu.Lock()
	s.shared.values = clean
	s.shared.mu.Unlock()
	if p := s.sharedPath(); p != "" {
		if len(clean) == 0 {
			_ = os.Remove(p)
		} else if b, err := json.MarshalIndent(clean, "", " "); err == nil {
			_ = os.WriteFile(p, b, 0o600)
		}
	}
	return clean, nil
}

// applySharedOA fills the shared Chat settings in where the request has none of its own.
func (s *StrataServer) applySharedOA(r *oaRequest) {
	d := s.sharedDefaults()
	if len(d) == 0 {
		return
	}
	num := func(k string) (float64, bool) { f, ok := d[k].(float64); return f, ok }
	if f, ok := num("temperature"); ok && r.Temperature == nil {
		r.Temperature = &f
	}
	if f, ok := num("top_p"); ok && r.TopP == nil {
		r.TopP = &f
	}
	if f, ok := num("top_k"); ok && r.TopK == nil {
		n := int(f)
		r.TopK = &n
	}
	if f, ok := num("seed"); ok && r.Seed == nil {
		n := int64(f)
		r.Seed = &n
	}
	if f, ok := num("max_tokens"); ok && r.MaxTokens == nil && r.MaxCompletionTokens == nil {
		n := int(f)
		r.MaxTokens = &n
	}
	if e, _ := d["reasoning_effort"].(string); e != "" && r.ReasoningEffort == "" && r.EnableThinking == nil {
		_, hasKwarg := r.ChatTemplateKwargs["reasoning_effort"]
		_, hasThinking := r.ChatTemplateKwargs["enable_thinking"]
		if !hasKwarg && !hasThinking {
			r.ReasoningEffort = e
		}
	}
}

func apiMessage(w http.ResponseWriter, status int, kind, msg string) {
	writeJSON(w, status, map[string]interface{}{"error": map[string]interface{}{"type": kind, "message": msg}})
}

// handleSettings is GET / POST /settings: "Use for other apps too".
func (s *StrataServer) handleSettings(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		d := s.sharedDefaults()
		writeJSON(w, http.StatusOK, map[string]interface{}{"shared": len(d) > 0, "defaults": d})
	case http.MethodPost:
		if !ownPage(w, r, "the settings can be changed") {
			return
		}
		var req struct {
			Defaults map[string]interface{} `json:"defaults"`
		}
		body, _ := io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<20))
		if len(body) > 0 && json.Unmarshal(body, &req) != nil {
			apiMessage(w, http.StatusBadRequest, "invalid_request_error", "the body must be a JSON object")
			return
		}
		d, err := s.setShared(req.Defaults)
		if err != nil {
			apiMessage(w, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"shared": len(d) > 0, "defaults": d})
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleRunConfig is GET / POST /config: a few documented keys of the run config (strata-<model>.json), #564.
func (s *StrataServer) handleRunConfig(w http.ResponseWriter, r *http.Request) {
	path := s.runConfigPath()
	if path == "" {
		apiMessage(w, http.StatusNotFound, "not_found", "this server was started without a run config")
		return
	}
	switch r.Method {
	case http.MethodGet:
		cfg, err := runconfig.LoadRaw(path)
		if err != nil {
			apiMessage(w, http.StatusInternalServerError, "server_error", "the run config cannot be read: "+err.Error())
			return
		}
		writeJSON(w, http.StatusOK, runconfig.View(cfg, path))
	case http.MethodPost:
		if !ownPage(w, r, "the run config can be changed") {
			return
		}
		var req struct {
			Set map[string]interface{} `json:"set"`
		}
		body, _ := io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<20))
		if err := json.Unmarshal(body, &req); err != nil {
			apiMessage(w, http.StatusBadRequest, "invalid_request_error", "the body must be a JSON object")
			return
		}
		s.cfgMu.Lock()
		defer s.cfgMu.Unlock()
		cfg, err := runconfig.LoadRaw(path)
		if err != nil {
			apiMessage(w, http.StatusInternalServerError, "server_error", "the run config cannot be read: "+err.Error())
			return
		}
		nw, changed, err := runconfig.Apply(cfg, req.Set)
		if err != nil {
			apiMessage(w, http.StatusBadRequest, "invalid_request_error", err.Error())
			return
		}
		if len(changed) > 0 {
			if _, err := runconfig.SaveRaw(path, nw); err != nil {
				apiMessage(w, http.StatusInternalServerError, "server_error", "the run config cannot be written: "+err.Error())
				return
			}
			fmt.Printf("[strata] the Settings view changed %s in %s; used from the next start\n", strings.Join(changed, ", "), filepath.Base(path))
		}
		view := runconfig.View(nw, path)
		view["changed"] = append([]string{}, changed...)
		writeJSON(w, http.StatusOK, view)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}
