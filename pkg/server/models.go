package server

import (
	"context"
	"path/filepath"
	"strings"
	"sync"
)

var swapMu sync.Mutex // one model swap at a time

// modelSpecs maps every strata-*.json engine config's model_name (lower case) to its file.
func modelSpecs() map[string]string {
	out := map[string]string{}
	matches, _ := filepath.Glob("strata-*.json")
	for _, m := range matches {
		if sp, err := readEngineSpec(m); err == nil && sp.ModelName != "" {
			out[strings.ToLower(sp.ModelName)] = m
		}
	}
	return out
}

// useModel makes the model a request names the loaded one: when it is another engine config's model_name, the running
// engine is stopped and that config started (one card holds one model, so models swap rather than coexist). A name no
// config knows is left to the current model, as before. It refuses while a request is running.
func (s *StrataServer) useModel(ctx context.Context, requested string) error {
	file, ok := modelSpecs()[strings.ToLower(strings.TrimSpace(requested))]
	if !ok {
		return nil
	}
	swapMu.Lock()
	defer swapMu.Unlock()
	s.eng.mu.RLock()
	cur, problem := s.eng.source, s.eng.problem
	s.eng.mu.RUnlock()
	a, _ := filepath.Abs(file)
	if b, _ := filepath.Abs(cur); a == b && problem == "" {
		return nil
	}
	if n, _ := s.mon.live(true)["running"].(int); n > 0 {
		return &apiError{Status: 503, Type: "engine_unavailable", Msg: "another model is answering; try again when it is done"}
	}
	_ = s.EngineIPC.Stop()
	s.Config.ConfigFile = file
	s.Config.ModelName = strings.TrimSpace(requested)
	if err := s.StartEngine(ctx); err != nil {
		return &apiError{Status: 503, Type: "engine_unavailable", Msg: err.Error()}
	}
	s.setProblem("")
	if err := s.EngineIPC.WaitReady(ctx); err != nil {
		return &apiError{Status: 503, Type: "engine_unavailable", Msg: "the model did not load: " + err.Error()}
	}
	return nil
}
