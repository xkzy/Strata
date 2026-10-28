package server

import (
	"context"
	"log"
	"path/filepath"
	"strings"
	"sync"
)

// modelMu orders requests and model swaps: a request holds the read side from before its prompt is built until its reply
// is over, so the engine, tokenizer and template it started with cannot change under it; a swap takes the write side,
// which waits for the running requests and holds back new ones until the new model is ready.
var modelMu sync.RWMutex

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

// wantsSwap reports the config file to start when requested names a model other than the loaded one ("" = keep).
func (s *StrataServer) wantsSwap(requested string) string {
	file, ok := modelSpecs()[strings.ToLower(strings.TrimSpace(requested))]
	if !ok {
		return "" // a name no config knows is left to the current model
	}
	s.eng.mu.RLock()
	cur, problem := s.eng.source, s.eng.problem
	s.eng.mu.RUnlock()
	a, _ := filepath.Abs(file)
	if b, _ := filepath.Abs(cur); a == b && problem == "" {
		return ""
	}
	return file
}

// pinModel makes the model a request names the loaded one and keeps it loaded until the returned func is called. Models
// swap rather than coexist (one card): the running engine is stopped and the other config's started, after the requests
// in flight have finished.
func (s *StrataServer) pinModel(ctx context.Context, requested string) (func(), *apiError) {
	for attempt := 0; attempt < 3; attempt++ {
		modelMu.RLock()
		file := s.wantsSwap(requested)
		if file == "" {
			return modelMu.RUnlock, nil
		}
		modelMu.RUnlock()
		modelMu.Lock()
		if s.wantsSwap(requested) != "" { // another request may have swapped while this one waited
			_ = s.EngineIPC.Stop()
			s.Config.ConfigFile = file
			s.Config.ModelName = strings.TrimSpace(requested)
			if err := s.StartEngine(ctx); err != nil {
				modelMu.Unlock()
				return nil, &apiError{Status: 503, Type: "engine_unavailable", Msg: err.Error()}
			}
			s.setProblem("")
			if err := s.EngineIPC.WaitReady(ctx); err != nil {
				modelMu.Unlock()
				return nil, &apiError{Status: 503, Type: "engine_unavailable", Msg: "the model did not load: " + err.Error()}
			}
		}
		modelMu.Unlock()
	}
	return nil, &apiError{Status: 503, Type: "engine_unavailable", Msg: "the model could not be switched"}
}

// autoModels is the order model "auto" tries: the small fast one first, the dense model only as the last resort.
var autoModels = []string{"flint-preview", "qwen3.8-flash-next-coder-iq1_m", "qwen3.8-27b-dense"}

// autoFailed says why a collected reply counts as a failure ("" if it does not); the checks are the router's (pkg/router).
func autoFailed(err error, evs []genEvent) string {
	if err != nil {
		return err.Error()
	}
	text, finish := false, ""
	for _, ev := range evs {
		if ev.Err != nil {
			return ev.Err.Error()
		}
		text = text || (ev.Kind == "content" && ev.Text != "") || ev.Kind == "tool"
		if ev.End {
			finish = ev.Finish
		}
	}
	switch {
	case !text:
		return "empty reply"
	case finish == "length":
		return "cut off at max_tokens"
	}
	return ""
}

// startChat runs a request. Model "auto" answers on the first model in autoModels that has a config and goes on to the
// next when the reply failed; a streamed reply is already on its way to the client, so it stays on the first model.
func (s *StrataServer) startChat(ctx context.Context, spec genSpec) (<-chan genEvent, int, error) {
	if !strings.EqualFold(strings.TrimSpace(spec.Model), "auto") {
		return s.startChatOne(ctx, spec)
	}
	have := modelSpecs()
	var chain []string
	for _, m := range autoModels {
		if _, ok := have[m]; ok {
			chain = append(chain, m)
		}
	}
	if len(chain) == 0 {
		spec.Model = "" // nothing to choose from: the loaded model
		return s.startChatOne(ctx, spec)
	}
	if spec.Stream {
		spec.Model = chain[0]
		return s.startChatOne(ctx, spec)
	}
	for i, m := range chain {
		spec.Model = m
		ch, n, err := s.startChatOne(ctx, spec)
		var evs []genEvent
		if err == nil {
			for ev := range ch {
				evs = append(evs, ev)
			}
		}
		why := autoFailed(err, evs)
		if i < len(chain)-1 && why != "" {
			log.Printf("auto: %s failed (%s), trying %s", m, why, chain[i+1])
			continue
		}
		if err != nil {
			return nil, 0, err
		}
		out := make(chan genEvent, len(evs))
		for _, ev := range evs {
			out <- ev
		}
		close(out)
		return out, n, nil
	}
	return nil, 0, &apiError{Status: 500, Type: "server_error", Msg: "auto: no model"}
}
