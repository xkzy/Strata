package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"strata/pkg/engineipc"
	"strata/pkg/tokenizer"
)

// supportedTemplateSHA256 is the hash of the chat_template.jinja that pkg/chattemplate reproduces (checked against Jinja
// in that package's tests). A pack with a different template (another fine-tune) is rendered with this one anyway, and
// the mismatch is reported in /status and the log instead of being silent.
const supportedTemplateSHA256 = "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce"

// engineSpec is a strata-<model>.json file: how to start the engine and where its tokenizer is.
type engineSpec struct {
	Exe       string            `json:"exe"`
	Args      []string          `json:"args"`
	Cwd       string            `json:"cwd"`
	Tokenizer string            `json:"tokenizer"`
	ModelName string            `json:"model_name"`
	Env       map[string]string `json:"env"`
}

func readEngineSpec(path string) (*engineSpec, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var sp engineSpec
	if err := json.Unmarshal(raw, &sp); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(sp.Args) == 0 {
		return nil, fmt.Errorf("%s has no \"args\" (it is not an engine config)", path)
	}
	return &sp, nil
}

// findEngineSpec picks the engine config: the --config file, else the single strata-*.json in the working directory
// that holds an engine command line.
func findEngineSpec(configFile string) (*engineSpec, string, error) {
	if configFile != "" {
		sp, err := readEngineSpec(configFile)
		return sp, configFile, err
	}
	matches, _ := filepath.Glob("strata-*.json")
	sort.Strings(matches)
	var found []string
	var spec *engineSpec
	for _, m := range matches {
		if sp, err := readEngineSpec(m); err == nil {
			found = append(found, m)
			spec = sp
		}
	}
	switch len(found) {
	case 0:
		return nil, "", nil
	case 1:
		return spec, found[0], nil
	}
	return nil, "", fmt.Errorf("several engine configs here (%s): choose one with --config", strings.Join(found, ", "))
}

func argValue(args []string, flag string) string {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag {
			return args[i+1]
		}
	}
	return ""
}

// engineState is what the server knows about its engine besides the process itself.
type engineState struct {
	mu          sync.RWMutex
	tok         *tokenizer.Tokenizer
	stopIDs     map[int]bool
	problem     string // why the engine cannot be started (no model configured, tokenizer missing ...)
	source      string // config file in use
	templateMsg string // set when the pack's chat template is not the supported one
	sampling    engineipc.SamplingParams
}

func (s *StrataServer) setProblem(msg string) {
	s.eng.mu.Lock()
	s.eng.problem = msg
	s.eng.mu.Unlock()
}

func (s *StrataServer) tokenizerOrNil() *tokenizer.Tokenizer {
	s.eng.mu.RLock()
	defer s.eng.mu.RUnlock()
	return s.eng.tok
}

// StartEngine reads the engine config, loads the tokenizer and starts the engine process in the background. It returns
// an error when the engine cannot be started; the server keeps running and every generation request then fails with
// that reason (HTTP 503) instead of being answered with invented text.
func (s *StrataServer) StartEngine(ctx context.Context) error {
	spec, source, err := findEngineSpec(s.Config.ConfigFile)
	if err != nil {
		s.setProblem(err.Error())
		return err
	}
	cfg := engineipc.EngineConfig{BinaryPath: s.Config.BinaryPath, ContextSize: s.Config.MaxContext}
	tokDir := os.Getenv("STRATA_TOKENIZER_DIR")
	switch {
	case spec != nil:
		s.eng.source = source
		if cfg.BinaryPath == "" {
			cfg.BinaryPath = spec.Exe
		}
		cfg.Args, cfg.Cwd = spec.Args, spec.Cwd
		for k, v := range spec.Env {
			cfg.Env = append(cfg.Env, k+"="+v)
		}
		if n, err := strconv.Atoi(argValue(spec.Args, "--max-context")); err == nil && n > 0 {
			cfg.ContextSize = n
		}
		if spec.Tokenizer != "" {
			tokDir = spec.Tokenizer
		}
		if tokDir == "" {
			if pack := argValue(spec.Args, "--pack"); pack != "" {
				tokDir = filepath.Join(pack, "tokenizer")
			}
		}
		if spec.ModelName != "" && s.Config.ModelName == "Qwen3.8-Flash-Next" {
			s.Config.ModelName = spec.ModelName
		}
	case s.Config.ModelPath != "":
		cfg.ModelPath = s.Config.ModelPath
		if tokDir == "" {
			tokDir = filepath.Join(s.Config.ModelPath, "tokenizer")
		}
	default:
		err := fmt.Errorf("no model is configured: start with --config strata-<model>.json (found none in %s) or --model-path", mustGetwd())
		s.setProblem(err.Error())
		return err
	}
	if tokDir == "" {
		err := fmt.Errorf("the model's tokenizer directory is unknown (set \"tokenizer\" in the engine config or STRATA_TOKENIZER_DIR)")
		s.setProblem(err.Error())
		return err
	}
	tok, err := tokenizer.Load(tokDir)
	if err != nil {
		err = fmt.Errorf("cannot load the model's tokenizer from %s: %w", tokDir, err)
		s.setProblem(err.Error())
		return err
	}
	stop := map[int]bool{}
	for _, lit := range []string{"<|im_end|>", "<|endoftext|>"} {
		if id, ok := tok.ID(lit); ok {
			stop[id] = true
		}
	}
	if tok.EOS >= 0 {
		stop[tok.EOS] = true
	}
	s.eng.mu.Lock()
	s.eng.tok, s.eng.stopIDs = tok, stop
	s.eng.mu.Unlock()
	if raw, err := os.ReadFile(filepath.Join(tokDir, "chat_template.jinja")); err == nil {
		sum := sha256.Sum256(raw)
		if hex.EncodeToString(sum[:]) != supportedTemplateSHA256 {
			s.eng.mu.Lock()
			s.eng.templateMsg = "the pack's chat template differs from the one this server renders (prompts may not match the model's training format)"
			s.eng.mu.Unlock()
			fmt.Fprintln(os.Stderr, "strata: warning:", s.eng.templateMsg)
		}
	}
	for _, p := range []string{s.Config.ConfigFile, "strata-config.json"} {
		if p == "" {
			continue
		}
		if d, ok := loadSamplingDefaults(p); ok {
			s.eng.mu.Lock()
			s.eng.sampling = d
			s.eng.mu.Unlock()
			break
		}
	}
	s.EngineIPC = engineipc.NewEngineOrchestrator(cfg)
	s.EngineIPC.SetTokenizer(tok)
	s.Config.MaxContext = cfg.ContextSize
	if err := s.EngineIPC.Start(ctx); err != nil {
		s.setProblem(err.Error())
		return err
	}
	return nil
}

func mustGetwd() string {
	d, _ := os.Getwd()
	return d
}

// engineUnavailable returns the HTTP status and message for a request that cannot be served right now ("" when it can).
func (s *StrataServer) engineUnavailable() (int, string) {
	s.eng.mu.RLock()
	problem, tok := s.eng.problem, s.eng.tok
	s.eng.mu.RUnlock()
	if problem != "" {
		return 503, "the model is not available: " + problem
	}
	if tok == nil || s.EngineIPC == nil {
		return 503, "the model is not available: the engine has not been started"
	}
	st, reason := s.EngineIPC.Status()
	switch st {
	case engineipc.StateReady:
		return 0, ""
	case engineipc.StateStarting:
		return 503, "the model is still loading; try again in a moment"
	}
	return 503, (&engineipc.ErrNotReady{State: st, Reason: reason}).Error()
}

// engineStatus is the engine's state for /health and /status.
func (s *StrataServer) engineStatus() map[string]interface{} {
	out := map[string]interface{}{"engine": "stopped", "loaded": false}
	s.eng.mu.RLock()
	problem, source, tmsg := s.eng.problem, s.eng.source, s.eng.templateMsg
	s.eng.mu.RUnlock()
	if source != "" {
		out["engine_config"] = source
	}
	if tmsg != "" {
		out["template_warning"] = tmsg
	}
	if problem != "" {
		out["engine"], out["error"] = "failed", problem
		return out
	}
	if s.EngineIPC != nil {
		st, reason := s.EngineIPC.Status()
		out["engine"] = st.String()
		out["loaded"] = st == engineipc.StateReady
		if reason != "" {
			out["error"] = reason
		} else if st == engineipc.StateStopped {
			out["error"] = "the engine has not been started"
		}
	}
	return out
}
