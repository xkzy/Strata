package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strata/pkg/gguf"
	"strconv"
	"strings"
	"sync"

	"strata/pkg/autotune"
	"strata/pkg/engineipc"
	"strata/pkg/runconfig"
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
	if fi, err := os.Stat(path); err == nil && fi.IsDir() {
		return nil, fmt.Errorf("%s is a directory, not an engine config file", path)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var sp engineSpec
	if err := json.Unmarshal(bytes.TrimPrefix(raw, []byte("\xef\xbb\xbf")), &sp); err != nil { // Notepad adds a BOM
		return nil, fmt.Errorf("%s: %w", path, jsonErrorPlace(raw, err))
	}
	sp.Args = splitJoinedArgs(sp.Args)
	if len(sp.Args) == 0 {
		return nil, fmt.Errorf("%s has no \"args\" (it is not an engine config)", path)
	}
	return &sp, nil
}

// jsonErrorPlace says where in the file a JSON mistake is. Editing "args" by hand is where they happen (a missing comma
// between two entries), so the hint names that.
func jsonErrorPlace(raw []byte, err error) error {
	var se *json.SyntaxError
	if !errors.As(err, &se) {
		return err
	}
	off := int(se.Offset)
	if off > len(raw) {
		off = len(raw)
	}
	line := 1 + bytes.Count(raw[:off], []byte("\n"))
	col := off - (bytes.LastIndexByte(raw[:off], '\n') + 1)
	return fmt.Errorf("not valid JSON: %v at line %d, column %d (check the commas between the entries of \"args\": each option and each value is its own quoted entry, e.g. \"--ple-io\", \"mmap\")", se, line, col)
}

var joinedFlag = regexp.MustCompile(`^(--[A-Za-z0-9][A-Za-z0-9-]*)(?:\s+|=)(\S.*)$`)
var bareFlag = regexp.MustCompile(`^--[A-Za-z0-9][A-Za-z0-9-]*$`)

// splitJoinedArgs: an engine option written as ONE entry of "args" ("--ple-io mmap" or "--ple-io=mmap") would reach the
// engine as a single argument and stop it ("unknown argument"). Such an entry is split into the flag and its value, and
// never one that sits in the value place of the flag before it (a value may itself contain a space).
func splitJoinedArgs(args []string) []string {
	out := make([]string, 0, len(args))
	for _, a := range args {
		m := joinedFlag.FindStringSubmatch(a)
		if m == nil || (len(out) > 0 && bareFlag.MatchString(out[len(out)-1])) {
			out = append(out, a)
			continue
		}
		fmt.Fprintf(os.Stderr, "strata: the config's args hold %q as one entry: split into %q and %q (write them as two entries)\n", a, m[1], strings.TrimSpace(m[2]))
		out = append(out, m[1], strings.TrimSpace(m[2]))
	}
	return out
}

// resolveEngineConfigFile turns a file path or pack/model directory into a valid engineSpec and config file path.
func resolveEngineConfigFile(path string) (string, *engineSpec, error) {
	if path == "" {
		return "", nil, nil
	}
	fi, err := os.Stat(path)
	if err == nil && fi.IsDir() {
		baseName := filepath.Base(path)
		cands := []string{
			fmt.Sprintf("strata-%s.json", strings.ToLower(baseName)),
			fmt.Sprintf("strata-%s.json", strings.ToLower(strings.ReplaceAll(baseName, "_", "-"))),
			fmt.Sprintf("strata-%s.json", strings.ToLower(strings.ReplaceAll(baseName, "-", "_"))),
		}
		for _, cand := range cands {
			if sp, err := readEngineSpec(cand); err == nil {
				return cand, sp, nil
			}
		}
		if matches, err := filepath.Glob("strata-*.json"); err == nil {
			for _, m := range matches {
				if sp, err := readEngineSpec(m); err == nil {
					if pack := argValue(sp.Args, "--pack"); pack == path || strings.EqualFold(filepath.Base(pack), baseName) {
						return m, sp, nil
					}
				}
			}
		}
		return "", nil, fmt.Errorf("%s is a directory, not an engine config file", path)
	}
	sp, err := readEngineSpec(path)
	if err != nil {
		return "", nil, err
	}
	return path, sp, nil
}

// findEngineSpec picks the engine config: the --config file, else the last session config or newest strata-*.json.
func findEngineSpec(configFile string, lastConfigFile string) (*engineSpec, string, error) {
	if configFile != "" {
		if realPath, sp, err := resolveEngineConfigFile(configFile); err == nil {
			return sp, realPath, nil
		} else if !os.IsNotExist(err) && !strings.Contains(err.Error(), "is a directory") {
			return nil, configFile, err
		}
	}
	if lastConfigFile != "" {
		if realPath, sp, err := resolveEngineConfigFile(lastConfigFile); err == nil {
			return sp, realPath, nil
		}
	}
	matches, _ := filepath.Glob("strata-*.json")
	var found []string
	specs := make(map[string]*engineSpec)
	for _, m := range matches {
		if sp, err := readEngineSpec(m); err == nil {
			found = append(found, m)
			specs[m] = sp
		}
	}
	switch len(found) {
	case 0:
		return nil, "", nil
	case 1:
		return specs[found[0]], found[0], nil
	}
	// Sort by modification time descending so newest config is chosen
	sort.Slice(found, func(i, j int) bool {
		fi, erri := os.Stat(found[i])
		fj, errj := os.Stat(found[j])
		if erri == nil && errj == nil {
			return fi.ModTime().After(fj.ModTime())
		}
		return found[i] < found[j]
	})
	return specs[found[0]], found[0], nil
}

func argValue(args []string, flag string) string {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag {
			return args[i+1]
		}
	}
	return ""
}

func resolveTokenizerDir(tokDir string, modelPath string) string {
	if tokDir != "" {
		if _, err := os.Stat(filepath.Join(tokDir, "vocab.json")); err == nil {
			return tokDir
		}
	}
	candidates := []string{
		tokDir,
		filepath.Join(modelPath, "tokenizer"),
		modelPath,
	}
	if modelPath != "" {
		base := filepath.Base(modelPath)
		dir := filepath.Dir(modelPath)
		parent := filepath.Dir(dir)
		candidates = append(candidates,
			filepath.Join(parent, "packs", strings.ToLower(base), "tokenizer"),
			filepath.Join(parent, "packs", strings.ToLower(strings.ReplaceAll(base, "_", "-")), "tokenizer"),
			filepath.Join(parent, "packs", strings.ToLower(strings.ReplaceAll(base, "-", "_")), "tokenizer"),
			filepath.Join("packs", strings.ToLower(base), "tokenizer"),
		)
	}
	packGlobs := []string{
		"packs/*/tokenizer",
		"../packs/*/tokenizer",
		"/home/khing/Downloads/Strata-data/packs/*/tokenizer",
		"/home/khing/Downloads/Strata-data/packs/coder-iq1_m/tokenizer",
	}
	for _, pg := range packGlobs {
		if matches, err := filepath.Glob(pg); err == nil {
			candidates = append(candidates, matches...)
		}
	}

	for _, c := range candidates {
		if c == "" {
			continue
		}
		if _, err := os.Stat(filepath.Join(c, "vocab.json")); err == nil {
			return c
		}
	}
	return tokDir
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
	samplingSet bool // the config has a "sampling" block; without one requests get the model card's (cardSampling)
	templateSHA string
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
	lastConfig := ""
	if s.ConfigManager != nil {
		lastConfig = s.ConfigManager.Get().LastConfigFile
	}
	spec, source, err := findEngineSpec(s.Config.ConfigFile, lastConfig)
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
		// Ensure --abliterate is accompanied by a measured refusal direction vector to avoid engine exit status 2
		hasAbliterate := false
		hasCvec := false
		for _, a := range cfg.Args {
			if a == "--abliterate" {
				hasAbliterate = true
			}
			if a == "--control-vector-scaled" || a == "--abliteration-vector" {
				hasCvec = true
			}
		}
		if hasAbliterate && !hasCvec {
			packDir := argValue(spec.Args, "--pack")
			vecPath := runconfig.FindOrFetchRefusalVector(packDir, s.Config.ModelPath)
			if vecPath != "" {
				cfg.Args = append(cfg.Args, "--control-vector-scaled", vecPath+":1.0", "--cvec-mode", "project")
			} else {
				var safeArgs []string
				for _, a := range cfg.Args {
					if a != "--abliterate" {
						safeArgs = append(safeArgs, a)
					}
				}
				cfg.Args = safeArgs
			}
		}
		for k, v := range spec.Env {
			cfg.Env = append(cfg.Env, k+"="+v)
		}
		if n, err := strconv.Atoi(argValue(spec.Args, "--max-context")); err == nil && n > 0 {
			cfg.ContextSize = n
		}
		if argValue(spec.Args, "--max-context") == "" { // none configured: the model file says how much it supports
			if n := ggufContext(argValue(spec.Args, "--native")); n > 0 {
				cfg.ContextSize = n
				cfg.Args = append(cfg.Args, "--max-context", strconv.Itoa(n))
			}
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
	tokDir = resolveTokenizerDir(tokDir, s.Config.ModelPath)
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
		s.eng.mu.Lock()
		s.eng.templateSHA = hex.EncodeToString(sum[:])[:12]
		s.eng.mu.Unlock()
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
			s.eng.sampling, s.eng.samplingSet = d, true
			s.eng.mu.Unlock()
			break
		}
	}
	var refineAfterStart func()
	if s.Config.AutoTune {
		recordPath := s.Config.TuningRecord
		if recordPath == "" {
			recordPath = autotune.DefaultRecordPath()
		}
		geom, maxCtx, gerr := autotune.GeometryFromArgs(cfg.Args)
		if gerr == nil {
			geom = geom.ApplyKVFlag(cfg.Args)
		}
		if maxCtx <= 0 {
			maxCtx = cfg.ContextSize
		}
		// Measure is false: a usable record supplies the hints, and only a first run or an invalidated record
		// pays for the host calibration. Measured knobs replace the config's only where evidence exists.
		sr := autotune.Startup(ctx, autotune.StartupOptions{
			Args: cfg.Args, Cwd: cfg.Cwd, RecordPath: recordPath, MaxContext: maxCtx, ModelName: s.Config.ModelName,
			FirstRunCalibration: true,
		})
		if len(sr.AppliedArgs) > 0 {
			cfg.Args = sr.AppliedArgs
		}
		s.setTuning(sr, false)
		s.setRuntimeTuneInfo(recordPath, cfg.Args, geom, maxCtx)
		// The refinement waits for READY, so it is started only after the orchestrator exists and the engine
		// has been started; starting it here would race with the assignment below.
		refineAfterStart = func() {
			go s.refineTuning(s.lifecycleCtx, recordPath, append([]string(nil), cfg.Args...), geom, maxCtx)
		}
	}
	s.EngineIPC = engineipc.NewEngineOrchestrator(cfg)
	s.EngineIPC.SetTokenizer(tok)
	s.Config.MaxContext = cfg.ContextSize
	if err := s.EngineIPC.Start(ctx); err != nil {
		s.setProblem(err.Error())
		return err
	}
	if refineAfterStart != nil {
		refineAfterStart()
	}
	s.setupRuntime(tok.VocabSize())
	if s.ConfigManager != nil && source != "" {
		cfg := s.ConfigManager.Get()
		cfg.LastConfigFile = source
		if s.Config.ModelName != "" {
			cfg.LastModel = s.Config.ModelName
		}
		s.ConfigManager.Update(cfg)
		_ = s.ConfigManager.SaveToFile("")
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
	out := map[string]interface{}{"engine": "stopped", "loaded": false, "runtime": s.runtimeStatus()}
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

// ggufContext is the context length the model file declares (0 when it cannot be read).
func ggufContext(path string) int {
	if path == "" {
		return 0
	}
	g, err := gguf.Open(path)
	if err != nil {
		return 0
	}
	defer g.Close()
	arch, _ := g.Metadata["general.architecture"].(string)
	n, _ := strconv.Atoi(fmt.Sprint(g.Metadata[arch+".context_length"]))
	return n
}
