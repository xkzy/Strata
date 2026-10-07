// pkg/engineipc/ipc_test.go - Unit Tests for Engine Orchestrator
package engineipc

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestByteTokenizerAndStopMatcher(t *testing.T) {
	tok := NewByteTokenizer()
	text := "<|im_start|>system\nYou are an AI assistant.<|im_end|>"
	ids := tok.Encode(text, true)
	decoded := tok.Decode(ids)
	if decoded != text {
		t.Fatalf("expected roundtrip decode %q, got %q", text, decoded)
	}

	// Test StopMatcher
	matcher := NewStopMatcher([]string{"<|im_end|>", "STOP"})
	out1 := matcher.Push("Hello world ")
	if out1 != "Hello world " {
		t.Fatalf("expected 'Hello world ', got %q", out1)
	}
	out2 := matcher.Push("<|im_")
	if out2 != "" {
		t.Fatalf("expected held partial stop, got %q", out2)
	}
	out3 := matcher.Push("end|> extra")
	if out3 != "" || matcher.Hit() != "<|im_end|>" {
		t.Fatalf("expected stop hit, got out3=%q hit=%q", out3, matcher.Hit())
	}
}

// ---- a fake engine: this test binary re-executed with STRATA_FAKE_ENGINE=1 speaks the engine's line protocol ----

func TestMain(m *testing.M) {
	if mode := os.Getenv("STRATA_FAKE_ENGINE"); mode != "" {
		fakeEngine(mode)
		return
	}
	os.Exit(m.Run())
}

func fakeEngine(mode string) {
	out := bufio.NewWriter(os.Stdout)
	flush := func() { out.Flush() }
	if mode == "crash" {
		fmt.Fprintln(os.Stderr, "fake engine: no model found")
		os.Exit(3)
	}
	fmt.Fprintln(out, "INFO context=4096 kv=q4_0")
	fmt.Fprintln(out, "READY 4096 stop")
	flush()
	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 1<<20), 1<<24)
	stop := make(chan struct{}, 1)
	lines := make(chan string, 16)
	go func() {
		for in.Scan() {
			l := in.Text()
			if l == "STOP" {
				select {
				case stop <- struct{}{}:
				default:
				}
				continue
			}
			lines <- l
		}
		close(lines)
	}()
	for l := range lines {
		if l == "QUIT" {
			return
		}
		if !strings.HasPrefix(l, "GEN ") {
			fmt.Fprintln(out, "ERR expected GEN")
			flush()
			continue
		}
		f := strings.Fields(l)
		ids := strings.Split(f[len(f)-1], ",")
		if mode == "err" {
			fmt.Fprintln(out, "ERR prompt (9 tokens) + max_new (9) exceeds the context (4)")
			flush()
			continue
		}
		select { // a STOP that arrived after the previous request had ended belongs to nobody
		case <-stop:
		default:
		}
		fmt.Fprintln(out, "RESUME 0")
		fmt.Fprintln(out, "PP 5 5 1.0 5000")
		reply := "got:" + strings.Join(ids, ",") + " keys:" + strings.Join(f[2:len(f)-1], ",")
		if mode == "slow" {
			reply = strings.Repeat("x", 400)
		}
		n := 0
		reason := "stop"
	gen:
		for _, b := range []byte(reply) {
			select {
			case <-stop:
				reason = "cancel"
				break gen
			default:
			}
			fmt.Fprintf(out, "T %d\n", int(b))
			flush()
			n++
			if mode == "slow" {
				time.Sleep(2 * time.Millisecond)
			}
		}
		if mode == "stopid" {
			fmt.Fprintln(out, "T 257") // <|im_end|> in the byte tokenizer
			flush()
			fmt.Fprintln(out, "T 65")
		}
		select { // a STOP that arrived after the last token
		case <-stop:
		default:
		}
		fmt.Fprintf(out, "DONE %d %d 1.0 10.0 %s 0 0 0\n", n, len(ids), reason)
		flush()
	}
}

func startFake(t *testing.T, mode string) *EngineOrchestrator {
	t.Helper()
	exe, _ := os.Executable()
	eo := NewEngineOrchestrator(EngineConfig{BinaryPath: exe, Args: []string{"-test.run=XXX"}, Env: []string{"STRATA_FAKE_ENGINE=" + mode}, StartTimeout: 20 * time.Second})
	if err := eo.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { eo.Stop() })
	return eo
}

func collect(out <-chan TokenEvent) (text string, end TokenEvent) {
	for ev := range out {
		if ev.IsEnd {
			end = ev
		} else {
			text += ev.Text
		}
	}
	return
}

func TestProtocolIdsAreCommaSeparatedAndSamplingKeysSent(t *testing.T) {
	eo := startFake(t, "echo")
	if err := eo.WaitReady(context.Background()); err != nil {
		t.Fatal(err)
	}
	if eo.MaxContext() != 4096 || eo.Info()["kv"] != "q4_0" {
		t.Fatalf("handshake not parsed: ctx=%d info=%v", eo.MaxContext(), eo.Info())
	}
	out := make(chan TokenEvent, 64)
	go eo.GenerateIDs(context.Background(), []int{72, 105, 300}, SamplingParams{Temperature: 0.5, TopP: 0.9, TopK: 40}, 50, nil, nil, out)
	text, end := collect(out)
	if end.Error != nil {
		t.Fatal(end.Error)
	}
	if !strings.HasPrefix(text, "got:72,105,300 keys:") || !strings.Contains(text, "temperature=0.5") || !strings.Contains(text, "top_p=0.9") || !strings.Contains(text, "top_k=40") {
		t.Fatalf("request line was not what the engine expects: %q", text)
	}
	if end.FinishReason != "stop" || end.PromptTokens != 3 || end.GenTokens == 0 {
		t.Fatalf("bad end event: %+v", end)
	}
}

func TestGreedyWhenNoTemperature(t *testing.T) {
	if k := samplingKeys(SamplingParams{}); k != "" {
		t.Fatalf("greedy request must send no keys, got %q", k)
	}
	if k := samplingKeys(SamplingParams{Temperature: 0.7, TopK: 0}); !strings.Contains(k, "top_k=64") {
		t.Fatalf("top_k 0 means all 64 candidates: %q", k)
	}
}

func TestEngineErrorIsReported(t *testing.T) {
	eo := startFake(t, "err")
	if err := eo.WaitReady(context.Background()); err != nil {
		t.Fatal(err)
	}
	out := make(chan TokenEvent, 8)
	go eo.GenerateIDs(context.Background(), []int{1}, SamplingParams{}, 5, nil, nil, out)
	_, end := collect(out)
	if end.Error == nil || !strings.Contains(end.Error.Error(), "exceeds the context") {
		t.Fatalf("expected the engine's ERR text, got %+v", end)
	}
	// the engine is still usable after a refused request
	out2 := make(chan TokenEvent, 8)
	go eo.GenerateIDs(context.Background(), []int{1}, SamplingParams{}, 5, nil, nil, out2)
	if _, end2 := collect(out2); end2.Error == nil {
		t.Fatal("second refusal expected too")
	}
}

func TestStopIDEndsTheReplyAndEngineIsDrained(t *testing.T) {
	eo := startFake(t, "stopid")
	if err := eo.WaitReady(context.Background()); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 2; i++ { // the second request proves the first left no stale lines behind
		out := make(chan TokenEvent, 128)
		go eo.GenerateIDs(context.Background(), []int{1, 2}, SamplingParams{}, 200, map[int]bool{257: true}, nil, out)
		text, end := collect(out)
		if end.Error != nil || end.FinishReason != "stop" || !strings.HasPrefix(text, "got:1,2") {
			t.Fatalf("round %d: %q %+v", i, text, end)
		}
		if strings.Contains(text, "A") && strings.HasSuffix(text, "A") {
			t.Fatalf("tokens after the stop id leaked: %q", text)
		}
	}
}

func TestCancelStopsEngineAndNextRequestWorks(t *testing.T) {
	eo := startFake(t, "slow")
	if err := eo.WaitReady(context.Background()); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	out := make(chan TokenEvent, 1024)
	go eo.GenerateIDs(ctx, []int{1}, SamplingParams{}, 1000, nil, nil, out)
	time.Sleep(60 * time.Millisecond)
	cancel()
	text, end := collect(out)
	if end.FinishReason != "cancel" || len(text) >= 400 {
		t.Fatalf("cancel did not stop the engine early: %d chars, %+v", len(text), end)
	}
	out2 := make(chan TokenEvent, 1024)
	go eo.GenerateIDs(context.Background(), []int{1}, SamplingParams{}, 1000, nil, nil, out2)
	text2, end2 := collect(out2)
	if end2.Error != nil || len(text2) != 400 {
		t.Fatalf("request after a cancel is broken: %d chars, %+v", len(text2), end2)
	}
}

func TestRequestsAreSerialized(t *testing.T) {
	eo := startFake(t, "slow")
	if err := eo.WaitReady(context.Background()); err != nil {
		t.Fatal(err)
	}
	var results [3]string
	done := make(chan int, 3)
	for i := 0; i < 3; i++ {
		go func(i int) {
			out := make(chan TokenEvent, 1024)
			go eo.GenerateIDs(context.Background(), []int{i}, SamplingParams{}, 1000, nil, nil, out)
			results[i], _ = collect(out)
			done <- i
		}(i)
	}
	for i := 0; i < 3; i++ {
		<-done
	}
	for i, r := range results {
		if len(r) != 400 || strings.Trim(r, "x") != "" {
			t.Fatalf("request %d got mixed-up output (%d chars)", i, len(r))
		}
	}
}

func TestEngineThatCrashesIsReportedNotFaked(t *testing.T) {
	eo := startFake(t, "crash")
	err := eo.WaitReady(context.Background())
	if err == nil {
		t.Fatal("a crashed engine must not report ready")
	}
	if !strings.Contains(err.Error(), "no model found") {
		t.Fatalf("the engine's own message should reach the caller: %v", err)
	}
	out := make(chan TokenEvent, 4)
	go eo.GenerateIDs(context.Background(), []int{1}, SamplingParams{}, 5, nil, nil, out)
	text, end := collect(out)
	if text != "" || end.Error == nil {
		t.Fatalf("no canned text when the engine is down: %q %+v", text, end)
	}
}

func TestMissingBinaryAndMissingModel(t *testing.T) {
	eo := NewEngineOrchestrator(EngineConfig{BinaryPath: "/nonexistent/strata", Args: []string{"--pack", "x"}})
	if err := eo.Start(context.Background()); err == nil || !strings.Contains(err.Error(), "not found") {
		t.Fatalf("expected a clear error, got %v", err)
	}
	exe, _ := os.Executable()
	eo2 := NewEngineOrchestrator(EngineConfig{BinaryPath: exe})
	if err := eo2.Start(context.Background()); err == nil || !strings.Contains(err.Error(), "no model") {
		t.Fatalf("expected a no-model error, got %v", err)
	}
	_ = strconv.Itoa
}
