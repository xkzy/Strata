// pkg/server/inference_interception_test.go - Comprehensive End-to-End Tests for Inference Math Interception & Injection
package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"

	"strata/pkg/mathruntime"
)

func TestInferenceMathInterceptionAndInjection(t *testing.T) {
	ts, promptFile := startChatServer(t, "Computing the product.\n</think>\n\n123456789 * 987654321 = 121932631112635269.")

	// 1. User sends plain inference request without any math tool call
	userQuery := `{"messages":[{"role":"user","content":"What is 123456789 * 987654321?"}]}`
	resp, body := postJSON(t, ts.URL+"/v1/chat/completions", userQuery)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", resp.StatusCode, string(body))
	}

	var chatResp map[string]interface{}
	json.Unmarshal(body, &chatResp)

	choices, ok := chatResp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Fatalf("expected at least 1 choice in response")
	}

	// 2. Verify that the internal prompt contains the injected verified fact
	rawPrompt, err := os.ReadFile(promptFile)
	if err != nil {
		t.Fatalf("failed reading prompt file: %v", err)
	}
	promptStr := string(rawPrompt)
	if !strings.Contains(promptStr, "121932631112635269") {
		t.Fatalf("expected prompt to carry injected verified math result, got:\n%s", promptStr)
	}
	if !strings.Contains(promptStr, "Verified fact for this answer") {
		t.Fatalf("expected prompt to carry 'Verified fact for this answer', got:\n%s", promptStr)
	}
}

func TestInferenceTheoremsAndSageInterception(t *testing.T) {
	srv := setupTestServer()

	// 1. Sage Number Theory: euler_phi(36)
	resPhi := srv.MathRuntime.InterceptAndVerifyIntent("Calculate euler_phi(36)")
	if resPhi == nil || resPhi.ExactResult != "12" {
		t.Fatalf("expected euler_phi(36) = 12, got %+v", resPhi)
	}

	// 2. Sage Primality: is_prime(104729)
	resPrime := srv.MathRuntime.InterceptAndVerifyIntent("Verify is_prime(104729)")
	if resPrime == nil || resPrime.ExactResult != "True" {
		t.Fatalf("expected is_prime(104729) = True, got %+v", resPrime)
	}

	// 3. Matrix Determinant
	resDet := srv.MathRuntime.InterceptAndVerifyIntent("Compute det([[1, 2], [3, 4]])")
	if resDet == nil || resDet.ExactResult != "-2" {
		t.Fatalf("expected det([[1,2],[3,4]]) = -2, got %+v", resDet)
	}

	// 4. Exact Rational Fraction: 1/3 + 1/3 + 1/3
	resFrac := srv.MathRuntime.InterceptAndVerifyIntent("What is 1/3 + 1/3 + 1/3?")
	if resFrac == nil || resFrac.ExactResult != "1" {
		t.Fatalf("expected 1/3 + 1/3 + 1/3 = 1, got %+v", resFrac)
	}
}

func TestMathematicalSessionIsolation(t *testing.T) {
	mr := mathruntime.NewMathRuntime(nil)

	// Session A sets x = 10
	resA := mr.ProcessRequest(mathruntime.MathRequest{
		Operation:  mathruntime.OpEvaluate,
		Expression: "x = 10",
		SessionID:  "session-alpha",
	})
	if resA.Status != mathruntime.StatusSuccess {
		t.Fatalf("failed session A assignment: %v", resA.ErrorMessage)
	}

	// Session B sets x = 20
	resB := mr.ProcessRequest(mathruntime.MathRequest{
		Operation:  mathruntime.OpEvaluate,
		Expression: "x = 20",
		SessionID:  "session-beta",
	})
	if resB.Status != mathruntime.StatusSuccess {
		t.Fatalf("failed session B assignment: %v", resB.ErrorMessage)
	}

	// Verify isolated evaluations
	evalA := mr.ProcessRequest(mathruntime.MathRequest{
		Operation:  mathruntime.OpEvaluate,
		Expression: "x",
		SessionID:  "session-alpha",
	})
	evalB := mr.ProcessRequest(mathruntime.MathRequest{
		Operation:  mathruntime.OpEvaluate,
		Expression: "x",
		SessionID:  "session-beta",
	})

	if evalA.ExactResult == evalB.ExactResult {
		t.Fatalf("session variables leaked: alpha=%s beta=%s", evalA.ExactResult, evalB.ExactResult)
	}
}

func TestConcurrentInferenceMathInterception(t *testing.T) {
	srv := setupTestServer()
	const numGoroutines = 50
	var wg sync.WaitGroup
	errCh := make(chan error, numGoroutines)

	expressions := []struct {
		query    string
		expected string
	}{
		{"What is 37 * 19?", "703"},
		{"Calculate 1/3 + 1/6", "1/2"},
		{"What is 2^10?", "1024"},
		{"Compute euler_phi(36)", "12"},
		{"Verify is_prime(97)", "True"},
		{"Calculate 1000 * 1000", "1000000"},
		{"What is 15 + 25 + 35?", "75"},
		{"Calculate det([[2, 0], [0, 5]])", "10"},
	}

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			item := expressions[idx%len(expressions)]
			res := srv.MathRuntime.InterceptAndVerifyIntent(item.query)
			if res == nil || res.Status != mathruntime.StatusSuccess {
				errCh <- fmt.Errorf("[%d] query %q failed interception", idx, item.query)
				return
			}
			val := res.ExactResult
			if val == "" {
				val = res.NumericResult
			}
			if val != item.expected {
				errCh <- fmt.Errorf("[%d] query %q expected %s, got %s", idx, item.query, item.expected, val)
				return
			}
		}(i)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("concurrency error: %v", err)
	}
}
