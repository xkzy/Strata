package bench

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// NeedleTestOptions controls parameters for needle-in-a-haystack evaluation.
type NeedleTestOptions struct {
	BaseURL     string
	APIKey      string
	Lengths     []int
	Depths      []int
	NeedleWords []string
	Timeout     time.Duration
}

// NeedleTestResult records the result of a single needle evaluation.
type NeedleTestResult struct {
	TargetLength int           `json:"target_length"`
	ActualTokens int           `json:"actual_tokens"`
	DepthPercent int           `json:"depth_percent"`
	NeedleWord   string        `json:"needle_word"`
	Found        bool          `json:"found"`
	Duration     time.Duration `json:"duration_ms"`
	Response     string        `json:"response"`
}

// DefaultNeedleWords provides standard probe words.
var DefaultNeedleWords = []string{
	"amber", "falcon", "quartz", "willow", "copper", "harbor", "saffron", "glacier",
}

// GenerateHaystack produces synthetic filler text of approximate character length.
func GenerateHaystack(nChars int, needle string, depthPercent int) string {
	filler := "The quick brown fox jumps over the lazy dog in high performance computing environments. "
	repeats := (nChars / len(filler)) + 1
	var sb strings.Builder
	sb.Grow(nChars + 200)

	needleSentence := fmt.Sprintf("\n\nCRITICAL SPECIAL SECRET CODE: The password is %s.\n\n", needle)

	insertCharPos := int(float64(nChars) * float64(depthPercent) / 100.0)
	if insertCharPos < 0 {
		insertCharPos = 0
	}
	if insertCharPos > nChars {
		insertCharPos = nChars
	}

	written := 0
	inserted := false
	for i := 0; i < repeats && written < nChars; i++ {
		if !inserted && written >= insertCharPos {
			sb.WriteString(needleSentence)
			inserted = true
		}
		sb.WriteString(filler)
		written += len(filler)
	}

	if !inserted {
		sb.WriteString(needleSentence)
	}

	sb.WriteString("\nQuestion: What is the secret password code mentioned in the text? Answer with only the password word.")
	return sb.String()
}

// RunNeedleEvaluation executes needle-in-a-haystack recall test against an active Strata server.
func RunNeedleEvaluation(ctx context.Context, opts NeedleTestOptions) ([]NeedleTestResult, error) {
	if opts.BaseURL == "" {
		opts.BaseURL = "http://127.0.0.1:8080"
	}
	if len(opts.Lengths) == 0 {
		opts.Lengths = []int{4096, 8192}
	}
	if len(opts.Depths) == 0 {
		opts.Depths = []int{10, 50, 90}
	}
	if len(opts.NeedleWords) == 0 {
		opts.NeedleWords = DefaultNeedleWords
	}
	if opts.Timeout <= 0 {
		opts.Timeout = 120 * time.Second
	}

	client := &http.Client{Timeout: opts.Timeout}
	results := make([]NeedleTestResult, 0, len(opts.Lengths)*len(opts.Depths))

	wordIdx := 0
	for _, targetLen := range opts.Lengths {
		for _, depth := range opts.Depths {
			word := opts.NeedleWords[wordIdx%len(opts.NeedleWords)]
			wordIdx++

			prompt := GenerateHaystack(targetLen*3, word, depth)

			reqBody, _ := json.Marshal(map[string]any{
				"model": "default",
				"messages": []map[string]string{
					{"role": "user", "content": prompt},
				},
				"temperature": 0.0,
				"max_tokens":  64,
			})

			req, err := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(opts.BaseURL, "/")+"/v1/chat/completions", bytes.NewReader(reqBody))
			if err != nil {
				return nil, err
			}
			req.Header.Set("Content-Type", "application/json")
			if opts.APIKey != "" {
				req.Header.Set("Authorization", "Bearer "+opts.APIKey)
			}

			start := time.Now()
			resp, err := client.Do(req)
			duration := time.Since(start)

			res := NeedleTestResult{
				TargetLength: targetLen,
				DepthPercent: depth,
				NeedleWord:   word,
				Duration:     duration,
			}

			if err != nil {
				res.Found = false
				res.Response = fmt.Sprintf("request error: %v", err)
				results = append(results, res)
				continue
			}

			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()

			var chatResp struct {
				Choices []struct {
					Message struct {
						Content string `json:"content"`
					} `json:"message"`
				} `json:"choices"`
				Usage struct {
					PromptTokens int `json:"prompt_tokens"`
				} `json:"usage"`
			}
			_ = json.Unmarshal(body, &chatResp)

			if len(chatResp.Choices) > 0 {
				ans := strings.ToLower(chatResp.Choices[0].Message.Content)
				res.Response = chatResp.Choices[0].Message.Content
				res.ActualTokens = chatResp.Usage.PromptTokens
				res.Found = strings.Contains(ans, strings.ToLower(word))
			}

			results = append(results, res)
		}
	}

	return results, nil
}
