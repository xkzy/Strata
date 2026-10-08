// pkg/structured/structured_test.go - Unit Tests for Structured Output Validator
package structured

import (
	"testing"
)

func TestStructuredValidator_JSONObject(t *testing.T) {
	v := NewStructuredValidator()
	rf := &ResponseFormat{Type: FormatJSONObject}

	// 1. Valid JSON object
	text := `{"name": "test", "score": 95.5}`
	data, err := v.ExtractAndValidate(text, rf)
	if err != nil {
		t.Fatalf("unexpected error for valid JSON: %v", err)
	}
	if data["name"] != "test" {
		t.Fatalf("expected name=test, got %v", data["name"])
	}

	// 2. Wrapped in Markdown fences
	fenced := "```json\n{\"status\": \"ok\", \"count\": 42}\n```"
	data2, err2 := v.ExtractAndValidate(fenced, rf)
	if err2 != nil {
		t.Fatalf("unexpected error for fenced JSON: %v", err2)
	}
	if data2["status"] != "ok" {
		t.Fatalf("expected status=ok, got %v", data2["status"])
	}

	// 3. Invalid JSON
	invalid := "Not JSON at all"
	_, err3 := v.ExtractAndValidate(invalid, rf)
	if err3 == nil {
		t.Fatalf("expected error for invalid JSON text")
	}
}

func TestStructuredValidator_JSONSchema(t *testing.T) {
	v := NewStructuredValidator()
	rf := &ResponseFormat{
		Type: FormatJSONSchema,
		JSONSchema: &JSONSchemaSpec{
			Name: "user_profile",
			Schema: map[string]interface{}{
				"type":     "object",
				"required": []interface{}{"user_id", "email"},
				"properties": map[string]interface{}{
					"user_id": map[string]interface{}{"type": "string"},
					"email":   map[string]interface{}{"type": "string"},
					"age":     map[string]interface{}{"type": "number"},
				},
			},
		},
	}

	// 1. Valid compliant JSON
	valid := `{"user_id": "usr_123", "email": "user@example.com", "age": 28}`
	data, err := v.ExtractAndValidate(valid, rf)
	if err != nil {
		t.Fatalf("expected schema pass, got error: %v", err)
	}
	if data["user_id"] != "usr_123" {
		t.Fatalf("unexpected data: %v", data)
	}

	// 2. Missing required property
	missing := `{"email": "user@example.com"}`
	_, errMissing := v.ExtractAndValidate(missing, rf)
	if errMissing == nil {
		t.Fatalf("expected failure on missing user_id")
	}

	// 3. Type mismatch
	mismatch := `{"user_id": "usr_123", "email": "user@example.com", "age": "twenty-eight"}`
	_, errMismatch := v.ExtractAndValidate(mismatch, rf)
	if errMismatch == nil {
		t.Fatalf("expected failure on age type mismatch")
	}
}
