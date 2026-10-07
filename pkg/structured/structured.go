// pkg/structured/structured.go - Structured Output JSON Schema Validator in Go
package structured

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

type FormatType string

const (
	FormatText       FormatType = "text"
	FormatJSONObject FormatType = "json_object"
	FormatJSONSchema FormatType = "json_schema"
)

type JSONSchemaSpec struct {
	Name        string                 `json:"name,omitempty"`
	Description string                 `json:"description,omitempty"`
	Strict      bool                   `json:"strict,omitempty"`
	Schema      map[string]interface{} `json:"schema,omitempty"`
}

type ResponseFormat struct {
	Type       FormatType      `json:"type"`
	JSONSchema *JSONSchemaSpec `json:"json_schema,omitempty"`
}

type StructuredValidator struct{}

func NewStructuredValidator() *StructuredValidator {
	return &StructuredValidator{}
}

// BuildPromptInstruction appends schema requirements to system instructions
func (v *StructuredValidator) BuildPromptInstruction(rf *ResponseFormat) string {
	if rf == nil || rf.Type == FormatText {
		return ""
	}

	if rf.Type == FormatJSONObject {
		return "\nRespond with a valid JSON object only. Do not include markdown formatting or explanation outside the JSON."
	}

	if rf.Type == FormatJSONSchema && rf.JSONSchema != nil && rf.JSONSchema.Schema != nil {
		schemaBytes, err := json.Marshal(rf.JSONSchema.Schema)
		if err == nil {
			return fmt.Sprintf("\nRespond with a valid JSON object strictly conforming to this schema:\n%s", string(schemaBytes))
		}
	}

	return "\nRespond with valid JSON."
}

// ExtractAndValidate extracts a JSON object from text and validates it against the response format
func (v *StructuredValidator) ExtractAndValidate(text string, rf *ResponseFormat) (map[string]interface{}, error) {
	if rf == nil || rf.Type == FormatText {
		return nil, nil
	}

	trimmed := strings.TrimSpace(text)
	// Strip markdown code fences if model enclosed in ```json ... ```
	if strings.HasPrefix(trimmed, "```json") && strings.HasSuffix(trimmed, "```") {
		trimmed = strings.TrimPrefix(trimmed, "```json")
		trimmed = strings.TrimSuffix(trimmed, "```")
		trimmed = strings.TrimSpace(trimmed)
	} else if strings.HasPrefix(trimmed, "```") && strings.HasSuffix(trimmed, "```") {
		trimmed = strings.TrimPrefix(trimmed, "```")
		trimmed = strings.TrimSuffix(trimmed, "```")
		trimmed = strings.TrimSpace(trimmed)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(trimmed), &parsed); err != nil {
		return nil, fmt.Errorf("response is not valid JSON object: %w", err)
	}

	if rf.Type == FormatJSONObject {
		return parsed, nil
	}

	if rf.Type == FormatJSONSchema && rf.JSONSchema != nil && rf.JSONSchema.Schema != nil {
		if err := v.validateObject(parsed, rf.JSONSchema.Schema); err != nil {
			return nil, fmt.Errorf("schema validation failed: %w", err)
		}
	}

	return parsed, nil
}

func (v *StructuredValidator) validateObject(data map[string]interface{}, schema map[string]interface{}) error {
	// Check type
	if t, ok := schema["type"].(string); ok && t != "object" {
		return fmt.Errorf("expected root type '%s', got object", t)
	}

	// Check required fields
	if req, ok := schema["required"].([]interface{}); ok {
		for _, r := range req {
			key, isStr := r.(string)
			if !isStr {
				continue
			}
			if _, exists := data[key]; !exists {
				return fmt.Errorf("missing required property: '%s'", key)
			}
		}
	}

	// Check properties
	if props, ok := schema["properties"].(map[string]interface{}); ok {
		for k, val := range data {
			propSchemaRaw, exists := props[k]
			if !exists {
				if additionalProps, ok := schema["additionalProperties"].(bool); ok && !additionalProps {
					return fmt.Errorf("additional property '%s' not allowed by schema", k)
				}
				continue
			}

			propSchema, ok := propSchemaRaw.(map[string]interface{})
			if !ok {
				continue
			}

			if expType, ok := propSchema["type"].(string); ok {
				if err := checkType(val, expType); err != nil {
					return fmt.Errorf("property '%s': %w", k, err)
				}
			}

			// Recursive object validation
			if subObj, ok := val.(map[string]interface{}); ok {
				if err := v.validateObject(subObj, propSchema); err != nil {
					return fmt.Errorf("in '%s': %w", k, err)
				}
			}
		}
	}

	return nil
}

func checkType(val interface{}, expType string) error {
	if val == nil {
		return fmt.Errorf("expected %s, got null", expType)
	}
	switch expType {
	case "string":
		if _, ok := val.(string); !ok {
			return fmt.Errorf("expected string, got %s", reflect.TypeOf(val))
		}
	case "number", "integer":
		if _, ok := val.(float64); !ok {
			return fmt.Errorf("expected number, got %s", reflect.TypeOf(val))
		}
	case "boolean":
		if _, ok := val.(bool); !ok {
			return fmt.Errorf("expected boolean, got %s", reflect.TypeOf(val))
		}
	case "array":
		if _, ok := val.([]interface{}); !ok {
			return fmt.Errorf("expected array, got %s", reflect.TypeOf(val))
		}
	case "object":
		if _, ok := val.(map[string]interface{}); !ok {
			return fmt.Errorf("expected object, got %s", reflect.TypeOf(val))
		}
	}
	return nil
}
