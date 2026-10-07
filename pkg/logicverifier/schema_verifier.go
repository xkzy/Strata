// pkg/logicverifier/schema_verifier.go - JSON Schema & Structure Verifier
package logicverifier

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
)

type SchemaVerifier struct{}

func NewSchemaVerifier() *SchemaVerifier {
	return &SchemaVerifier{}
}

// SimpleSchema represents expected field types and required properties
type SimpleSchema struct {
	Type       string                 `json:"type"`
	Required   []string               `json:"required"`
	Properties map[string]FieldSchema `json:"properties"`
}

type FieldSchema struct {
	Type string `json:"type"` // "string", "number", "boolean", "array", "object"
}

func (sv *SchemaVerifier) Verify(claim VerificationClaim) VerificationResult {
	res := VerificationResult{
		ClaimID:     claim.ClaimID,
		Type:        ClaimSchemaType,
		BackendUsed: "deterministic_schema_validator",
		Status:      StatusUnknown,
	}

	targetJSON := claim.ClaimedValue
	if targetJSON == "" {
		targetJSON = claim.CodeSnippet
	}
	if targetJSON == "" {
		targetJSON = claim.Expression
	}
	targetJSON = strings.TrimSpace(targetJSON)

	// 1. Syntax Validation
	var parsed interface{}
	if err := json.Unmarshal([]byte(targetJSON), &parsed); err != nil {
		res.Status = StatusFail
		res.FailureReason = fmt.Sprintf("invalid JSON syntax: %v", err)
		res.CompactObservation = fmt.Sprintf("Verification: FAIL | Invalid JSON: %v", err)
		return res
	}

	// If no schema provided, syntax check passed
	if claim.SchemaJSON == "" {
		res.Status = StatusPass
		res.Evidence = "valid JSON syntax confirmed"
		res.CompactObservation = "Verification: PASS | Valid JSON"
		return res
	}

	// 2. Schema Validation
	var schema SimpleSchema
	if err := json.Unmarshal([]byte(claim.SchemaJSON), &schema); err != nil {
		res.Status = StatusUnknown
		res.Evidence = fmt.Sprintf("failed to parse schema specification: %v", err)
		return res
	}

	objMap, ok := parsed.(map[string]interface{})
	if !ok && schema.Type == "object" {
		res.Status = StatusFail
		res.FailureReason = fmt.Sprintf("expected JSON object, got %s", reflect.TypeOf(parsed))
		res.CompactObservation = "Verification: FAIL | Root is not a JSON object"
		return res
	}

	// Check required fields
	for _, req := range schema.Required {
		val, exists := objMap[req]
		if !exists {
			res.Status = StatusFail
			res.FailureReason = fmt.Sprintf("missing required property: '%s'", req)
			res.CompactObservation = fmt.Sprintf("Verification: FAIL | Missing required property '%s'", req)
			return res
		}

		// Check type if specified in properties
		if propSchema, propExists := schema.Properties[req]; propExists {
			if !checkType(val, propSchema.Type) {
				res.Status = StatusFail
				res.FailureReason = fmt.Sprintf("property '%s' expected type '%s', got %s", req, propSchema.Type, reflect.TypeOf(val))
				res.CompactObservation = fmt.Sprintf("Verification: FAIL | Type mismatch on '%s'", req)
				return res
			}
		}
	}

	// Check remaining properties
	for propName, propSchema := range schema.Properties {
		val, exists := objMap[propName]
		if exists && !checkType(val, propSchema.Type) {
			res.Status = StatusFail
			res.FailureReason = fmt.Sprintf("property '%s' expected type '%s', got %s", propName, propSchema.Type, reflect.TypeOf(val))
			res.CompactObservation = fmt.Sprintf("Verification: FAIL | Type mismatch on '%s'", propName)
			return res
		}
	}

	res.Status = StatusPass
	res.Evidence = "JSON conforms to schema invariants and required properties"
	res.CompactObservation = "Verification: PASS | Schema compliant"
	return res
}

func checkType(val interface{}, expectedType string) bool {
	if val == nil {
		return false
	}
	switch expectedType {
	case "string":
		_, ok := val.(string)
		return ok
	case "number", "integer", "float":
		_, ok := val.(float64)
		return ok
	case "boolean", "bool":
		_, ok := val.(bool)
		return ok
	case "array", "list":
		_, ok := val.([]interface{})
		return ok
	case "object", "map":
		_, ok := val.(map[string]interface{})
		return ok
	default:
		return true
	}
}
