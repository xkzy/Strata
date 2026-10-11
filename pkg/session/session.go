// pkg/session/session.go - Chat session models, serialization, and ID management.
package session

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"
)

var validIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_\-]+$`)

// Message represents a single turn in a chat conversation.
type Message struct {
	Role      string      `json:"role"`
	Text      string      `json:"text,omitempty"`
	Reasoning string      `json:"reasoning,omitempty"`
	Time      int64       `json:"time,omitempty"`
	Images    interface{} `json:"images,omitempty"`
	Files     interface{} `json:"files,omitempty"`
	Meta      string      `json:"meta,omitempty"`
	Stopped   bool        `json:"stopped,omitempty"`
	Error     string      `json:"error,omitempty"`
	Tools     interface{} `json:"tools,omitempty"`
}

// Session represents a complete conversation with history and settings.
type Session struct {
	ID        string    `json:"id"`
	Title     string    `json:"title"`
	CreatedAt int64     `json:"created_at"` // milliseconds Unix timestamp
	UpdatedAt int64     `json:"updated_at"` // milliseconds Unix timestamp
	Model     string    `json:"model,omitempty"`
	Messages  []Message `json:"messages"`
	Pinned    bool      `json:"pinned,omitempty"`
	Tags      []string  `json:"tags,omitempty"`
}

// SessionSummary provides lightweight metadata for listing and sidebars.
type SessionSummary struct {
	ID           string   `json:"id"`
	Title        string   `json:"title"`
	CreatedAt    int64    `json:"created_at"`
	UpdatedAt    int64    `json:"updated_at"`
	Model        string   `json:"model,omitempty"`
	MessageCount int      `json:"message_count"`
	Preview      string   `json:"preview"`
	Pinned       bool     `json:"pinned,omitempty"`
	Tags         []string `json:"tags,omitempty"`
}

// UnmarshalJSON supports both snake_case and camelCase timestamp fields.
func (s *Session) UnmarshalJSON(data []byte) error {
	type Alias Session
	aux := struct {
		CreatedAtSnake int64 `json:"created_at"`
		CreatedAtCamel int64 `json:"createdAt"`
		UpdatedAtSnake int64 `json:"updated_at"`
		UpdatedAtCamel int64 `json:"updatedAt"`
		*Alias
	}{
		Alias: (*Alias)(s),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	if aux.CreatedAtSnake != 0 {
		s.CreatedAt = aux.CreatedAtSnake
	} else if aux.CreatedAtCamel != 0 {
		s.CreatedAt = aux.CreatedAtCamel
	}
	if aux.UpdatedAtSnake != 0 {
		s.UpdatedAt = aux.UpdatedAtSnake
	} else if aux.UpdatedAtCamel != 0 {
		s.UpdatedAt = aux.UpdatedAtCamel
	}
	return nil
}

// MarshalJSON outputs both snake_case and camelCase timestamps for frontend compatibility.
func (s *Session) MarshalJSON() ([]byte, error) {
	type Alias Session
	return json.Marshal(&struct {
		CreatedAtCamel int64 `json:"createdAt"`
		UpdatedAtCamel int64 `json:"updatedAt"`
		*Alias
	}{
		CreatedAtCamel: s.CreatedAt,
		UpdatedAtCamel: s.UpdatedAt,
		Alias:          (*Alias)(s),
	})
}

// MarshalJSON outputs both snake_case and camelCase timestamps for frontend compatibility.
func (sm SessionSummary) MarshalJSON() ([]byte, error) {
	type Alias SessionSummary
	return json.Marshal(&struct {
		CreatedAtCamel int64 `json:"createdAt"`
		UpdatedAtCamel int64 `json:"updatedAt"`
		Alias
	}{
		CreatedAtCamel: sm.CreatedAt,
		UpdatedAtCamel: sm.UpdatedAt,
		Alias:          (Alias)(sm),
	})
}

// Summary returns the metadata summary of a session.
func (s *Session) Summary() SessionSummary {
	preview := ""
	for _, m := range s.Messages {
		txt := strings.TrimSpace(m.Text)
		if m.Role == "user" && txt != "" {
			preview = txt
			break
		}
	}
	if preview == "" {
		for _, m := range s.Messages {
			txt := strings.TrimSpace(m.Text)
			if txt != "" {
				preview = txt
				break
			}
		}
	}
	if len(preview) > 120 {
		preview = preview[:117] + "..."
	}

	return SessionSummary{
		ID:           s.ID,
		Title:        s.Title,
		CreatedAt:    s.CreatedAt,
		UpdatedAt:    s.UpdatedAt,
		Model:        s.Model,
		MessageCount: len(s.Messages),
		Preview:      preview,
		Pinned:       s.Pinned,
		Tags:         s.Tags,
	}
}

// GenerateID produces a cryptographically random session ID string ("sess-" + 16 random hex bytes).
func GenerateID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("sess-%d", time.Now().UnixNano())
	}
	return "sess-" + hex.EncodeToString(b[:])
}

// SanitizeID verifies and cleanses a session ID, strictly rejecting any path traversal or invalid characters.
func SanitizeID(raw string) (string, error) {
	clean := strings.TrimSpace(raw)
	if clean == "" {
		return "", fmt.Errorf("empty session id")
	}
	if strings.Contains(clean, "..") || strings.Contains(clean, "/") || strings.Contains(clean, "\\") || strings.Contains(clean, "\x00") {
		return "", fmt.Errorf("session id cannot contain path elements or separators: %q", raw)
	}
	if !validIDPattern.MatchString(clean) {
		return "", fmt.Errorf("invalid session id format: %q", raw)
	}
	return clean, nil
}
