package multitenant

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"sync"
)

type SecurityScope struct {
	TenantID    string `json:"tenant_id"`
	UserID      string `json:"user_id"`
	WorkspaceID string `json:"workspace_id"`
	AgentID     string `json:"agent_id"`
	SessionID   string `json:"session_id"`
	// Credential tells callers that present different API keys apart, so a session id they happen to share does not
	// join their data. It is a short hash of the key, never the key, and it is not serialized.
	Credential string `json:"-"`
}

// credentialTag is a short hash of the credential a request presents ("" when it presents none).
func credentialTag(h http.Header) string {
	cred := h.Get("Authorization")
	if cred == "" {
		cred = h.Get("X-Api-Key")
	}
	if cred == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(cred))
	return hex.EncodeToString(sum[:6])
}

func DefaultSecurityScope() SecurityScope {
	return SecurityScope{
		TenantID:    "default_tenant",
		UserID:      "default_user",
		WorkspaceID: "default_workspace",
		AgentID:     "default_agent",
		SessionID:   "default_session",
	}
}

func FromHeaders(headers http.Header, bodyScope *SecurityScope) SecurityScope {
	scope := DefaultSecurityScope()
	if bodyScope != nil {
		if bodyScope.TenantID != "" {
			scope.TenantID = bodyScope.TenantID
		}
		if bodyScope.UserID != "" {
			scope.UserID = bodyScope.UserID
		}
		if bodyScope.WorkspaceID != "" {
			scope.WorkspaceID = bodyScope.WorkspaceID
		}
		if bodyScope.AgentID != "" {
			scope.AgentID = bodyScope.AgentID
		}
		if bodyScope.SessionID != "" {
			scope.SessionID = bodyScope.SessionID
		}
	}

	if t := headers.Get("X-Tenant-ID"); t != "" {
		scope.TenantID = t
	}
	if u := headers.Get("X-User-ID"); u != "" {
		scope.UserID = u
	}
	if s := headers.Get("X-Session-ID"); s != "" {
		scope.SessionID = s
	}
	scope.Credential = credentialTag(headers)
	return scope
}

type MultiTenantManager struct {
	mu       sync.RWMutex
	sessions map[string]SecurityScope
	metrics  map[string]map[string]interface{}
}

func NewMultiTenantManager() *MultiTenantManager {
	return &MultiTenantManager{
		sessions: make(map[string]SecurityScope),
		metrics:  make(map[string]map[string]interface{}),
	}
}

func (m *MultiTenantManager) RegisterSession(scope SecurityScope) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sessions[scope.SessionID] = scope
}

func (m *MultiTenantManager) DeleteSession(sessionID string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.sessions[sessionID]; ok {
		delete(m.sessions, sessionID)
		return true
	}
	return false
}

func (m *MultiTenantManager) ListSessions(tenantID string) []SecurityScope {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var list []SecurityScope
	for _, s := range m.sessions {
		if tenantID == "" || s.TenantID == tenantID {
			list = append(list, s)
		}
	}
	return list
}

func (m *MultiTenantManager) GetTenantMetrics(tenantID string) map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if met, ok := m.metrics[tenantID]; ok {
		return met
	}
	return map[string]interface{}{
		"tenant_id":       tenantID,
		"active_sessions": len(m.sessions),
		"quota_remaining": 1000000,
	}
}
