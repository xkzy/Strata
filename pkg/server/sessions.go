// pkg/server/sessions.go - REST API endpoints for persistent Web Chat session management.
package server

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"strings"

	"strata/pkg/multitenant"
	"strata/pkg/session"
)

// handleSessions handles /api/sessions and /v1/strata/sessions:
// GET: lists all sessions (or searches with ?q=...)
// POST: creates or updates a session
// DELETE: clears all sessions
func (s *StrataServer) handleSessions(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		q := strings.TrimSpace(r.URL.Query().Get("q"))
		var list []session.SessionSummary
		var err error
		if q != "" {
			list, err = s.SessionStore.Search(q)
		} else {
			list, err = s.SessionStore.List()
		}
		if err != nil {
			apiMessage(w, http.StatusInternalServerError, "server_error", "failed to list sessions: "+err.Error())
			return
		}
		if list == nil {
			list = []session.SessionSummary{}
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"sessions": list,
			"total":    len(list),
		})

	case http.MethodPost:
		bodyBytes, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 32<<20))
		if err != nil {
			apiMessage(w, http.StatusBadRequest, "invalid_request_error", "request body too large or unreadable")
			return
		}

		// Backward compatibility: if empty body or only registering tenant scope
		if len(bodyBytes) == 0 || strings.TrimSpace(string(bodyBytes)) == "{}" {
			scope := multitenant.FromHeaders(r.Header, nil)
			s.MultiTenant.RegisterSession(scope)
			writeJSON(w, http.StatusOK, map[string]interface{}{
				"status":     "active",
				"session_id": scope.SessionID,
			})
			return
		}

		var sess session.Session
		if err := json.Unmarshal(bodyBytes, &sess); err != nil {
			apiMessage(w, http.StatusBadRequest, "invalid_request_error", "invalid session json: "+err.Error())
			return
		}

		if sess.ID == "" {
			sess.ID = session.GenerateID()
		} else {
			cleanID, err := session.SanitizeID(sess.ID)
			if err != nil {
				sess.ID = session.GenerateID()
			} else {
				sess.ID = cleanID
			}
		}

		if err := s.SessionStore.Save(&sess); err != nil {
			apiMessage(w, http.StatusInternalServerError, "server_error", "failed to save session: "+err.Error())
			return
		}

		// Register in multi-tenant manager
		scope := multitenant.FromHeaders(r.Header, nil)
		scope.SessionID = sess.ID
		s.MultiTenant.RegisterSession(scope)

		writeJSON(w, http.StatusOK, sess)

	case http.MethodDelete:
		if err := s.SessionStore.Clear(); err != nil {
			apiMessage(w, http.StatusInternalServerError, "server_error", "failed to clear sessions: "+err.Error())
			return
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"status":  "ok",
			"cleared": true,
		})

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleSessionDetail handles subpaths under /api/sessions/
// /api/sessions/{id}: GET, PUT, PATCH, DELETE
// /api/sessions/{id}/fork: POST
func (s *StrataServer) handleSessionDetail(w http.ResponseWriter, r *http.Request) {
	subPath := strings.TrimPrefix(r.URL.Path, "/api/sessions/")
	subPath = strings.Trim(subPath, "/")
	if subPath == "" {
		s.handleSessions(w, r)
		return
	}

	parts := strings.Split(subPath, "/")
	id := parts[0]
	action := ""
	if len(parts) > 1 {
		action = parts[1]
	}

	cleanID, err := session.SanitizeID(id)
	if err != nil {
		apiMessage(w, http.StatusBadRequest, "invalid_request_error", "invalid session id: "+err.Error())
		return
	}

	// Sub-action: /api/sessions/{id}/fork
	if action == "fork" {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var forkReq struct {
			NewSessionID string `json:"new_session_id"`
			MessageIndex int    `json:"message_index"`
			NewTitle     string `json:"new_title"`
		}
		_ = json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&forkReq)

		forked, err := s.SessionStore.Fork(cleanID, forkReq.NewSessionID, forkReq.MessageIndex, forkReq.NewTitle)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				apiMessage(w, http.StatusNotFound, "not_found", "parent session not found")
				return
			}
			apiMessage(w, http.StatusInternalServerError, "server_error", "failed to fork session: "+err.Error())
			return
		}

		scope := multitenant.DefaultSecurityScope()
		scope.SessionID = forked.ID
		s.MultiTenant.RegisterSession(scope)

		writeJSON(w, http.StatusOK, forked)
		return
	}

	// Direct session operations on /api/sessions/{id}
	switch r.Method {
	case http.MethodGet:
		sess, err := s.SessionStore.Get(cleanID)
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				apiMessage(w, http.StatusNotFound, "not_found", "session not found")
				return
			}
			apiMessage(w, http.StatusInternalServerError, "server_error", "failed to read session: "+err.Error())
			return
		}
		writeJSON(w, http.StatusOK, sess)

	case http.MethodPut, http.MethodPatch:
		bodyBytes, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 32<<20))
		if err != nil {
			apiMessage(w, http.StatusBadRequest, "invalid_request_error", "request body too large")
			return
		}
		var sess session.Session
		if err := json.Unmarshal(bodyBytes, &sess); err != nil {
			apiMessage(w, http.StatusBadRequest, "invalid_request_error", "invalid json: "+err.Error())
			return
		}
		sess.ID = cleanID
		if err := s.SessionStore.Save(&sess); err != nil {
			apiMessage(w, http.StatusInternalServerError, "server_error", "failed to save session: "+err.Error())
			return
		}
		writeJSON(w, http.StatusOK, sess)

	case http.MethodDelete:
		deleted, err := s.SessionStore.Delete(cleanID)
		if err != nil {
			apiMessage(w, http.StatusInternalServerError, "server_error", "failed to delete session: "+err.Error())
			return
		}
		_ = s.MultiTenant.DeleteSession(cleanID)
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"status":     "ok",
			"deleted":    deleted,
			"session_id": cleanID,
		})

	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleSessionFork handles legacy /v1/strata/sessions/fork
func (s *StrataServer) handleSessionFork(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var body struct {
		ParentSessionID string `json:"parent_session_id"`
		NewSessionID    string `json:"new_session_id"`
		NewTitle        string `json:"new_title"`
		MessageIndex    int    `json:"message_index"`
	}
	_ = json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body)

	if body.ParentSessionID != "" {
		forked, err := s.SessionStore.Fork(body.ParentSessionID, body.NewSessionID, body.MessageIndex, body.NewTitle)
		if err == nil {
			scope := multitenant.DefaultSecurityScope()
			scope.SessionID = forked.ID
			s.MultiTenant.RegisterSession(scope)
			writeJSON(w, http.StatusOK, map[string]interface{}{
				"status":     "forked",
				"session_id": forked.ID,
				"parent_id":  body.ParentSessionID,
				"session":    forked,
			})
			return
		}
	}

	// Fallback to legacy in-memory registration if parent wasn't found on disk
	if body.NewSessionID == "" {
		body.NewSessionID = session.GenerateID()
	}
	scope := multitenant.DefaultSecurityScope()
	scope.SessionID = body.NewSessionID
	s.MultiTenant.RegisterSession(scope)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":     "forked",
		"session_id": scope.SessionID,
		"parent_id":  body.ParentSessionID,
	})
}

// handleSessionDelete handles both:
// POST /v1/strata/sessions/delete
// GET/POST /api/strata/sessions/delete?id=...
func (s *StrataServer) handleSessionDelete(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if id == "" {
		id = r.URL.Query().Get("session_id")
	}

	if id == "" && (r.Method == http.MethodPost || r.Method == http.MethodDelete) {
		var body struct {
			SessionID string `json:"session_id"`
			ID        string `json:"id"`
		}
		_ = json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body)
		if body.SessionID != "" {
			id = body.SessionID
		} else {
			id = body.ID
		}
	}

	if id == "" {
		apiMessage(w, http.StatusBadRequest, "invalid_request_error", "session_id or id parameter required")
		return
	}

	cleanID, err := session.SanitizeID(id)
	if err != nil {
		apiMessage(w, http.StatusBadRequest, "invalid_request_error", "invalid session id: "+err.Error())
		return
	}

	deleted, err := s.SessionStore.Delete(cleanID)
	if err != nil {
		apiMessage(w, http.StatusInternalServerError, "server_error", "failed to delete session: "+err.Error())
		return
	}
	_ = s.MultiTenant.DeleteSession(cleanID)

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":     "ok",
		"deleted":    deleted,
		"session_id": cleanID,
	})
}
