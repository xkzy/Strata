package multitenant

import (
	"net/http"
	"testing"
)

func TestMultiTenantManager(t *testing.T) {
	mgr := NewMultiTenantManager()

	headers := make(http.Header)
	headers.Set("X-Tenant-ID", "tenant_alpha")
	headers.Set("X-Session-ID", "sess_alpha_1")

	scope := FromHeaders(headers, nil)
	if scope.TenantID != "tenant_alpha" || scope.SessionID != "sess_alpha_1" {
		t.Errorf("scope parsing failed: %+v", scope)
	}

	mgr.RegisterSession(scope)
	list := mgr.ListSessions("tenant_alpha")
	if len(list) != 1 {
		t.Errorf("expected 1 session")
	}

	deleted := mgr.DeleteSession("sess_alpha_1")
	if !deleted || len(mgr.ListSessions("tenant_alpha")) != 0 {
		t.Errorf("session deletion failed")
	}
}
