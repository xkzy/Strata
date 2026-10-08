// pkg/frontend/frontend_test.go - Unit Tests for Embedded Frontend
package frontend

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestFrontendHandler_ServeIndexAndAssets(t *testing.T) {
	h := NewFrontendHandler()
	ts := httptest.NewServer(h)
	defer ts.Close()

	// 1. Root /
	resp, err := http.Get(ts.URL + "/")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /: %v", err)
	}
	contentType := resp.Header.Get("Content-Type")
	if !strings.Contains(contentType, "text/html") {
		t.Errorf("expected text/html for root, got %s", contentType)
	}
	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "Strata") {
		t.Errorf("expected HTML body to contain 'Strata'")
	}

	// 2. /setup SPA route
	respSetup, err := http.Get(ts.URL + "/setup")
	if err != nil || respSetup.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /setup: %v", err)
	}

	// 3. /monitor route
	respMon, err := http.Get(ts.URL + "/monitor")
	if err != nil || respMon.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /monitor: %v", err)
	}
	monBody, _ := io.ReadAll(respMon.Body)
	if !strings.Contains(string(monBody), "Monitor") {
		t.Errorf("expected monitor.html content")
	}

	// 4. /web/app.css
	respCSS, err := http.Get(ts.URL + "/web/app.css")
	if err != nil || respCSS.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /web/app.css: %v", err)
	}
	if !strings.Contains(respCSS.Header.Get("Content-Type"), "text/css") {
		t.Errorf("expected text/css, got %s", respCSS.Header.Get("Content-Type"))
	}

	// 5. /web/app.js
	respJS, err := http.Get(ts.URL + "/web/app.js")
	if err != nil || respJS.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /web/app.js: %v", err)
	}
	if !strings.Contains(respJS.Header.Get("Content-Type"), "javascript") {
		t.Errorf("expected javascript, got %s", respJS.Header.Get("Content-Type"))
	}

	// 6. /web/sprite.svg
	respSVG, err := http.Get(ts.URL + "/web/sprite.svg")
	if err != nil || respSVG.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /web/sprite.svg: %v", err)
	}
	if !strings.Contains(respSVG.Header.Get("Content-Type"), "image/svg+xml") {
		t.Errorf("expected image/svg+xml, got %s", respSVG.Header.Get("Content-Type"))
	}

	// 7. /fonts/outfit-latin-wght.woff2
	respFont, err := http.Get(ts.URL + "/fonts/outfit-latin-wght.woff2")
	if err != nil || respFont.StatusCode != http.StatusOK {
		t.Fatalf("failed GET /fonts/outfit-latin-wght.woff2: %v", err)
	}
	if !strings.Contains(respFont.Header.Get("Content-Type"), "font/woff2") {
		t.Errorf("expected font/woff2, got %s", respFont.Header.Get("Content-Type"))
	}
}

// The session id keys the conversation's server-side state (the math variables), so it must not be guessable.
func TestAppJSSessionIDsUseTheCSPRNG(t *testing.T) {
	data, err := embeddedWeb.ReadFile("web/app.js")
	if err != nil {
		t.Fatal(err)
	}
	src := string(data)
	if strings.Contains(src, "Math.random") {
		t.Error("app.js must not use Math.random (not a CSPRNG)")
	}
	if strings.Contains(src, "crypto.randomUUID") {
		t.Error("crypto.randomUUID needs a secure context: a page on a plain-http LAN address would break")
	}
	if !strings.Contains(src, "crypto.getRandomValues(new Uint8Array(16))") || strings.Count(src, "randomId()") < 3 {
		t.Error("app.js must make its session ids with randomId() (128 bits from crypto.getRandomValues)")
	}
}

// The server reports live.state "unloaded" when no model is loaded; the Monitor's model-state card must show it
// instead of lighting no badge and saying "Waiting for a request".
func TestMonitorShowsTheUnloadedState(t *testing.T) {
	html, err := embeddedWeb.ReadFile("web/index.html")
	if err != nil {
		t.Fatal(err)
	}
	js, err := embeddedWeb.ReadFile("web/app.js")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(html), `data-s="unloaded"`) {
		t.Error(`index.html needs a state badge with data-s="unloaded"`)
	}
	if !strings.Contains(string(js), `live.state === "unloaded"`) {
		t.Error("app.js must label the unloaded state")
	}
}

func TestMonitorShowsLoadingAndErrorStates(t *testing.T) {
	html, _ := embeddedWeb.ReadFile("web/index.html")
	js, _ := embeddedWeb.ReadFile("web/app.js")
	if !strings.Contains(string(html), `data-s="loading"`) {
		t.Error(`index.html needs a state badge with data-s="loading"`)
	}
	for _, s := range []string{`live.state === "loading"`, `live.state === "error"`} {
		if !strings.Contains(string(js), s) {
			t.Errorf("app.js must label %s", s)
		}
	}
}
