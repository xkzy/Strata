// pkg/frontend/frontend.go - Embedded Web UI Frontend Server in Go
package frontend

import (
	"embed"
	"io/fs"
	"net/http"
	"path/filepath"
	"strings"
)

//go:embed all:web
var embeddedWeb embed.FS

type FrontendHandler struct {
	fileServer http.Handler
}

func NewFrontendHandler() *FrontendHandler {
	sub, err := fs.Sub(embeddedWeb, "web")
	if err != nil {
		return &FrontendHandler{fileServer: http.NotFoundHandler()}
	}
	return &FrontendHandler{
		fileServer: http.FileServer(http.FS(sub)),
	}
}

func (h *FrontendHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	cleanPath := strings.TrimPrefix(path, "/")

	// Direct routes for Monitor UI
	if path == "/monitor" || path == "/monitor.html" || path == "/api-monitor" {
		data, err := embeddedWeb.ReadFile("web/monitor.html")
		if err == nil {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.WriteHeader(http.StatusOK)
			w.Write(data)
			return
		}
	}

	// SPA routes for Main Web App (Chat, Setup, Monitor, Settings, About)
	if path == "/" || path == "/chat" || path == "/setup" || path == "/settings" || path == "/about" || path == "/index.html" {
		data, err := embeddedWeb.ReadFile("web/index.html")
		if err == nil {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.WriteHeader(http.StatusOK)
			w.Write(data)
			return
		}
	}

	// Route /web/<file> -> web/<file>
	if strings.HasPrefix(cleanPath, "web/") {
		relPath := strings.TrimPrefix(cleanPath, "web/")
		data, err := embeddedWeb.ReadFile("web/" + relPath)
		if err == nil {
			setMimeType(w, relPath)
			w.WriteHeader(http.StatusOK)
			w.Write(data)
			return
		}
	}

	// Route /fonts/<file> -> web/fonts/<file>
	if strings.HasPrefix(cleanPath, "fonts/") {
		data, err := embeddedWeb.ReadFile("web/" + cleanPath)
		if err == nil {
			w.Header().Set("Content-Type", "font/woff2")
			w.Header().Set("Cache-Control", "max-age=86400")
			w.WriteHeader(http.StatusOK)
			w.Write(data)
			return
		}
	}

	// Check if file exists directly under web/
	data, err := embeddedWeb.ReadFile("web/" + cleanPath)
	if err == nil {
		setMimeType(w, cleanPath)
		w.WriteHeader(http.StatusOK)
		w.Write(data)
		return
	}

	h.fileServer.ServeHTTP(w, r)
}

func setMimeType(w http.ResponseWriter, filename string) {
	ext := strings.ToLower(filepath.Ext(filename))
	switch ext {
	case ".html", ".htm":
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
	case ".js":
		w.Header().Set("Content-Type", "application/javascript; charset=utf-8")
	case ".css":
		w.Header().Set("Content-Type", "text/css; charset=utf-8")
	case ".svg":
		w.Header().Set("Content-Type", "image/svg+xml")
	case ".woff2":
		w.Header().Set("Content-Type", "font/woff2")
	case ".json":
		w.Header().Set("Content-Type", "application/json")
	case ".png":
		w.Header().Set("Content-Type", "image/png")
	case ".ico":
		w.Header().Set("Content-Type", "image/x-icon")
	}
}
