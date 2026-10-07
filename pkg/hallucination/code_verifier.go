// pkg/hallucination/code_verifier.go - Codebase AST, Filesystem & API Route Verifier
package hallucination

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
)

// Known APIRoutes in Strata
var knownAPIRoutes = map[string]bool{
	"/v1/chat/completions":               true,
	"/v1/models":                         true,
	"/v1/messages":                       true,
	"/v1/strata/version":                 true,
	"/v1/strata/status":                  true,
	"/v1/strata/math/verify":             true,
	"/v1/strata/logic/verify":            true,
	"/v1/strata/context/inject":          true,
	"/v1/strata/adaptive/status":         true,
	"/v1/strata/hallucination/evaluate":  true,
	"/v1/strata/hallucination/claims":    true,
	"/v1/strata/hallucination/metrics":   true,
	"/health":                            true,
	"/metrics":                           true,
	"/api/tags":                          true,
	"/api/generate":                      true,
	"/api/chat":                          true,
}

// CodeSymbol represents an indexed code symbol in the workspace
type CodeSymbol struct {
	Name     string `json:"name"`
	Kind     string `json:"kind"` // func, type, const, var
	FilePath string `json:"file_path"`
	Line     int    `json:"line"`
}

// CodeVerifier verifies code, filesystem, symbol and API route claims against disk
type CodeVerifier struct {
	workspaceRoot string
	mu            sync.RWMutex
	symbols       map[string][]CodeSymbol // name -> occurrences
	indexed       bool
}

// NewCodeVerifier creates a new CodeVerifier
func NewCodeVerifier(root string) *CodeVerifier {
	if root == "" {
		root = "."
	}
	return &CodeVerifier{
		workspaceRoot: root,
		symbols:       make(map[string][]CodeSymbol),
	}
}

// EnsureIndexed scans the workspace AST for symbols
func (cv *CodeVerifier) EnsureIndexed() error {
	cv.mu.Lock()
	defer cv.mu.Unlock()

	if cv.indexed {
		return nil
	}

	cv.symbols = make(map[string][]CodeSymbol)

	// Scan Go files
	_ = filepath.Walk(cv.workspaceRoot, func(path string, info os.FileInfo, err error) error {
		if err != nil || info == nil {
			return nil
		}
		// Skip hidden, vendor, git, cache dirs
		if info.IsDir() {
			name := info.Name()
			if strings.HasPrefix(name, ".") || name == "vendor" || name == "node_modules" || name == "dist" {
				return filepath.SkipDir
			}
			return nil
		}

		if strings.HasSuffix(path, ".go") {
			cv.indexGoFile(path)
		} else if strings.HasSuffix(path, ".h") || strings.HasSuffix(path, ".hpp") || strings.HasSuffix(path, ".cpp") {
			cv.indexCppFile(path)
		}
		return nil
	})

	cv.indexed = true
	return nil
}

// indexGoFile uses Go's standard AST parser
func (cv *CodeVerifier) indexGoFile(filePath string) {
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, filePath, nil, parser.ParseComments)
	if err != nil {
		return
	}

	relPath, err := filepath.Rel(cv.workspaceRoot, filePath)
	if err != nil {
		relPath = filePath
	}

	ast.Inspect(node, func(n ast.Node) bool {
		switch decl := n.(type) {
		case *ast.FuncDecl:
			if decl.Name != nil {
				pos := fset.Position(decl.Pos())
				sym := CodeSymbol{
					Name:     decl.Name.Name,
					Kind:     "func",
					FilePath: relPath,
					Line:     pos.Line,
				}
				cv.symbols[sym.Name] = append(cv.symbols[sym.Name], sym)
			}
		case *ast.TypeSpec:
			if decl.Name != nil {
				pos := fset.Position(decl.Pos())
				sym := CodeSymbol{
					Name:     decl.Name.Name,
					Kind:     "type",
					FilePath: relPath,
					Line:     pos.Line,
				}
				cv.symbols[sym.Name] = append(cv.symbols[sym.Name], sym)
			}
		}
		return true
	})
}

var cppFuncRegex = regexp.MustCompile(`(?m)^(?:[a-zA-Z0-9_:*&<>]+\s+)+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(`)

// indexCppFile uses regex scan for C/C++ function symbols
func (cv *CodeVerifier) indexCppFile(filePath string) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return
	}

	relPath, err := filepath.Rel(cv.workspaceRoot, filePath)
	if err != nil {
		relPath = filePath
	}

	lines := strings.Split(string(content), "\n")
	for lineIdx, line := range lines {
		matches := cppFuncRegex.FindAllStringSubmatch(line, -1)
		for _, m := range matches {
			if len(m) > 1 {
				name := m[1]
				// Filter common keywords
				if name == "if" || name == "for" || name == "while" || name == "switch" || name == "return" {
					continue
				}
				sym := CodeSymbol{
					Name:     name,
					Kind:     "cpp_func",
					FilePath: relPath,
					Line:     lineIdx + 1,
				}
				cv.symbols[name] = append(cv.symbols[name], sym)
			}
		}
	}
}

// VerifyFileClaim checks if a claimed file exists on disk
func (cv *CodeVerifier) VerifyFileClaim(claim *Claim) (ClaimStatus, Evidence, error) {
	relPath := strings.TrimPrefix(claim.Subject, "./")
	fullPath := filepath.Join(cv.workspaceRoot, relPath)

	info, err := os.Stat(fullPath)
	if err == nil && !info.IsDir() {
		ev := Evidence{
			ID:            fmt.Sprintf("ev-file-%s", relPath),
			SourceType:    SourceFilesystem,
			SourceID:      relPath,
			Location:      relPath,
			Authority:     AuthorityAuthoritativeRepo,
			Timestamp:     info.ModTime(),
			ExtractedFact: fmt.Sprintf("File %s exists on disk (size: %d bytes)", relPath, info.Size()),
			Provenance:    "FilesystemStatVerifier",
			Confidence:    1.0,
			Immutable:     false,
		}
		return StatusVerified, ev, nil
	}

	// Also check directly without workspace prefix in case subject was absolute or clean
	if info2, err2 := os.Stat(claim.Subject); err2 == nil && !info2.IsDir() {
		ev := Evidence{
			ID:            fmt.Sprintf("ev-file-%s", claim.Subject),
			SourceType:    SourceFilesystem,
			SourceID:      claim.Subject,
			Location:      claim.Subject,
			Authority:     AuthorityAuthoritativeRepo,
			Timestamp:     info2.ModTime(),
			ExtractedFact: fmt.Sprintf("File %s exists on disk (size: %d bytes)", claim.Subject, info2.Size()),
			Provenance:    "FilesystemStatVerifier",
			Confidence:    1.0,
			Immutable:     false,
		}
		return StatusVerified, ev, nil
	}

	// File does not exist
	ev := Evidence{
		ID:            fmt.Sprintf("ev-file-missing-%s", relPath),
		SourceType:    SourceFilesystem,
		SourceID:      relPath,
		Location:      relPath,
		Authority:     AuthorityAuthoritativeRepo,
		ExtractedFact: fmt.Sprintf("File %s does NOT exist in workspace", relPath),
		Provenance:    "FilesystemStatVerifier",
		Confidence:    1.0,
	}
	return StatusContradicted, ev, nil
}

// VerifySymbolClaim checks if a function/symbol exists in the AST symbol index
func (cv *CodeVerifier) VerifySymbolClaim(claim *Claim) (ClaimStatus, Evidence, error) {
	_ = cv.EnsureIndexed()

	cv.mu.RLock()
	defer cv.mu.RUnlock()

	syms, ok := cv.symbols[claim.Subject]
	if ok && len(syms) > 0 {
		first := syms[0]
		ev := Evidence{
			ID:            fmt.Sprintf("ev-sym-%s", claim.Subject),
			SourceType:    SourceSymbolTable,
			SourceID:      claim.Subject,
			Location:      fmt.Sprintf("%s:%d", first.FilePath, first.Line),
			Authority:     AuthorityAuthoritativeRepo,
			ExtractedFact: fmt.Sprintf("Symbol %s (%s) defined at %s:%d", first.Name, first.Kind, first.FilePath, first.Line),
			Provenance:    "ASTSymbolVerifier",
			Confidence:    1.0,
		}
		return StatusVerified, ev, nil
	}

	// Not found in symbol index
	ev := Evidence{
		ID:            fmt.Sprintf("ev-sym-missing-%s", claim.Subject),
		SourceType:    SourceSymbolTable,
		SourceID:      claim.Subject,
		Authority:     AuthorityAuthoritativeRepo,
		ExtractedFact: fmt.Sprintf("Symbol %s is not defined in codebase AST", claim.Subject),
		Provenance:    "ASTSymbolVerifier",
		Confidence:    0.95,
	}
	return StatusContradicted, ev, nil
}

// VerifyAPIClaim checks if an API endpoint route exists
func (cv *CodeVerifier) VerifyAPIClaim(claim *Claim) (ClaimStatus, Evidence, error) {
	route := claim.Subject
	// Normalize route
	route = strings.TrimSpace(route)

	if knownAPIRoutes[route] {
		ev := Evidence{
			ID:            fmt.Sprintf("ev-api-%s", route),
			SourceType:    SourceConfiguration,
			SourceID:      route,
			Authority:     AuthorityTrustedSystem,
			ExtractedFact: fmt.Sprintf("API Route %s is registered in Strata Router", route),
			Provenance:    "APIRouterVerifier",
			Confidence:    1.0,
		}
		return StatusVerified, ev, nil
	}

	ev := Evidence{
		ID:            fmt.Sprintf("ev-api-missing-%s", route),
		SourceType:    SourceConfiguration,
		SourceID:      route,
		Authority:     AuthorityTrustedSystem,
		ExtractedFact: fmt.Sprintf("API Route %s is NOT registered in Strata Router", route),
		Provenance:    "APIRouterVerifier",
		Confidence:    1.0,
	}
	return StatusContradicted, ev, nil
}
