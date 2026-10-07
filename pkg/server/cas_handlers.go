package server

import (
	"encoding/json"
	"net/http"

	"strata/pkg/mathruntime"
)

func (s *StrataServer) handleCASSolve(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":{"message":"Method not allowed","type":"invalid_request_error"}}`, http.StatusMethodNotAllowed)
		return
	}

	var prob mathruntime.CASProblem
	if err := json.NewDecoder(r.Body).Decode(&prob); err != nil {
		http.Error(w, `{"error":{"message":"Invalid JSON body","type":"invalid_request_error"}}`, http.StatusBadRequest)
		return
	}

	// The session keys the math runtime's variables and the tenant picks backends: both come from the request's scope
	// (headers, as in every other handler), never from the body, which could name another caller's session.
	sc := requestScope(r, "", "")
	prob.TenantID, prob.SessionID = sc.TenantID, sc.SessionID

	cas := mathruntime.GetUnifiedCASEngine()
	res, err := cas.Solve(r.Context(), prob)
	if err != nil {
		writeJSON(w, http.StatusOK, mathruntime.CASResult{
			Status:       mathruntime.StatusExecutionError,
			ErrorMessage: err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, res)
}

func (s *StrataServer) handleCASConstants(w http.ResponseWriter, r *http.Request) {
	consts := mathruntime.GetConstantRegistry().ListAll()
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"count":     len(consts),
		"constants": consts,
	})
}

func (s *StrataServer) handleCASFormulas(w http.ResponseWriter, r *http.Request) {
	domain := r.URL.Query().Get("domain")
	reg := mathruntime.GetFormulaRegistry()
	var formulas []mathruntime.ScientificFormula
	if domain != "" {
		formulas = reg.ListByDomain(mathruntime.CASDomainType(domain))
	} else {
		formulas = reg.ListAll()
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"count":    len(formulas),
		"formulas": formulas,
	})
}
