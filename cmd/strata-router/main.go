// strata-router fronts the MoE and dense engines: MoE first, dense when a reply looks like a failure (see pkg/router).
package main

import (
	"flag"
	"log"
	"net/http"

	"strata/pkg/router"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:8000", "listen address")
	fastURL := flag.String("fast-url", "http://127.0.0.1:8080", "MoE engine address")
	fastCmd := flag.String("fast-cmd", "./run.sh serve -config strata-coder-iq1_m.json -port 8080", "command that starts the MoE engine")
	denseURL := flag.String("dense-url", "http://127.0.0.1:8082", "dense engine address")
	denseCmd := flag.String("dense-cmd", "./bin/strata serve -config strata-dense.json -port 8082", "command that starts the dense engine")
	n := flag.Int("dense-requests", 20, "requests the dense engine serves after an escalation")
	flag.Parse()
	r := &router.Router{
		Fast: router.ProcBackend("moe", *fastURL, *fastCmd), Dense: router.ProcBackend("dense", *denseURL, *denseCmd),
		DenseRequests: *n, Wait: router.WaitReady,
	}
	http.HandleFunc("/v1/chat/completions", r.Chat)
	log.Printf("strata-router on %s", *addr)
	log.Fatal(http.ListenAndServe(*addr, nil))
}
