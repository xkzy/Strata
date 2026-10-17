package main

import "testing"

func TestFindEngine(t *testing.T) {
	cases := []struct {
		name, pgrep string
		want        bool
	}{
		{"nothing", "", false},
		{"a shell that quotes the words", "222059 /bin/bash -c pgrep -af \"strata --serve|strata serve\"", false},
		{"the tuner itself", "100 bin/strata tune --config strata-x.json", false},
		{"the engine", "200 /x/engine/strata --serve --pack /p --mmap-experts", true},
		{"the Go server", "300 bin/strata serve --config strata-x.json", true},
		{"the runtime server", "400 /x/bin/strata_rt_server --port 9", true},
		{"the standalone server", "500 /x/bin/strata-server --port 8080", true},
	}
	for _, c := range cases {
		if got := findEngine(c.pgrep, 1) != ""; got != c.want {
			t.Errorf("%s: found = %v, want %v", c.name, got, c.want)
		}
	}
	if findEngine("200 /x/engine/strata --serve", 200) != "" {
		t.Error("the tuner's own process is skipped")
	}
}
