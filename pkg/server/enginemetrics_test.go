package server

import "testing"

// The engine prints "INFO k=v ..." once at startup; the Monitor's slots, cache and version cards read them from
// /metrics' engine block (expert_slots, expert_cache_mib, ...).
func TestMergeEngineInfoFillsTheEngineBlock(t *testing.T) {
	eng := map[string]interface{}{"version": "0.1.39", "kv": "int8"}
	mergeEngineInfo(eng, map[string]string{
		"expert_slots": "321", "expert_cache_mib": "634", "vram_free_mib": "285", "kv": "fp8", "engine": "0.1.40",
		"cvec": "none", "unrelated": "x", "expert_slots_primary": "bad",
	})
	if eng["expert_slots"] != int64(321) || eng["expert_cache_mib"] != int64(634) || eng["vram_free_mib"] != int64(285) {
		t.Errorf("numbers not merged: %v", eng)
	}
	if eng["version"] != "0.1.40" || eng["kv"] != "fp8" {
		t.Errorf("the engine's own version and kv must win: %v", eng)
	}
	if _, ok := eng["unrelated"]; ok {
		t.Error("only known facts are passed on")
	}
	if _, ok := eng["expert_slots_primary"]; ok {
		t.Error("a value that is not a number must be dropped")
	}
	mergeEngineInfo(eng, nil) // an engine that is not up: nothing changes
	if eng["expert_slots"] != int64(321) {
		t.Error("nil info must change nothing")
	}
}

func TestConversationCacheView(t *testing.T) {
	reqs := []map[string]interface{}{
		{"prompt_tokens": 100, "reused": 60}, // newest first
		{"prompt_tokens": 50, "reused": nil},
		{"prompt_tokens": 40, "reused": 0},
	}
	totals := map[string]interface{}{"reused": 60, "prompt_tokens": 190}
	v := conversationCacheView(map[string]string{"conversation_cache_mib": "2048", "conversation_cache_slots": "4"}, reqs, totals)
	if v["enabled"] != true || v["budget_mib"] != int64(2048) || v["slots"] != int64(4) {
		t.Errorf("settings: %v", v)
	}
	if v["requests"] != 3 || v["requests_reused"] != 1 || v["reused_tokens"] != 60 || v["prompt_tokens"] != 190 {
		t.Errorf("reuse figures: %v", v)
	}
	if v["last_reused"] != 60 || v["last_prompt"] != 100 {
		t.Errorf("last request: %v", v)
	}
	off := conversationCacheView(map[string]string{}, nil, map[string]interface{}{})
	if off["enabled"] != false || off["requests"] != 0 {
		t.Errorf("disabled cache: %v", off)
	}
}
