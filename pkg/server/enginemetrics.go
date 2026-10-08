package server

import "strconv"

// engineIntFacts are the numeric "INFO k=v" facts the engine prints at startup that the Monitor shows.
var engineIntFacts = []string{
	"context", "kv_resident", "expert_slots", "expert_cache_mib", "expert_slots_primary", "expert_cache_primary_mib",
	"vram_free_mib", "conversation_cache_mib", "conversation_cache_slots", "pool_workers", "arena_mib",
}

// mergeEngineInfo adds the engine's startup facts to /metrics' engine block. The engine's own version and kv
// mode replace the ones the server assumed; a missing or non-numeric fact is left out.
func mergeEngineInfo(engine map[string]interface{}, info map[string]string) {
	for _, k := range engineIntFacts {
		if v, err := strconv.ParseInt(info[k], 10, 64); err == nil {
			engine[k] = v
		}
	}
	if v := info["engine"]; v != "" {
		engine["version"] = v
	}
	if v := info["kv"]; v != "" {
		engine["kv"] = v
	}
}

func infoInt(info map[string]string, key string) (int64, bool) {
	v, err := strconv.ParseInt(info[key], 10, 64)
	return v, err == nil
}

func asInt(v interface{}) int {
	switch n := v.(type) {
	case int:
		return n
	case int64:
		return int(n)
	case float64:
		return int(n)
	}
	return 0
}

// conversationCacheView is the Monitor's Conversation cache card: the cache's settings (the engine's opt-in
// --conversation-cache-mib) and how much of the prompts it gave back. requests are newest first.
func conversationCacheView(info map[string]string, requests []map[string]interface{}, totals map[string]interface{}) map[string]interface{} {
	mib, hasMib := infoInt(info, "conversation_cache_mib")
	slots, hasSlots := infoInt(info, "conversation_cache_slots")
	out := map[string]interface{}{
		"enabled": hasMib && mib > 0, "budget_mib": nil, "slots": nil,
		"parked": 0, "bytes": 0, "evictions": 0, "parks": 0, "restores": 0, "last_event": nil, "last_tokens": nil, "last_at": nil,
		"requests": len(requests), "requests_reused": 0,
		"reused_tokens": asInt(totals["reused"]), "prompt_tokens": asInt(totals["prompt_tokens"]),
		"last_reused": nil, "last_prompt": nil,
	}
	if hasMib {
		out["budget_mib"] = mib
	}
	if hasSlots {
		out["slots"] = slots
	}
	for _, r := range requests {
		if asInt(r["reused"]) > 0 {
			out["requests_reused"] = out["requests_reused"].(int) + 1
		}
	}
	if len(requests) > 0 {
		out["last_reused"], out["last_prompt"] = asInt(requests[0]["reused"]), asInt(requests[0]["prompt_tokens"])
	}
	return out
}
