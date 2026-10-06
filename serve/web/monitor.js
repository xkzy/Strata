"use strict";
const $ = id => document.getElementById(id);
let selected = null, detail = null, pane = "input", records = [], state = null, operating = false, stamp = "", actionError = "";
let apiKey = sessionStorage.getItem("strata.monitor.key") || "";
document.documentElement.dataset.theme = localStorage.getItem("strata.theme") || "dark";

const seconds = n => typeof n === "number" ? `${n.toFixed(2)} s` : "—";
const speed = n => typeof n === "number" ? n.toFixed(1) : "—";
const active = r => !["completed", "error", "disconnected"].includes(r.state);
const formatNumber = n => typeof n === "number" ? n.toLocaleString() : "—";

function text(id, value) {
  const el = $(id);
  if (el && el.textContent !== String(value)) el.textContent = value;
}

async function api(path, body) {
  const response = await fetch(path, {
    cache: "no-store",
    headers: {
      ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
      ...(body !== undefined ? { "Content-Type": "application/json" } : {})
    },
    ...(body !== undefined ? { method: "POST", body: JSON.stringify(body) } : {})
  });
  const value = await response.json();
  if (!response.ok) {
    if (response.status === 401) $("auth").hidden = false;
    throw new Error(value.error?.message || `HTTP ${response.status}`);
  }
  return value;
}

function notice(message) {
  $("error").hidden = !message;
  text("error", message || "");
}

function list() {
  const query = $("search").value.toLowerCase(), filter = $("filter").value;
  const visible = records.filter(r => {
    const scopeStr = r.scope ? `${r.scope.tenant_id} ${r.scope.agent_id} ${r.scope.session_id}` : "";
    const matchQuery = `${r.id} ${r.path} ${r.model} ${scopeStr}`.toLowerCase().includes(query);
    const matchFilter = (filter === "all" || (filter === "active" ? active(r) : r.state === filter));
    return matchQuery && matchFilter;
  });

  const nextStamp = JSON.stringify([visible, selected]);
  if (stamp === nextStamp) return;
  stamp = nextStamp;

  const focusId = document.activeElement?.dataset.request;
  $("requests").replaceChildren();
  if (!visible.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = records.length ? "No matching requests." : "No requests yet. API calls will appear here automatically.";
    $("requests").append(empty);
  }

  for (const r of visible) {
    const button = document.createElement("button");
    button.className = `request${selected === r.id ? " selected" : ""}`;
    button.dataset.request = r.id;
    button.setAttribute("aria-label", `Request ${r.id}`);
    button.setAttribute("aria-pressed", String(selected === r.id));

    const row = document.createElement("div");
    row.className = "row";
    const time = document.createElement("span");
    time.textContent = new Date(r.started_at * 1000).toLocaleTimeString();
    const tag = document.createElement("span");
    tag.className = `tag ${r.state}`;
    tag.textContent = r.state;
    row.append(time, tag);

    const endpoint = document.createElement("div");
    endpoint.className = "endpoint";
    endpoint.textContent = `POST ${r.path}`;

    const meta = document.createElement("div");
    meta.className = "small";
    const scopeTag = r.scope?.session_id ? ` · session:${r.scope.session_id.slice(0, 8)}` : "";
    meta.textContent = `${r.id} · ${seconds(r.wallclock_s)} · ${r.stream ? "stream" : "JSON"}${scopeTag}${r.http_status ? ` · HTTP ${r.http_status}` : ""}`;

    button.append(row, endpoint, meta);
    button.onclick = async () => {
      selected = r.id;
      detail = null;
      list();
      await showDetail();
    };
    $("requests").append(button);
    if (focusId === r.id) button.focus({ preventScroll: true });
  }
}

function content() {
  if (!detail) return;

  let value = "";
  let note = "";

  if (pane === "input") {
    value = detail.input || "No input body.";
    note = "Original request body.";
  } else if (pane === "output") {
    value = detail.output || "No output generated.";
    try { value = JSON.stringify(JSON.parse(value), null, 2); } catch (_) {}
    note = detail.error ? "Raw model output retained for diagnosis. The API returned an error." : "Model answer. Formatted for readability.";
  } else if (pane === "reasoning") {
    value = detail.reasoning || "No separate reasoning content.";
    note = "Extracted reasoning content (<think>), when available.";
  } else if (pane === "vctx") {
    const vctxInfo = detail.virtual_context || {};
    value = JSON.stringify({
      virtual_context_tokens: vctxInfo.virtual_tokens ?? "Active tracking",
      physical_window_tokens: vctxInfo.physical_tokens ?? detail.usage?.prompt_tokens ?? "Bounded",
      compaction_applied: vctxInfo.compacted ?? false,
      retrieved_rag_items: vctxInfo.retrieved_items ?? [],
      token_budget: {
        physical_limit: state?.cache_max_tokens ?? 32768,
        reserve_tokens: 1024
      }
    }, null, 2);
    note = "Virtual Context & RAG Index state for this request.";
  } else if (pane === "tool") {
    const toolInfo = detail.tool_runtime || {};
    value = JSON.stringify({
      tool_calls: toolInfo.tool_calls ?? 0,
      raw_tokens_produced: toolInfo.raw_tokens ?? 0,
      emitted_tokens: toolInfo.emitted_tokens ?? 0,
      tokens_saved_ratio: toolInfo.reduction_ratio ?? "1.0x",
      cas_hashes: toolInfo.cas_hashes ?? []
    }, null, 2);
    note = "Tool execution state, observations, and Content-Addressed Storage (CAS) hashes.";
  } else if (pane === "scope") {
    value = JSON.stringify(detail.scope || {
      tenant_id: "default",
      user_id: "default",
      workspace_id: "default",
      agent_id: "default",
      session_id: "default",
      sharing_scope: "SESSION"
    }, null, 2);
    note = "Multi-Tenant & Agent security isolation scope.";
  } else if (pane === "response") {
    value = detail.response || (detail.stream ? "Streaming response: see Output and errors." : "No response payload.");
    try { value = JSON.stringify(JSON.parse(value), null, 2); } catch (_) {}
    note = "Complete API response payload.";
  }

  text("content", value);
  text("content-note", note + (detail[`${pane}_truncated`] ? " Capture truncated at 256K chars." : ""));
}

async function showDetail() {
  if (!selected) return;
  const id = selected;
  const value = await api(`/api/requests?id=${encodeURIComponent(id)}`);
  if (selected !== id) return;
  detail = value;

  $("selection").hidden = false;
  $("no-selection").hidden = true;

  text("detail-title", `${value.id} · ${value.path}`);
  text("detail-status", value.state);
  $("detail-status").className = `tag ${value.state}`;
  text("detail-time", new Date(value.started_at * 1000).toLocaleString());
  text("detail-wall", seconds(value.wallclock_s ?? ((Date.now() / 1000) - value.started_at)));
  text("detail-load", seconds(value.load_s));
  text("detail-queue", seconds(value.queue_s));
  text("detail-first", seconds(value.first_token_s));
  text("detail-prompt", value.usage?.prompt_tokens ?? value.usage?.input_tokens ?? "—");
  text("detail-tokens", value.usage?.completion_tokens ?? value.usage?.output_tokens ?? "—");
  text("detail-speed", speed(value.timings?.predicted_per_second));

  const vTokens = value.virtual_context?.virtual_tokens ?? value.usage?.prompt_tokens;
  text("detail-vctx", typeof vTokens === "number" ? `${vTokens.toLocaleString()} tok` : "Active");

  $("detail-error").hidden = !value.error;
  text("detail-error", value.error?.message || "");
  content();
}

async function refresh() {
  try {
    const [status, history] = await Promise.all([api("/v1/status"), api("/api/requests")]);
    state = status;
    records = history.requests;
    status.loaded = history.loaded;
    status.auto_load = history.auto_load;

    if (selected && !records.some(r => r.id === selected)) {
      selected = detail = null;
      $("selection").hidden = true;
      $("no-selection").hidden = false;
    }

    const inflight = Math.max(status.activity.in_flight, records.filter(active).length);
    const loading = records.some(r => r.state === "loading");

    text("loaded", loading ? "Loading…" : inflight ? "In use" : status.loaded ? "Loaded" : "Unloaded");
    $("model-state").className = `value ${inflight || loading ? "busy" : status.loaded ? "loaded" : ""}`;
    text("model", status.model);

    // Arch and SWA Badge
    const archName = status.architecture || "qwen_moe";
    const swaWindow = status.swa?.window_size ? `SWA: ${status.swa.window_size}` : "SWA Ring";
    text("arch-badge", `${archName.toUpperCase()} · ${swaWindow}`);

    // Compute Hardware (dGPU & iGPU)
    const mac = status.machine || {};
    const gpuInfo = mac.gpu || {};
    const igpuInfo = mac.igpu || {};

    if (gpuInfo.name) {
      const gpuUtil = gpuInfo.util_pct !== undefined && gpuInfo.util_pct !== null ? `${gpuInfo.util_pct}%` : "0%";
      const mainGpu = gpuInfo.name.split('+')[0].trim();
      text("hw-gpu", `${mainGpu}`);
    } else {
      text("hw-gpu", mac.ram ? `CPU (${mac.ram.used_gib} / ${mac.ram.total_gib} GiB)` : "Host CPU Engine");
    }

    if (igpuInfo.name) {
      const igpuUtil = igpuInfo.util_pct !== undefined && igpuInfo.util_pct !== null ? `${igpuInfo.util_pct}%` : "0%";
      const gttStr = igpuInfo.gtt_total_mib ? ` · ${Math.round(igpuInfo.gtt_total_mib / 1024)} GB GTT` : "";
      text("hw-igpu", `iGPU: ${igpuInfo.name} (${igpuUtil}${gttStr})`);
    } else if (gpuInfo.name && gpuInfo.name.includes("[iGPU]")) {
      const igpuPart = gpuInfo.name.split('+').find(p => p.includes('[iGPU]')) || "iGPU Active";
      text("hw-igpu", `iGPU: ${igpuPart.trim()}`);
    } else {
      text("hw-igpu", "iGPU: Shared System DRAM / Host");
    }

    // Runtime Stats (Virtual Context, CAS, Multi-Tenant)
    const rt = status.runtime_stats || {};
    const vRatio = rt.context_compression_ratio ? `${rt.context_compression_ratio}x` : "1.0x";
    text("vctx-ratio", vRatio);
    text("vctx-detail", `${formatNumber(rt.total_virtual_tokens || 0)} virt / ${formatNumber(rt.total_physical_tokens || 0)} phys`);

    const toolSaved = formatNumber(rt.tool_tokens_saved || 0);
    text("tool-saved", `${toolSaved} tok`);
    const casObj = rt.cas_storage?.total_objects || 0;
    const casKb = rt.cas_storage?.total_bytes ? Math.round(rt.cas_storage.total_bytes / 1024) : 0;
    text("cas-detail", `${casObj} CAS items (${casKb} KB)`);

    const activeSessions = rt.active_sessions || 0;
    text("tenant-stats", `${activeSessions} sessions`);
    text("agent-detail", `${rt.active_tenants || 1} tenants · ${rt.active_agents || 1} agents`);

    text("compactions", formatNumber(rt.total_compaction_events || 0));
    text("count", records.length);
    text("active", inflight ? `${inflight} active / queued` : status.auto_load ? "Loads automatically on next request" : "No active requests");

    const last = records.find(r => !active(r));
    text("wall", last ? seconds(last.wallclock_s) : "—");
    text("speed", speed(last?.timings?.predicted_per_second));

    $("load").disabled = operating || Boolean(inflight) || status.loaded;
    $("unload").disabled = operating || Boolean(inflight) || !status.loaded;

    list();
    if (selected) await showDetail();
    text("updated", `Live · updated ${new Date().toLocaleTimeString()}`);
    if (!operating) notice(actionError);
  } catch (error) {
    notice(error.message);
    text("updated", "Disconnected · retrying");
    $("load").disabled = $("unload").disabled = true;
  }
}

async function control(load) {
  actionError = "";
  operating = true;
  $("load").disabled = $("unload").disabled = true;
  text("loaded", load ? "Loading…" : "Unloading…");
  try {
    await api(load ? "/load" : "/unload", {});
    notice("");
  } catch (error) {
    actionError = error.message;
    notice(actionError);
  } finally {
    operating = false;
    await refresh();
  }
}

$("load").onclick = () => control(true);
$("unload").onclick = () => control(false);
$("key").onclick = () => {
  $("auth").hidden = !$("auth").hidden;
  if (!$("auth").hidden) $("api-key").focus();
};
$("auth").onsubmit = async event => {
  event.preventDefault();
  apiKey = $("api-key").value.trim();
  sessionStorage.setItem("strata.monitor.key", apiKey);
  $("auth").hidden = true;
  await refresh();
};
$("theme").onclick = () => {
  const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("strata.theme", theme);
};
$("search").oninput = list;
$("filter").onchange = list;

document.querySelectorAll("[data-pane]").forEach(button => {
  button.onclick = () => {
    pane = button.dataset.pane;
    document.querySelectorAll("[data-pane]").forEach(b => b.setAttribute("aria-selected", String(b === button)));
    content();
  };
});

$("copy").onclick = async () => {
  try {
    await navigator.clipboard.writeText($("content").textContent);
    text("copy", "Copied");
    setTimeout(() => text("copy", "Copy"), 1500);
  } catch (_) {
    notice("Clipboard unavailable; select the text to copy it.");
  }
};

async function poll() {
  await refresh();
  setTimeout(poll, 2000);
}
poll();
