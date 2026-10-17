// pkg/frontend/web/app.js - the Strata web app: Chat, Monitor, About. No framework, no network beyond this server.
// The Monitor tab rebuilds PR #22's dashboard idea (code-martin) on the server's own /metrics.
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
// a context size: 32768 -> "32K" (powers of two), else like kfmt
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));   // memory: binary GB, as Windows shows it

const store = {
  get(k, d) { try { const v = localStorage.getItem("strata." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("strata." + k, JSON.stringify(v)); } catch (e) { /* private mode: in memory only */ } },
};

// The API key is a credential: it lives in this tab's sessionStorage (gone when the tab closes), never in localStorage,
// which every page script on this origin and every later session can read. An old localStorage copy is removed.
const secret = {
  get() { try { localStorage.removeItem("strata.apikey"); return sessionStorage.getItem("strata.apikey") || ""; } catch (e) { return ""; } },
  set(v) { try { v ? sessionStorage.setItem("strata.apikey", v) : sessionStorage.removeItem("strata.apikey"); } catch (e) { /* in memory only */ } },
};

// ------------------------------------------------------------------ toasts
function toast(kind, title, text = "", ms = 3500, action = null) {
  const names = {info: "info", success: "check", warn: "warning", error: "error"};
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.innerHTML = `${icon(names[kind] || "info")}<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  if (action) {
    const b = document.createElement("button");
    b.className = "st-btn st-btn--secondary";
    b.style.height = "32px";
    b.style.marginLeft = "auto";
    b.textContent = action.label;
    b.onclick = () => { action.run(); el.remove(); };
    el.appendChild(b);
  }
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {                                   // http on another host: no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (btn) {
    const use = btn.querySelector("use");
    use.setAttribute("href", `${SPRITE}#i-check`);
    setTimeout(() => use.setAttribute("href", `${SPRITE}#i-copy`), 1500);
  }
  toast("success", "Copied to clipboard", "", 1800);
}

// ------------------------------------------------------------------ theme and tabs
// the system's theme until the user picks one (only a click is saved)
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("strata.theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `${SPRITE}#i-${t === "dark" ? "sun" : "moon"}`);
  $("dark-toggle").setAttribute("aria-checked", String(t === "dark"));
}
const flipTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("theme-btn").onclick = flipTheme;
$("dark-toggle").onclick = flipTheme;
setTheme(document.documentElement.dataset.theme || "light", false);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let saved = null;
  try { saved = localStorage.getItem("strata.theme"); } catch (err) { /* ignore */ }
  if (!saved) setTheme(e.matches ? "dark" : "light", false);
});

let tab = "chat";
function showTab(name) {
  tab = ["chat", "setup", "monitor", "about"].includes(name) ? name : "chat";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of ["chat", "setup", "monitor", "about"]) $(`view-${v}`).hidden = v !== tab;
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${tab}`);
  if (tab === "chat") $("input").focus();
  if (tab === "setup") loadSetup();
  if (tab === "about") loadConfig();
  if (lastMetrics) render(lastMetrics);
}
for (const b of document.querySelectorAll(".st-tab")) b.onclick = () => showTab(b.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ------------------------------------------------------------------ server access
// 128 random bits from the browser's CSPRNG (getRandomValues works on a plain-http LAN address too; the UUID helper needs https or localhost)
function randomId() {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}
function getSessionId() {
  if (typeof activeSessionId !== "undefined" && activeSessionId) return activeSessionId;
  let sid = store.get("session_id", "");
  if (!sid) {
    sid = "sess-" + randomId();
    store.set("session_id", sid);
  }
  return sid;
}

function headers(json = false) {
  const h = {};
  const key = secret.get();
  if (key) h.Authorization = "Bearer " + key;
  h["X-Session-ID"] = getSessionId();
  if (json) h["Content-Type"] = "application/json";
  return h;
}
$("api-key").value = secret.get();
$("api-key").onchange = () => {
  secret.set($("api-key").value.trim());
  toast("success", "API key saved", "Kept for this tab only; enter it again after closing the tab.");
  if (typeof updateClaudeCodeBlock === "function") updateClaudeCodeBlock();
  if (lastMetrics) render(lastMetrics);
};

let health = {model: "strata", images: false, max_context: 0, models: []};

function syncModelSelect(models) {
  const sel = $("model-select");
  if (!sel) return;
  const list = models && models.length ? models : (health.models && health.models.length ? health.models : [{id: health.model, name: health.model}]);
  const stored = store.get("active_model", health.model);
  const currentVal = sel.value || stored;
  
  const currentOptions = Array.from(sel.options).map(o => o.value).join(",");
  const newOptions = list.map(m => m.id).join(",");
  if (currentOptions !== newOptions) {
    sel.innerHTML = list.map(m => `<option value="${esc(m.id)}">${esc(m.name || m.id)}</option>`).join("");
  }
  
  if (list.some(m => m.id === currentVal)) {
    sel.value = currentVal;
  } else if (list.length > 0) {
    sel.value = list[0].id;
  }
  
  sel.onchange = () => {
    store.set("active_model", sel.value);
    $("chat-empty-sub").textContent = `${sel.value} runs on this PC. Nothing leaves it.`;
    toast("info", "Active Model", `Switched to ${sel.value}`);
  };
}

function getSelectedModel() {
  const sel = $("model-select");
  return (sel && sel.value) || store.get("active_model", "") || health.model;
}

async function loadHealth() {
  try {
    const r = await fetch("health", {headers: headers()});
    if (r.ok) {
      health = await r.json();
      $("attach-btn").title = health.images ? "Attach a text file or a picture (or drop it here)"
                                            : "Attach a text file (or drop it here)";
      syncModelSelect(health.models);
      const curModel = getSelectedModel();
      $("chat-empty-sub").textContent = `${curModel} runs on this PC. Nothing leaves it.`;
    }
  } catch (e) {
    setTimeout(loadHealth, 2000);
  }
}

// ------------------------------------------------------------------ Monitor
const METRICS = [
  {key: "speed", label: "Speed", icon: "gauge", unit: "t/s", series: "tok_s"},
  {key: "gpu", label: "GPU load", icon: "gpu", unit: "%", series: "gpu_util", max: 100},
  {key: "igpu", label: "iGPU load", icon: "gpu", unit: "%", series: "igpu_util", max: 100},
  {key: "vram", label: "VRAM", icon: "layers", unit: "GB", series: "gpu_mem_used"},
  {key: "temp", label: "GPU temp", icon: "thermometer", unit: "°C", series: "gpu_temp", tone: "warn"},
  {key: "power", label: "Power", icon: "bolt", unit: "W", series: "gpu_power"},
  {key: "pcie", label: "PCIe", icon: "link", unit: "", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", label: "CPU", icon: "cpu", unit: "%", series: "cpu", max: 100},
  {key: "disk", label: "Disk read", icon: "disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="st-card metric-card"><div class="st-metric">
    <span class="st-metric__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    ${m.key === "speed" ? `<div class="speed-values">
      <div><span class="st-metric__value" id="mv-speed">-</span><span class="st-metric__sub" id="ms-speed">Decode</span></div>
      <div class="speed-prefill"><span class="st-metric__value" id="mv-prefill">-</span><span class="st-metric__sub" id="ms-prefill">Prefill</span></div>
    </div>` : `<span class="st-metric__value" id="mv-${m.key}">–</span>
    <span class="st-metric__sub" id="ms-${m.key}"></span>`}
    <svg class="st-metric__spark" id="sp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
      ${m.key === "speed" ? `<g id="sp-prefill" class="speed-prefill"><path class="area" fill="currentColor" opacity=".12"/>
        <path class="line" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"
        stroke-linecap="round" vector-effect="non-scaling-stroke"/></g>` : ""}</svg>
  </div></div>`).join("");

function spark(id, values, max) {
  const svg = $(id);
  const v = (values || []).map((x) => (x == null ? 0 : x));
  if (v.length < 2) { svg.querySelector(".line").setAttribute("d", ""); svg.querySelector(".area").setAttribute("d", ""); return; }
  const top = Math.max(max || 0, ...v, 1e-9);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * 100, 30 - (x / top) * 26]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", `${line}L100,32L0,32Z`);
}
function setMetric(key, value, unit, sub) {
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
}

let lastMetrics = null, metricsFailures = 0, keyWarned = false;
let reqShowAll = false;   // the Monitor's request table: the last 12, or every one the server keeps (issue #35)
async function poll() {
  try {
    const r = await fetch(reqShowAll ? "metrics?requests=all" : "metrics", {headers: headers()});
    if (r.status === 401) {
      setPill("error", "API key needed");
      if (!keyWarned) { keyWarned = true; toast("warn", "API key needed", "This server needs a key: add it under About > Settings.", 6000); }
    } else if (r.ok) {
      lastMetrics = await r.json();
      metricsFailures = 0;
      render(lastMetrics);
    } else {
      throw new Error(`HTTP ${r.status}`);
    }
  } catch (e) {
    if (++metricsFailures === 3) setPill("error", "Server not reachable");
  }
  setTimeout(poll, 1000);
}

function setPill(state, text) {
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
}

function render(m) {
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
  const last = (m.requests || [])[0];
  // the header pill
  if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    setPill("reading", pct != null ? `Reading prompt · ${pct}%` : "Reading prompt");
  } else if (live.state === "generating") {
    setPill("generating", `Generating · ${fmt(live.tok_s, 1)} tok/s`);
  } else {
    setPill("idle", "Idle");
  }
  if (eng.models) syncModelSelect(eng.models);
  if (live.queued > 0) setPill("queued", `${live.queued} queued`);
  if (tab === "monitor") renderMonitor(live, hw, st, eng, h, last, m.requests || [], m.totals, m.requests_kept);
  if (tab === "monitor") renderConvCache(m.conversation_cache, m.requests);
  if (tab === "about") renderAbout(eng, hw, st);
}

// #596: the conversation cache - the prompt's state the engine keeps between requests (always), and the whole
// conversations it parks in RAM when "--conversation-cache-mib N" is in the run config's args (opt-in)
function since(t) {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? "just now" : s < 3600 ? `${fmt(s / 60)} min ago` : `${fmt(s / 3600, 1)} h ago`;
}
function renderConvCache(c, requests) {
  $("cc-card").hidden = !c;
  if (!c) return;                                  // an older server
  const pct = (a, b) => (b ? `${Math.min(100, (100 * a) / b)}%` : "0%");
  $("cc-bars").hidden = !c.enabled;
  if (c.enabled) {
    $("cc-slots-text").textContent = `${fmt(c.parked)} / ${fmt(c.slots)}`;
    $("cc-slots-bar").style.width = pct(c.parked, c.slots);
    const budget = c.budget_mib * 1048576;
    $("cc-mem-text").textContent = `${gb(c.bytes)} / ${gb(budget)} GB`;
    $("cc-mem-bar").style.width = pct(c.bytes, budget);
  }
  $("cc-sum").textContent = c.requests ? `${fmt(c.requests_reused)} of ${fmt(c.requests)} requests reused part of their prompt` : "";
  const share = c.prompt_tokens ? ` (${fmt((100 * c.reused_tokens) / c.prompt_tokens)}% of all prompt tokens)` : "";
  const event = c.last_event ? `${c.last_event === "parked" ? "Parked" : "Restored"} ${fmt(c.last_tokens)} tokens, ${since(c.last_at)}` : null;
  // the engine's own measurements (conversation_cache.ledger, requests[].restore_ms): null is "not known", never 0
  const le = c.ledger && c.ledger.last_event;
  const ledgerEvent = le ? `${le.event} (${le.tier}) ${fmt(le.tokens)} tokens, ${fmt(le.bytes / 1048576)} MB in ${fmt(le.ms, 1)} ms, ${since(le.at)}` : null;
  const r0 = requests && requests[0];
  const ms = (v, est) => (v == null ? "n/a" : `${est ? "~" : ""}${fmt(v)} ms`);
  const cost = r0 && r0.prompt_ms != null ? `restore ${ms(r0.restore_ms)} · read ${ms(r0.prefill_ms, true)} · saved ${ms(r0.saved_ms, true)}` : null;
  facts($("cc-facts"), [
    ["Last request cost", cost],
    ["Last cache event", ledgerEvent],
    ["Last request", c.last_prompt != null ? `${fmt(c.last_reused || 0)} of ${fmt(c.last_prompt)} prompt tokens reused` : null],
    ["Reused since start", c.requests ? `${fmt(c.reused_tokens)} tokens${share}` : null],
    ["Parked / restored", c.enabled ? `${fmt(c.parks)} / ${fmt(c.restores)}${c.evictions ? ` · ${fmt(c.evictions)} evicted` : ""}` : null],
    ["Last switch", c.enabled ? event : null],
  ]);
  $("cc-note").textContent = c.enabled
    ? "A request that continues a parked conversation gets its state back instead of reading it again; the oldest goes when the slots or the memory are full."
    : "The engine keeps the last conversation's state, so a follow-up reads only what is new. To keep several conversations (agents taking turns), add \"--conversation-cache-mib\", \"8192\" to the run config's args (docs/DETAILS.md).";
}

function renderTotals(t) {
  if (!t || !t.requests) return "";
  const since = new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"});
  const read = t.prompt_tokens - t.reused;
  const pSpeed = t.prompt_ms > 0 && read > 0 ? ` at ${fmt(read / (t.prompt_ms / 1000))} tok/s` : "";
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? ` at ${fmt(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
  return `Since ${since}: ${fmt(t.requests)} requests · ${fmt(read)} prompt tokens read${pSpeed} (${fmt(t.reused)} reused) · ` +
         `${fmt(t.output_tokens)} written${oSpeed}`;
}
function renderMonitor(live, hw, st, eng, h, last, requests, totals, kept) {
  // model state
  const on = live.queued > 0 ? "queued" : live.state;
  for (const b of document.querySelectorAll("#state-badges .st-badge")) b.classList.toggle("on", b.dataset.s === on || b.dataset.s === live.state);
  const prog = $("state-progress");
  let label = "Waiting for a request", detail = "", pct = 0;
  if (live.state === "reading") {
    label = "Reading prompt";
    prog.dataset.tone = "info";
    if (live.prompt_total) {
      pct = (100 * live.prompt_read) / live.prompt_total;
      detail = `${fmt(live.prompt_read)} / ${fmt(live.prompt_total)} tokens · ${fmt(pct)}%`;
    } else {
      detail = `${fmt(live.prompt_tokens)} tokens`;
    }
  } else if (live.state === "generating") {
    label = live.phase ? live.phase[0].toUpperCase() + live.phase.slice(1) : "Generating";
    delete prog.dataset.tone;
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = `${fmt(live.generated)} tokens · ${fmt(live.tok_s, 1)} tok/s`;
  } else if (live.state === "loading") {
    label = "Loading the model";
    prog.dataset.tone = "info";
    detail = "the engine is starting; requests wait until it is ready";
  } else if (live.state === "error") {
    label = "The model failed to load";
    prog.dataset.tone = "danger";
    detail = live.detail || "see the server log";
  } else if (live.state === "unloaded") {
    label = "No model loaded";
    delete prog.dataset.tone;
    detail = "load one in the Setup tab";
  } else if (last) {
    delete prog.dataset.tone;
    detail = `last: ${fmt(last.output_tokens)} tokens${last.decode_tok_s ? ` at ${fmt(last.decode_tok_s, 1)} tok/s` : ""}`;
  }
  $("state-label").textContent = label;
  $("state-detail").textContent = detail;
  $("state-bar").style.width = `${pct}%`;

  // the eight cards
  const speed = live.state === "generating" ? live.tok_s : last ? last.decode_tok_s : null;
  setMetric("speed", speed == null ? null : fmt(speed, 1), "t/s",
            live.state === "generating" ? "Decode now" : last ? "Decode last request" : "Decode");
  const prefill = live.state !== "idle" ? live.prefill_tok_s_mean
                : last && last.prompt_ms > 0 ? Math.max(0, last.prompt_tokens - (last.reused || 0)) / (last.prompt_ms / 1000) : null;
  setMetric("prefill", prefill == null ? null : fmt(prefill), "t/s",
            live.state === "reading" ? "Prefill now" : live.state === "generating" ? "Prefill this request" : last ? "Prefill last request" : "Prefill");
  spark("sp-speed", h.tok_s);
  spark("sp-prefill", h.prefill_tok_s_mean);
  // a model split across several cards (issue #112): the cards show their total / mean / hottest, and each card's own
  const per = (f) => (hw.gpus || []).map((g) => `GPU ${g.index} ${f(g)}`).join(" · ");
  const multi = (hw.gpus || []).length > 1;
  setMetric("gpu", hw.gpu_util == null ? null : fmt(hw.gpu_util), "%",
            multi ? per((g) => (g.util == null ? "–" : `${fmt(g.util)}%`)) : st.gpu_name || "");
  spark("sp-gpu", h.gpu_util, 100);
  const igpuUtil = hw.igpu_util != null ? hw.igpu_util : (hw.igpu ? hw.igpu.util : null);
  const igpuGtt = hw.igpu_gtt_used != null ? hw.igpu_gtt_used : (hw.igpu ? hw.igpu.gtt_used : null);
  const igpuGttTotal = hw.igpu_gtt_total != null ? hw.igpu_gtt_total : (hw.igpu ? hw.igpu.gtt_total : null);
  const igpuSub = st.igpu_name || (hw.igpu && hw.igpu.name) || "Integrated Graphics";
  setMetric("igpu", igpuUtil == null ? (igpuGtt != null ? gb(igpuGtt) : null) : fmt(igpuUtil), igpuUtil == null && igpuGtt != null ? "GB" : "%",
            igpuGttTotal ? `${gb(igpuGtt || 0)} / ${gb(igpuGttTotal, 0)} GB GTT · ${igpuSub}` : igpuSub);
  spark("sp-igpu", h.igpu_util, 100);
  setMetric("vram", hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
            multi ? per((g) => (g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`))
                  : eng.expert_slots ? `${fmt(eng.expert_slots)} experts cached` : "");
  spark("sp-vram", h.gpu_mem_used, hw.gpu_mem_total);
  setMetric("temp", hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C",
            multi ? per((g) => (g.temp == null ? "–" : `${fmt(g.temp)}°`)) : "");
  spark("sp-temp", h.gpu_temp, 90);
  setMetric("power", hw.gpu_power == null ? null : fmt(hw.gpu_power), "W", hw.gpu_power_limit ? `of ${fmt(hw.gpu_power_limit)} W limit` : "");
  spark("sp-power", h.gpu_power, hw.gpu_power_limit);
  const gen = hw.gpu_pcie_gen_max || hw.gpu_pcie_gen;
  let pcieSub = "";
  if (hw.gpu_pcie_rx_mb != null && hw.gpu_pcie_rx_mb > 0) {
    pcieSub = `to GPU ${fmt(hw.gpu_pcie_rx_mb, hw.gpu_pcie_rx_mb < 10 ? 1 : 0)} MB/s`;
    if (hw.gpu_pcie_bw_gb_s) pcieSub += ` · cap ~${hw.gpu_pcie_bw_gb_s} GB/s`;
  } else if (hw.gpu_pcie_bw_gb_s) {
    pcieSub = `Max ~${hw.gpu_pcie_bw_gb_s} GB/s`;
    if (hw.gpu_pcie_gen && gen && hw.gpu_pcie_gen < gen) pcieSub += ` · idle Gen${hw.gpu_pcie_gen}`;
  } else if (hw.gpu_pcie_gen && gen && hw.gpu_pcie_gen < gen) {
    pcieSub = `idle Gen${hw.gpu_pcie_gen}`;
  }
  setMetric("pcie", gen ? `Gen${gen}` : null, hw.gpu_pcie_width ? `x${hw.gpu_pcie_width}` : "", pcieSub);
  spark("sp-pcie", h.gpu_pcie_rx_mb);
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", st.threads ? `${st.cores ? `${st.cores} cores · ` : ""}${st.threads} threads` : "");
  spark("sp-cpu", h.cpu, 100);
  if (hw.disk_read_mb == null) {
    setMetric("disk", null, "", st.psutil ? "" : "needs psutil (setup installs it)");
  } else {
    const big = hw.disk_read_mb >= 1000;
    setMetric("disk", big ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), big ? "GB/s" : "MB/s",
              hw.disk_write_mb == null ? "" : `write ${fmt(hw.disk_write_mb, 1)} MB/s`);
  }
  spark("sp-disk", h.disk_read_mb);

  // context fill: the running request, else the last one
  const ctx = eng.max_context || eng.n_ctx_physical || 0;
  const virtCtx = eng.virtual_context_limit || eng.n_ctx_virtual || (ctx ? ctx * 32 : 1048576);
  const swaWin = eng.swa_window || 4096;
  let used = 0;
  if (live.state !== "idle") used = (live.prompt_tokens || 0) + (live.generated || 0);
  else if (last) used = (last.prompt_tokens || 0) + (last.output_tokens || 0);

  const rt = (window.lastMetrics && window.lastMetrics.runtime_stats) || {};
  const virtUsed = (rt.total_virtual_tokens || 0) + used;
  const virtRatio = rt.context_compression_ratio ? `${rt.context_compression_ratio}x` : `${Math.max(1, Math.round(virtCtx / Math.max(1, ctx)))}x`;

  const physFrac = ctx ? Math.min(1, used / ctx) : 0;
  const virtFrac = virtCtx ? Math.min(1, virtUsed / virtCtx) : 0;
  const swaFrac = swaWin ? Math.min(1, Math.min(used, swaWin) / swaWin) : 0;

  $("ctx-fill").setAttribute("stroke-dasharray", `${(235.6 * physFrac).toFixed(1)} 314.2`);
  $("ctx-fill").style.opacity = 235.6 * physFrac >= 3 ? "1" : "0";         // a near-zero arc would draw just its round cap
  $("ctx-pct").textContent = `${Math.round(physFrac * 100)}%`;
  $("ctx-sub").textContent = ctx ? `${kfmt(used)} / ${ctxfmt(ctx)}` : "–";

  if ($("ctx-summary-badge")) {
    $("ctx-summary-badge").textContent = ctx ? `${ctxfmt(ctx)} Phys · ${ctxfmt(virtCtx)} Virt (${virtRatio})` : "Dual Context Active";
  }
  if ($("ctx-phys-text") && $("ctx-phys-bar")) {
    $("ctx-phys-text").textContent = ctx ? `${kfmt(used)} / ${ctxfmt(ctx)} (${Math.round(physFrac * 100)}%)` : "–";
    $("ctx-phys-bar").style.width = `${physFrac * 100}%`;
  }
  if ($("ctx-virt-text") && $("ctx-virt-bar")) {
    $("ctx-virt-text").textContent = virtCtx ? `${kfmt(virtUsed)} / ${ctxfmt(virtCtx)} (${(virtFrac * 100).toFixed(1)}%)` : "–";
    $("ctx-virt-bar").style.width = `${Math.max(1, virtFrac * 100)}%`;
  }
  if ($("ctx-swa-text") && $("ctx-swa-bar")) {
    const swaUsed = Math.min(used, swaWin);
    $("ctx-swa-text").textContent = `${kfmt(swaUsed)} / ${kfmt(swaWin)} window (${Math.round(swaFrac * 100)}%)`;
    $("ctx-swa-bar").style.width = `${swaFrac * 100}%`;
  }

  const cacheBytes = (eng.expert_cache_mib || 0) * 1048576;
  $("slots-text").textContent = eng.expert_slots ? `${fmt(eng.expert_slots)} · ${gb(cacheBytes)} GB` : "–";
  $("slots-bar").style.width = hw.gpu_mem_total ? `${Math.min(100, (100 * cacheBytes) / hw.gpu_mem_total)}%` : "0%";

  const ramSlots = eng.expert_cache_ram_slots || (eng.expert_slots ? Math.max(0, 24576 - eng.expert_slots) : 24576);
  const ramCacheBytes = (eng.expert_cache_ram_mib || 0) * 1048576 || (ramSlots * 1382400);
  if ($("ram-slots-text") && $("ram-slots-bar")) {
    $("ram-slots-text").textContent = ramSlots ? `${fmt(ramSlots)} · ${gb(ramCacheBytes)} GB (RAM Cache)` : "–";
    $("ram-slots-bar").style.width = hw.ram_total ? `${Math.min(100, (100 * ramCacheBytes) / hw.ram_total)}%` : "0%";
  }

  if ($("igpu-text") && $("igpu-bar")) {
    const igpuMem = igpuGtt != null ? igpuGtt : (hw.igpu ? hw.igpu.mem_used : null);
    const igpuTot = igpuGttTotal != null ? igpuGttTotal : (hw.igpu ? hw.igpu.gtt_total : (hw.igpu ? hw.igpu.mem_total : null));
    if (igpuMem != null && igpuTot) {
      const igpuPct = Math.min(100, (100 * igpuMem) / igpuTot);
      $("igpu-text").textContent = `${gb(igpuMem)} / ${gb(igpuTot, 0)} GB GTT (${fmt(igpuPct, 0)}%)`;
      $("igpu-bar").style.width = `${igpuPct}%`;
      if ($("igpu-row")) $("igpu-row").hidden = false;
      if ($("igpu-progress")) $("igpu-progress").hidden = false;
    } else if (st.igpu_name || (hw.igpu && hw.igpu.name)) {
      $("igpu-text").textContent = `${st.igpu_name || hw.igpu.name} (Active)`;
      $("igpu-bar").style.width = igpuUtil != null ? `${Math.min(100, igpuUtil)}%` : "0%";
      if ($("igpu-row")) $("igpu-row").hidden = false;
      if ($("igpu-progress")) $("igpu-progress").hidden = false;
    } else {
      if ($("igpu-row")) $("igpu-row").hidden = true;
      if ($("igpu-progress")) $("igpu-progress").hidden = true;
    }
  }

  $("ram-text").textContent = hw.ram_total ? `${gb(hw.ram_used)} / ${gb(hw.ram_total, 0)} GB` : "–";
  const ramPct = hw.ram_total ? (100 * hw.ram_used) / hw.ram_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;
  $("temp-text").textContent = hw.gpu_temp == null ? "–" : `${fmt(hw.gpu_temp)} °C`;
  $("temp-bar").style.width = hw.gpu_temp == null ? "0%" : `${Math.min(100, hw.gpu_temp)}%`;

  // recent requests
  const body = $("req-body");
  if (!requests.length) {
    body.innerHTML = `<tr><td colspan="8" class="muted">No requests yet</td></tr>`;
  } else {
    const badge = {stop: ["", "Done"], length: ["", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                   disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"]};
    body.innerHTML = requests.slice(0, reqShowAll ? requests.length : 12).map((r, idx) => {
      const [cls, text] = badge[r.finish] || ["", r.finish || "–"];
      const t = new Date(r.time * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
      const proj = r.projection == null ? "" : ` <span class="st-badge${r.projection ? " st-badge--reading" : ""}" title="experimental speed projection ${r.projection ? "on" : "off"}">${r.projection ? "ESP" : "stock"}</span>`;
      // #588: the VRAM share; the PCIe share (--pcie-frac) beside it when there is one
      const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%` +
        (r.pcie_share ? ` <span class="muted" title="routed experts the GPU read over PCIe (--pcie-frac) or another GPU computed">+${(r.pcie_share * 100).toFixed(1)}% PCIe</span>` : "");
      const isSelected = selectedMonReq && (selectedMonReq === r.id || selectedMonReq === String(r.time));
      return `<tr class="clickable${isSelected ? " selected" : ""}" data-req-idx="${idx}"><td>${esc(t)}</td><td><span class="st-badge ${cls}">${esc(text)}</span>${proj}</td><td class="num">${fmt(r.prompt_tokens)}</td>
        <td class="num">${fmt(r.reused)}</td><td class="num">${fmt(r.output_tokens)}</td><td class="num">${fmt(r.decode_tok_s, 1)}</td>
        <td class="num">${hit}</td><td class="num">${fmt(r.duration_s, 1)} s</td></tr>`;
    }).join("");

    body.querySelectorAll("tr.clickable").forEach((tr) => {
      const idx = Number(tr.dataset.reqIdx);
      tr.onclick = () => {
        selectMonitorRequest(requests[idx]);
        body.querySelectorAll("tr.clickable").forEach(t => t.classList.remove("selected"));
        tr.classList.add("selected");
      };
    });
  }
  const all = $("req-all");
  kept = kept == null ? requests.length : kept;
  all.hidden = kept <= 12;
  all.textContent = reqShowAll ? "Show fewer" : `Show all (${kept})`;
  $("req-wrap").classList.toggle("all", reqShowAll);
  $("req-totals").textContent = renderTotals(totals);
}

let selectedMonReq = null;
let monDetail = null;
let monPane = "input";

async function selectMonitorRequest(reqObj) {
  if (!reqObj) return;
  selectedMonReq = reqObj.id || String(reqObj.time);
  if ($("mon-detail-card")) $("mon-detail-card").hidden = false;

  if (reqObj.id) {
    try {
      const res = await fetch(`api/requests?id=${encodeURIComponent(reqObj.id)}`, {headers: headers(false)});
      if (res.ok) {
        monDetail = await res.json();
      } else {
        monDetail = reqObj;
      }
    } catch (e) {
      monDetail = reqObj;
    }
  } else {
    monDetail = reqObj;
  }
  renderMonDetail();
}

function renderMonDetail() {
  if (!monDetail) return;
  const d = monDetail;
  if ($("mon-detail-title")) {
    $("mon-detail-title").textContent = d.id ? `Request ${d.id} · ${d.path || "POST /v1/chat/completions"}` : `Request at ${new Date((d.time || Date.now()/1000)*1000).toLocaleTimeString()}`;
  }
  if ($("mon-detail-time")) {
    $("mon-detail-time").textContent = d.started_at ? new Date(d.started_at * 1000).toLocaleString() : (d.time ? new Date(d.time * 1000).toLocaleString() : "");
  }

  const st = $("mon-detail-status");
  if (st) {
    st.textContent = d.state || d.finish || "completed";
    st.className = `st-badge ${d.state === "error" || d.finish === "error" ? "st-badge--error" : "st-badge--reading"}`;
  }

  if ($("mon-fact-wall")) $("mon-fact-wall").textContent = d.wallclock_s ? `${fmt(d.wallclock_s, 2)} s` : (d.duration_s ? `${fmt(d.duration_s, 2)} s` : "–");
  if ($("mon-fact-first")) $("mon-fact-first").textContent = d.first_token_s ? `${fmt(d.first_token_s, 2)} s` : "–";
  if ($("mon-fact-tokens")) $("mon-fact-tokens").textContent = `${fmt(d.prompt_tokens || (d.usage && d.usage.prompt_tokens) || 0)} in / ${fmt(d.output_tokens || (d.usage && d.usage.completion_tokens) || 0)} out`;
  if ($("mon-fact-speed")) $("mon-fact-speed").textContent = d.decode_tok_s ? `${fmt(d.decode_tok_s, 1)} t/s` : "–";

  renderMonPaneContent();
}

// The debug events of a request: what the server, the runtime and the engine did for it, oldest first.
function debugLines(d, filter) {
  return (d.debug || []).filter((e) => !filter || filter(e))
    .map((e) => `${String(Math.round(e.ms)).padStart(8)} ms  ${String(e.source).padEnd(9)} ${e.msg}`);
}

function renderMonPaneContent() {
  if (!monDetail) return;
  const d = monDetail;
  let text = "", note = "";
  if (monPane === "input") {
    text = d.input || JSON.stringify(d.messages || d.prompt || "No raw input captured", null, 2);
    note = d.input_truncated ? "Input payload (truncated)" : "Input payload";
  } else if (monPane === "output") {
    text = d.output || d.text || "(no output content)";
    note = "Generated output";
  } else if (monPane === "reasoning") {
    text = d.reasoning || "(no reasoning content generated)";
    note = "Reasoning trajectory (<think>)";
  } else if (monPane === "debug") {
    const lines = debugLines(d);
    text = lines.length ? lines.join("\n") : "No debug events were recorded for this request.";
    note = `Debug trace: ${lines.length} events (server, rtclient, engine, runtime), time since the request arrived`;
  } else if (monPane === "vctx") {
    const lines = debugLines(d, (e) => e.source === "runtime" && /\b(context|kv|retrieval|memory)\b/.test(e.msg))
      .concat(debugLines(d, (e) => /CUT FROM THE FRONT|prompt built|reading the prompt|RESUME|REUSED/.test(e.msg)));
    text = lines.join("\n") || "The runtime recorded no context or KV events for this request.";
    note = "What the runtime and the engine did with the context for this request (from the debug trace, not an estimate)";
  } else if (monPane === "tool") {
    text = debugLines(d, (e) => e.source === "runtime" && /\b(recovery|verif|claim|tool_output|math|hedge)/.test(e.msg)).join("\n")
      || "The runtime recorded no verification or tool events for this request.";
    note = "Verification and tool events of the runtime for this request (from the debug trace)";
  } else if (monPane === "scope") {
    text = JSON.stringify(d.scope || {session_id: getSessionId(), tenant_id: "default"}, null, 2);
    note = "Multi-tenant isolation & security scope";
  } else if (monPane === "response") {
    text = d.response || JSON.stringify(d, null, 2);
    note = "Complete API response payload";
  }

  if ($("mon-content-note")) $("mon-content-note").textContent = note;
  if ($("mon-content")) $("mon-content").textContent = text;
}

function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="Copy">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
// INFO cvec=project:4-44[:singleL] | add:A-B | 0
function projectionText(c) {
  if (!c || c === "0" || c === 0) return null;
  const [mode, range, single] = String(c).split(":");
  const [a, b] = (range || "").split("-");
  return `${mode === "project" ? "Projection" : "Additive"} control vector on layers ${a}–${b}` +
         `${single ? ` (layer ${single.replace("single", "")}'s direction)` : ""}. Per chat in Sampling. Its package ` +
         "describes the vector as a refusal-direction projection; measure the speed yourself";
}
function renderAbout(eng, hw, st) {
  const kv = {int8: "8-bit", q4_0: "4-bit (Hadamard-rotated)", fp16: "16-bit"}[eng.kv] || eng.kv;
  const igpuDetail = st.igpu_name
    ? `${st.igpu_name}${hw.igpu && hw.igpu.gtt_total ? `, ${gb(hw.igpu.gtt_total, 0)} GB GTT` : (hw.igpu_gtt_total ? `, ${gb(hw.igpu_gtt_total, 0)} GB GTT` : "")}`
    : (hw.igpu && hw.igpu.name ? `${hw.igpu.name}${hw.igpu.gtt_total ? `, ${gb(hw.igpu.gtt_total, 0)} GB GTT` : ""}` : null);

  facts($("facts-engine"), [
    ["Model", eng.model],
    ["Engine", eng.version ? `v${eng.version}` : "built from source"],
    ["Context", eng.max_context ? `${fmt(eng.max_context)} tokens` : null],
    ["KV cache", kv ? `${kv}${eng.kv_resident ? `, streamed: ${fmt(eng.kv_resident)} positions per layer in VRAM, the rest in RAM` : ", all in VRAM"}` : null],
    ["Experts in VRAM", eng.expert_slots ? `${fmt(eng.expert_slots)} (${gb((eng.expert_cache_mib || 0) * 1048576)} GB)` : null],
    ["Speculation", eng.spec ? `MTP drafts up to ${Math.max(0, (eng.mtp_max || eng.spec) - 1)} tokens${eng.lookup ? ", prompt lookup on" : ""}` : null],
    ["Images", eng.images ? "on" : "off"],
    ["Experimental speed projection", projectionText(eng.cvec)],
  ]);
  facts($("facts-hw"), [
    [igpuDetail ? "dGPU" : "GPU", st.gpu_name ? `${st.gpu_name}${hw.gpu_mem_total ? `, ${gb(hw.gpu_mem_total, 0)} GB` : ""}` : "not readable (NVML)"],
    ["iGPU", igpuDetail],
    ["Compute Devices", st.compute_devices && st.compute_devices.length > 1 ? st.compute_devices.join(" · ") : null],
    ["CPU", st.cpu_name ? `${st.cpu_name}${st.threads ? `, ${st.threads} threads` : ""}` : null],
    ["RAM", hw.ram_total ? `${gb(hw.ram_total, 0)} GB` : null],
  ]);
  const base = location.origin;
  const key = secret.get();
  const claudeDefaultCmd = getClaudeCommand("bash", base, key);
  facts($("facts-api"), [
    ["OpenAI base URL", `${base}/v1`, true],
    ["Anthropic base URL", base, true],
    ["Model name", eng.model, true],
    ["Claude Code (Bash)", claudeDefaultCmd, true],
  ]);
  updateClaudeCodeBlock();
}

let activeClaudeOS = "bash";

function getClaudeCommand(os, base, key) {
  const k = key || "strata";
  switch (os) {
    case "npx":
      return `ANTHROPIC_BASE_URL="${base}" ANTHROPIC_API_KEY="${k}" npx @anthropic-ai/claude-code`;
    case "pwsh":
      return `$env:ANTHROPIC_BASE_URL="${base}"; $env:ANTHROPIC_API_KEY="${k}"; claude`;
    case "cmd":
      return `set ANTHROPIC_BASE_URL=${base}&& set ANTHROPIC_API_KEY=${k}&& claude`;
    case "bash":
    default:
      return `ANTHROPIC_BASE_URL="${base}" ANTHROPIC_API_KEY="${k}" claude`;
  }
}

function updateClaudeCodeBlock() {
  const base = location.origin;
  const key = secret.get();
  const cmd = getClaudeCommand(activeClaudeOS, base, key);
  const el = $("claude-code-cmd");
  if (el) el.textContent = cmd;
}

document.querySelectorAll("#claude-os-tabs .st-pill").forEach((pill) => {
  pill.onclick = () => {
    activeClaudeOS = pill.dataset.os || "bash";
    document.querySelectorAll("#claude-os-tabs .st-pill").forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
    updateClaudeCodeBlock();
  };
});

if ($("claude-copy-btn")) {
  $("claude-copy-btn").onclick = () => {
    const el = $("claude-code-cmd");
    if (el) copyText(el.textContent, $("claude-copy-btn"));
  };
}

document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (b) copyText(b.dataset.copy, b);
});
$("req-all").addEventListener("click", () => { reqShowAll = !reqShowAll; if (lastMetrics) render(lastMetrics); });

// ------------------------------------------------------------------ Model settings (GET / POST /config, #564)
// A few documented keys of the run config (strata-<model>.json), for every client, from the next start on.  The
// server lists them, checks every value and keeps every other key of the file as it is.
let cfgKeys = [];
async function loadConfig() {
  let r;
  try { r = await fetch("config", {headers: headers()}); } catch (e) { return; }
  if (!r.ok) { $("cfg-card").hidden = true; return; }       // no run config, an older server, or no key yet
  const c = await r.json();
  cfgKeys = c.keys || [];
  $("cfg-file").textContent = c.file || "";
  $("cfg-form").innerHTML = cfgKeys.map((k, i) => {
    const id = `cfg-${i}`, v = k.value;
    let input;
    if (k.kind === "bool" || k.kind === "enum") {
      const opts = k.kind === "bool" ? [["true", "on"], ["false", "off"]] : k.choices.map((x) => [x, x]);
      const cur = v == null ? "" : String(v);
      input = `<select class="st-input" id="${id}"><option value=""${cur === "" ? " selected" : ""}>default</option>` +
        opts.map(([val, text]) => `<option value="${esc(val)}"${cur === val ? " selected" : ""}>${esc(text)}</option>`).join("") + `</select>`;
    } else {
      const text = v == null ? "" : Array.isArray(v) ? v.join(", ") : String(v);
      input = `<input class="st-input" id="${id}" ${k.kind === "number" ? 'type="number" step="any" min="0"' : 'type="text"'} ` +
        `value="${esc(text)}" placeholder="default" autocomplete="off">`;
    }
    return `<label for="${id}" title="${esc(k.help)}">${esc(k.help)}<code>${esc(k.key)}</code></label>${input}`;
  }).join("");
  $("cfg-card").hidden = false;
  renderTune(c.tuned);
}
// The last `strata tune`: the candidates it measured and the one it chose (marked), from the run config's "tuned" record.
function renderTune(t) {
  const card = $("tune-card");
  if (!t || !Array.isArray(t.results) || !t.results.length) { card.hidden = true; return; }
  const fix = (n) => (typeof n === "number" ? n.toFixed(1) : "-");
  $("tune-when").textContent = t.at ? new Date(t.at).toLocaleString() : "";
  $("tune-hw").textContent = Array.isArray(t.hardware) ? t.hardware.join(" · ") : "";
  $("tune-rows").innerHTML = t.results.map((r) => {
    const chosen = r.name === t.chosen;
    return `<tr${chosen ? ' class="tune-chosen"' : ""}><td>${esc(r.name || "")}${chosen ? " <strong>(in use)</strong>" : ""}</td>` +
      `<td class="num">${fix(r.decode_tok_s)}</td><td class="num">${fix(r.prefill_tok_s)}</td></tr>`;
  }).join("");
  card.hidden = false;
}
function configValue(k, el) {
  const s = el.value.trim();
  if (s === "") return null;
  if (k.kind === "bool") return s === "true";
  if (k.kind === "number") return Number(s);
  return s;                                         // enum, or names (the server splits them at commas)
}
$("cfg-save").addEventListener("click", async () => {
  const set = {};
  cfgKeys.forEach((k, i) => {
    const v = configValue(k, $(`cfg-${i}`));
    const old = Array.isArray(k.value) ? k.value.join(", ") : k.value;
    if (JSON.stringify(v) !== JSON.stringify(old ?? null)) set[k.key] = v;
  });
  if (!Object.keys(set).length) { $("cfg-msg").textContent = "Nothing changed."; return; }
  try {
    const r = await fetch("config", {method: "POST", headers: headers(true), body: JSON.stringify({set})});
    const b = await r.json();
    if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
    $("cfg-msg").textContent = b.changed.length
      ? `Saved (${b.changed.join(", ")}); the earlier file is ${b.file}.bak. Start the model again to use it.` : "Nothing changed.";
    loadConfig();
  } catch (e) {
    $("cfg-msg").textContent = "";
    toast("error", "Not saved", String(e.message || e), 6000);
  }
});

// ------------------------------------------------------------------ Markdown (escaped first, then formatted)
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code class="inline">${esc(codes[+i])}</code>`);
}
function codeBlock(lang, code) {
  return `<div class="st-code"><div class="st-code__head"><span>${esc(lang || "code")}</span>` +
    `<button class="st-btn st-btn--icon" data-code-copy aria-label="Copy code">${icon("copy")}</button></div>` +
    `<pre><code>${esc(code)}</code></pre></div>`;
}
function blocks(text) {
  const out = [], lines = text.split("\n");
  let para = [], list = null;
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
  const flushList = () => { if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    let m;
    if (!l.trim()) { flushPara(); flushList(); continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); flushList(); out.push(`<${m[1].length <= 2 ? "h3" : "h4"}>${inline(m[2])}</${m[1].length <= 2 ? "h3" : "h4"}>`); continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) { flushPara(); flushList(); out.push("<hr>"); continue; }
    if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); flushList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      flushPara(); flushList();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      let html = `<table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
      i--;
      out.push(html + "</tbody></table>");
      continue;
    }
    if ((m = l.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/))) {
      flushPara();
      const tag = m[1] ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = {tag, items: []}; }
      list.items.push(m[2]);
      continue;
    }
    if (list && /^\s{2,}\S/.test(l)) { list.items[list.items.length - 1] += " " + l.trim(); continue; }
    flushList();
    para.push(l);
  }
  flushPara(); flushList();
  return out.join("");
}
function markdown(text) {
  let html = "", rest = text;
  for (;;) {
    const m = rest.match(/(^|\n)```([^\n`]*)\n/);
    if (!m) { html += blocks(rest); break; }
    html += blocks(rest.slice(0, m.index));
    rest = rest.slice(m.index + m[0].length);
    const end = rest.match(/(^|\n)```[ \t]*(\n|$)/);
    if (!end) { html += codeBlock(m[2].trim(), rest); break; }         // still streaming
    html += codeBlock(m[2].trim(), rest.slice(0, end.index));
    rest = rest.slice(end.index + end[0].length);
  }
  return html;
}

// ------------------------------------------------------------------ Chat & Multi-Session
const DEFAULTS = {thinking: "high", temperature: 0.6, top_p: 0.95, top_k: 20, max: "", seed: "", show: true, esp: true, abliterate: false, abliteration_strength: 1.0};
let settings = {...DEFAULTS, ...store.get("sampling", {})};
let sessions = [];
let activeSessionId = "";
let sessionSearchQuery = "";
let messages = [];
let attachments = [];                 // {name, url}
let busy = null;                      // {controller, msg}

function timeSince(t) {
  if (!t) return "";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  if (s < 604800) return `${Math.floor(s / 86400)}d ago`;
  return new Date(t).toLocaleDateString([], {month: "short", day: "numeric"});
}

function initSessions() {
  const storedList = store.get("sessions_list", null);
  if (Array.isArray(storedList) && storedList.length > 0) {
    sessions = storedList;
    activeSessionId = store.get("active_session_id", sessions[0].id);
    if (!sessions.some((s) => s.id === activeSessionId)) {
      activeSessionId = sessions[0].id;
    }
  } else {
    // Migration: check legacy single-session chat store
    const legacy = store.get("chat", []);
    const firstUser = legacy.find((m) => m.role === "user");
    const title = firstUser && firstUser.text ? (firstUser.text.slice(0, 32) + (firstUser.text.length > 32 ? "…" : "")) : "New Chat";
    const initialId = store.get("session_id", "") || ("sess-" + randomId());
    sessions = [{
      id: initialId,
      title: title,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: legacy,
    }];
    activeSessionId = initialId;
  }

  const active = sessions.find((s) => s.id === activeSessionId) || sessions[0];
  messages = active.messages || [];
  saveSessions(true);

  if (store.get("sidebar_collapsed", false)) {
    const sb = $("chat-sidebar");
    if (sb) sb.classList.add("collapsed");
  }
}

function saveSessions(renderList = true) {
  const active = sessions.find((s) => s.id === activeSessionId);
  if (active) {
    active.messages = messages.map((m) => ({
      ...m,
      images: (m.images || []).map((i) => ({name: i.name})),
      files: (m.files || []).map((f) => ({name: f.name})),
    }));
  }
  store.set("sessions_list", sessions);
  store.set("active_session_id", activeSessionId);
  store.set("session_id", activeSessionId);
  store.set("chat", messages.map((m) => ({
    ...m,
    images: (m.images || []).map((i) => ({name: i.name})),
    files: (m.files || []).map((f) => ({name: f.name})),
  })));

  if ($("active-session-title")) {
    $("active-session-title").textContent = (active && active.title) ? active.title : "New Chat";
  }
  if ($("session-count-label")) {
    $("session-count-label").textContent = `${sessions.length} chat${sessions.length === 1 ? "" : "s"}`;
  }
  if (renderList) renderSessionList();
}

function saveChat() {
  saveSessions(true);
}

function renderSessionList() {
  const listEl = $("session-list");
  if (!listEl) return;
  const query = (sessionSearchQuery || "").toLowerCase().trim();
  const filtered = sessions.filter((s) => {
    if (!query) return true;
    if (s.title && s.title.toLowerCase().includes(query)) return true;
    if (s.messages && s.messages.some((m) => m.text && m.text.toLowerCase().includes(query))) return true;
    return false;
  });

  if (filtered.length === 0) {
    listEl.innerHTML = `<div class="muted small" style="padding: 16px 8px; text-align: center;">${query ? "No matching chats" : "No chats yet"}</div>`;
    return;
  }

  listEl.innerHTML = filtered.map((s) => {
    const isActive = s.id === activeSessionId;
    const tStr = s.updatedAt ? timeSince(s.updatedAt) : "";
    return `
      <div class="session-item ${isActive ? "active" : ""}" data-id="${esc(s.id)}" role="button" tabindex="0">
        <svg class="st-icon st-icon--sm" style="flex:none"><use href="${SPRITE}#i-chat"/></svg>
        <span class="session-item__title" title="${esc(s.title || "New Chat")}">${esc(s.title || "New Chat")}</span>
        ${tStr ? `<span class="session-item__time">${esc(tStr)}</span>` : ""}
        <div class="session-item__actions">
          <button type="button" class="session-item__btn btn-edit" data-act="rename" data-id="${esc(s.id)}" title="Rename chat" aria-label="Rename chat">
            <svg class="st-icon st-icon--sm"><use href="${SPRITE}#i-settings"/></svg>
          </button>
          <button type="button" class="session-item__btn btn-delete" data-act="delete" data-id="${esc(s.id)}" title="Delete chat" aria-label="Delete chat">
            <svg class="st-icon st-icon--sm"><use href="${SPRITE}#i-trash"/></svg>
          </button>
        </div>
      </div>
    `;
  }).join("");
}

function createNewSession() {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  const active = sessions.find((s) => s.id === activeSessionId);
  if (active && (!messages || messages.length === 0)) {
    $("input").focus();
    return;
  }
  const newSess = {
    id: "sess-" + randomId(),
    title: "New Chat",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  };
  sessions.unshift(newSess);
  activeSessionId = newSess.id;
  messages = [];
  attachments = [];
  renderAttachments();
  saveSessions(true);
  renderChat();
  $("input").value = "";
  autosize();
  $("input").focus();
}

function switchSession(id) {
  if (id === activeSessionId) return;
  if (busy) { toast("warn", "Still writing", "Stop the current generation first."); return; }
  const target = sessions.find((s) => s.id === id);
  if (!target) return;
  activeSessionId = target.id;
  messages = target.messages || [];
  attachments = [];
  renderAttachments();
  saveSessions(true);
  renderChat();
  $("input").focus();
}

function deleteSession(id, e) {
  if (e) e.stopPropagation();
  if (busy && activeSessionId === id) {
    toast("warn", "Chat in progress", "Stop generation before deleting.");
    return;
  }
  sessions = sessions.filter((s) => s.id !== id);
  if (sessions.length === 0) {
    const fresh = {
      id: "sess-" + randomId(),
      title: "New Chat",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    };
    sessions = [fresh];
    activeSessionId = fresh.id;
    messages = [];
  } else if (activeSessionId === id) {
    activeSessionId = sessions[0].id;
    messages = sessions[0].messages || [];
  }
  saveSessions(true);
  renderChat();
  // Server-side cleanup notification (non-blocking)
  fetch(`api/strata/sessions/delete?id=${encodeURIComponent(id)}`, { method: "POST", headers: headers() }).catch(() => {});
  toast("info", "Chat deleted");
}

function renameSession(id) {
  const s = sessions.find((x) => x.id === id);
  if (!s) return;
  const currentTitle = s.title || "New Chat";
  const name = prompt("Rename chat:", currentTitle);
  if (name !== null && name.trim()) {
    s.title = name.trim();
    s.updatedAt = Date.now();
    saveSessions(true);
    toast("success", "Chat renamed");
  }
}

function forkSession() {
  const active = sessions.find((s) => s.id === activeSessionId);
  if (!active || !messages.length) {
    toast("info", "Nothing to fork", "Start a conversation first.");
    return;
  }
  const forked = {
    id: "sess-" + randomId(),
    title: `${active.title || "Chat"} (Fork)`,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: JSON.parse(JSON.stringify(messages)),
  };
  sessions.unshift(forked);
  activeSessionId = forked.id;
  messages = forked.messages;
  saveSessions(true);
  renderChat();
  toast("success", "Chat Forked", "Created a new branch from this conversation.");
}

function clearAllSessions() {
  if (!confirm("Are you sure you want to clear all chat history?")) return;
  const fresh = {
    id: "sess-" + randomId(),
    title: "New Chat",
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  };
  sessions = [fresh];
  activeSessionId = fresh.id;
  messages = [];
  saveSessions(true);
  renderChat();
  toast("info", "All chats cleared");
}
function timeStr(t) { return new Date(t).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }

function msgEl(m, i) {
  const el = document.createElement("div");
  el.className = `st-msg st-msg--${m.role}`;
  el.dataset.i = i;
  if (m.role === "user") {
    if (m.files && m.files.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const f of m.files) {
        const c = document.createElement("span");
        c.className = "chip";
        c.innerHTML = icon("attach", "st-icon st-icon--sm");
        c.append(f.name);
        wrap.appendChild(c);
      }
      el.appendChild(wrap);
    }
    if (m.images && m.images.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const im of m.images) {
        if (im.url) { const img = document.createElement("img"); img.src = im.url; img.alt = im.name || "image"; wrap.appendChild(img); }
        else { const c = document.createElement("span"); c.className = "chip"; c.innerHTML = icon("image", "st-icon st-icon--sm"); c.append(im.name || "image"); wrap.appendChild(c); }
      }
      el.appendChild(wrap);
    }
    const b = document.createElement("div");
    b.className = "st-bubble";
    b.textContent = m.text;
    el.appendChild(b);
    const meta = document.createElement("div");
    meta.className = "st-msg__meta";
    meta.textContent = `You · ${timeStr(m.time)}`;
    el.appendChild(meta);
  } else {
    el.innerHTML = `<details class="st-collapse think" hidden><summary>${icon("thinking", "st-icon st-icon--sm")}<span class="think-title"></span>` +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body thinking"></div></details>` +
      `<div class="st-bubble"></div><div class="st-msg__meta"><span class="meta-text"></span>` +
      `<button class="st-btn st-btn--icon" data-msg-copy aria-label="Copy the answer" title="Copy">${icon("copy")}</button></div>`;
    updateAssistant(el, m, false);
  }
  return el;
}
// One tool call in the answer: a compact block (name, state, a one-line preview) that opens to the arguments and
// the result as the model read it.  Its body is built only while open: a result can be 20,000 characters.
const TOOL_STATE = {writing: ["st-badge--reading", "Writing"], running: ["st-badge--generating", "Running"], done: ["", "Done"],
                    error: ["st-badge--error", "Error"], skipped: ["st-badge--queued", "Not run"]};
function toolHtml(t, k) {
  const [cls, label] = TOOL_STATE[t.state] || ["", t.state];
  const args = t.arguments == null ? "" : JSON.stringify(t.arguments, null, 2);
  const preview = t.result != null ? t.result : args.replace(/\s+/g, " ");
  let body = "";
  if (t.open) {
    body = `<div class="tool-call__label">Arguments</div><pre class="tool-call__pre">${esc(args || "(being written)")}</pre>`;
    if (t.result != null) {
      body += `<div class="tool-call__label">${t.ok ? "Result" : "Error"}${t.chars ? ` · ${fmt(t.chars)} characters` : ""}` +
              `${t.truncated ? ", cut for the model" : ""}</div><pre class="tool-call__pre">${esc(t.result)}</pre>`;
    }
  }
  return `<details class="st-collapse tool-call" data-tool="${k}" data-state="${esc(t.state)}"${t.open ? " open" : ""}>` +
    `<summary>${icon("tool", "st-icon st-icon--sm")}<span class="tool-call__name" title="${esc(t.name || "")}">${esc(t.tool || t.name || "tool")}</span>` +
    (t.server ? `<span class="muted small">${esc(t.server)}</span>` : "") +
    `<span class="tool-call__preview muted">${esc(preview.slice(0, 200))}</span>` +
    `<span class="st-badge ${cls}">${esc(label)}</span>${t.ms != null && t.state !== "skipped" ? `<span class="muted small">${fmt(t.ms / 1000, 1)} s</span>` : ""}` +
    `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body">${body}</div></details>`;
}
// the answer's text with the tool blocks where the model called them
function answerHtml(m) {
  if (!m.tools || !m.tools.length) return markdown(m.text || "");
  let html = "", pos = 0;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), m.text.length);
    if (at > pos) html += markdown(m.text.slice(pos, at));
    pos = at;
    html += toolHtml(t, k);
  });
  return html + markdown(m.text.slice(pos));
}
function updateAssistant(el, m, streaming) {
  const det = el.querySelector("details.think");
  if (m.reasoning) {
    det.hidden = false;
    const thinkingNow = streaming && !m.text;
    el.querySelector(".think-title").textContent = thinkingNow ? "Thinking…" :
      m.thinkSecs != null ? `Thought for ${fmt(m.thinkSecs, 1)} s` : "Thoughts";
    const body = el.querySelector(".thinking");
    if (det.open || thinkingNow) body.textContent = m.reasoning;
    else body.dataset.pending = "1";
    // open while it streams (if wanted), closed once the answer starts - unless the user toggled it themselves
    if (thinkingNow && settings.show && !det.dataset.touched && !det.open) { det._auto = true; det.open = true; }
    if (!thinkingNow && det.open && !det.dataset.touched) { det._auto = true; det.open = false; }
  }
  const bubble = el.querySelector(".st-bubble");
  if (m.error) {
    bubble.innerHTML = `<div class="msg-error"></div>`;
    bubble.firstChild.textContent = m.error;
  } else if (!m.text && streaming && !(m.tools && m.tools.length)) {
    bubble.innerHTML = m.reasoning ? `<span class="muted cursor">Writing</span>` : `<span class="cursor"></span>`;
  } else {
    bubble.innerHTML = answerHtml(m);
    if (streaming) bubble.classList.add("cursor"); else bubble.classList.remove("cursor");
  }
  el.querySelector(".meta-text").textContent = m.meta || (streaming ? "" : m.stopped ? "Stopped" : "");
  el.querySelector("[data-msg-copy]").hidden = streaming || !m.text;
}
function renderChat() {
  const chat = $("chat");
  chat.querySelectorAll(".st-msg").forEach((e) => e.remove());
  $("chat-empty").hidden = messages.length > 0;
  messages.forEach((m, i) => chat.appendChild(msgEl(m, i)));
  scrollDown(true);
}
function nearBottom() { const s = $("chat-scroll"); return s.scrollHeight - s.scrollTop - s.clientHeight < 120; }
function scrollDown(force) { const s = $("chat-scroll"); if (force || nearBottom()) s.scrollTop = s.scrollHeight; }

$("chat").addEventListener("click", (e) => {
  const cc = e.target.closest("[data-code-copy]");
  if (cc) { copyText(cc.closest(".st-code").querySelector("pre").textContent, cc); return; }
  const mc = e.target.closest("[data-msg-copy]");
  if (mc) { const i = +mc.closest(".st-msg").dataset.i; copyText(messages[i].text, mc); return; }
  // a tool block: its open state lives in the message (the answer is rebuilt while it streams), so the click sets it
  const sum = e.target.closest(".tool-call > summary");
  if (sum) {
    e.preventDefault();
    const el = sum.closest(".st-msg"), m = messages[+el.dataset.i], t = m && m.tools && m.tools[+sum.parentElement.dataset.tool];
    if (!t) return;
    t.open = !t.open;
    updateAssistant(el, m, !!busy && busy.msg === m);
  }
});
$("chat").addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.tagName !== "DETAILS" || !d.classList.contains("think")) return;
  if (d._auto) { d._auto = false; return; }          // our own open/close, not the user's
  d.dataset.touched = "1";
  const body = d.querySelector(".thinking");
  if (d.open && body.dataset.pending) { body.textContent = messages[+d.closest(".st-msg").dataset.i].reasoning; delete body.dataset.pending; }
}, true);

function apiMessages() {
  const out = [];
  for (const m of messages) {
    if (m.role === "user") {
      const imgs = (m.images || []).filter((i) => i.url);
      const text = userText(m);
      out.push({role: "user", content: imgs.length ? [{type: "text", text},
        ...imgs.map((i) => ({type: "image_url", image_url: {url: i.url}}))] : text});
    } else if (!m.error) {
      out.push(...assistantMessages(m));
    }
  }
  return out;
}
// An answer that used tools goes back as the model wrote it: per round the text before the calls, the calls and
// their results (as the model read them), then the rest - so the next question can build on what the tools found.
function assistantMessages(m) {
  const ran = (m.tools || []).filter((t) => t.round != null && t.result != null && t.state !== "skipped");
  if (!ran.length) return m.text ? [{role: "assistant", content: m.text}] : [];
  const out = [];
  let pos = 0;
  for (const r of [...new Set(ran.map((t) => t.round))]) {
    const calls = ran.filter((t) => t.round === r);
    const at = Math.min(Math.max(pos, calls[0].at || 0), m.text.length);
    out.push({role: "assistant", content: m.text.slice(pos, at).trim(),
              tool_calls: calls.map((t) => ({id: t.id, type: "function", function: {name: t.name, arguments: JSON.stringify(t.arguments || {})}}))});
    for (const t of calls) out.push({role: "tool", tool_call_id: t.id, content: t.result});
    pos = at;
  }
  const rest = m.text.slice(pos).trim();
  if (rest) out.push({role: "assistant", content: rest});
  return out;
}

function setBusy(on) {
  $("stop-btn").hidden = !on;
  $("send-btn").disabled = on;
  $("composer-hint").textContent = on ? "" : "Shift+Enter: new line";
}

async function send() {
  const text = $("input").value.trim();
  if ((!text && !attachments.length) || busy) return;

  const active = sessions.find((s) => s.id === activeSessionId);
  if (active && (!active.title || active.title === "New Chat") && text) {
    const clean = text.replace(/\s+/g, " ").trim();
    active.title = clean.slice(0, 36) + (clean.length > 36 ? "…" : "");
    if ($("active-session-title")) $("active-session-title").textContent = active.title;
  }
  if (active) active.updatedAt = Date.now();

  messages.push({role: "user", text, images: attachments.filter((a) => a.kind !== "file"),
                 files: attachments.filter((a) => a.kind === "file"), time: Date.now()});
  attachments = [];
  renderAttachments();
  $("input").value = "";
  autosize();
  const m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
  messages.push(m);
  renderChat();
  renderSessionList();
  const el = $("chat").lastElementChild;
  const controller = new AbortController();
  busy = {controller, msg: m};
  setBusy(true);

  const body = {model: getSelectedModel() || health.model, messages: apiMessages(), stream: true,
                reasoning_effort: settings.thinking};
  if (settings.temperature > 0) {
    Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p, top_k: +settings.top_k});
  } else {
    body.temperature = 0;
  }
  if (settings.seed) body.seed = +settings.seed;
  if (settings.max) body.max_tokens = +settings.max;
  if (projectionLoaded()) body.experimental_speed_projection = !!settings.esp;
  if (settings.abliterate) {
    body.abliterate = true;
    if (settings.abliteration_strength !== undefined) {
      body.abliteration_strength = +settings.abliteration_strength;
    }
  }

  let firstAt = null, thinkStart = null, usage = null, frame = 0;
  const paint = () => { frame = 0; updateAssistant(el, m, true); scrollDown(); };
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify(body),
                                                   signal: controller.signal});
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
      if (r.status === 401) msg = "This server needs an API key: add it under About > Settings.";
      throw new Error(msg);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;              // ": keep-alive" comments while a long prompt is read
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let j;
        try { j = JSON.parse(data); } catch (e) { continue; }
        if (j.error) throw new Error(j.error.message || "the engine reported an error");
        if (j.usage) usage = j.usage;
        const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        const lastTool = m.tools && m.tools.length ? m.tools[m.tools.length - 1] : null;   // a new round after a tool
        if (d.reasoning_content) {
          if (!firstAt) firstAt = performance.now();
          if (!thinkStart) thinkStart = performance.now();
          if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) m.reasoning += "\n\n";
          m.reasoning += d.reasoning_content;
        }
        if (d.content) {
          if (!firstAt) firstAt = performance.now();
          if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
          if (lastTool && m.text && lastTool.at === m.text.length) m.text += "\n\n";
          m.text += d.content;
        }
        if (!frame) frame = requestAnimationFrame(paint);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else { m.error = e.message || String(e); toast("error", "The request failed", m.error, 6000); }
  }
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  const n = usage ? usage.completion_tokens : null;
  if (n && firstAt) {
    const secs = (performance.now() - firstAt) / 1000;
    m.meta = `${fmt(n)} tokens${secs > 0.25 ? ` · ${fmt(n / secs, 1)} tok/s` : ""}${m.stopped ? " · stopped" : ""}` +
             (projectionLoaded() ? (settings.esp ? " · projection on" : " · projection off") : "");
  } else if (m.stopped) {
    m.meta = "Stopped";
  }
  for (const t of m.tools || []) if (t.state === "writing" || t.state === "running") { t.state = "skipped"; t.ms = null; }
  const ran = (m.tools || []).filter((t) => t.state === "done" || t.state === "error").length;
  if (ran) m.meta = `${m.meta ? `${m.meta} · ` : ""}${ran} tool call${ran > 1 ? "s" : ""}`;
  busy = null;
  setBusy(false);
  if (frame) cancelAnimationFrame(frame);
  updateAssistant(el, m, false);
  saveChat();
  scrollDown();
}

$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
$("stop-btn").onclick = () => { if (busy) busy.controller.abort(); };
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, innerHeight * 0.4)}px`; }
$("input").addEventListener("input", autosize);

$("new-btn").onclick = () => createNewSession();
if ($("sidebar-new-btn")) $("sidebar-new-btn").onclick = () => createNewSession();

if ($("sidebar-close-btn")) {
  $("sidebar-close-btn").onclick = () => {
    const sb = $("chat-sidebar");
    if (sb) {
      sb.classList.add("collapsed");
      store.set("sidebar_collapsed", true);
    }
  };
}

if ($("sidebar-open-btn")) {
  $("sidebar-open-btn").onclick = () => {
    const sb = $("chat-sidebar");
    if (sb) {
      sb.classList.toggle("collapsed");
      store.set("sidebar_collapsed", sb.classList.contains("collapsed"));
    }
  };
}

if ($("active-session-title")) $("active-session-title").onclick = () => renameSession(activeSessionId);
if ($("rename-session-btn")) $("rename-session-btn").onclick = () => renameSession(activeSessionId);
if ($("fork-session-btn")) $("fork-session-btn").onclick = () => forkSession();
if ($("clear-all-sessions-btn")) $("clear-all-sessions-btn").onclick = () => clearAllSessions();

if ($("session-search-input")) {
  $("session-search-input").oninput = (e) => {
    sessionSearchQuery = e.target.value;
    renderSessionList();
  };
}

if ($("session-list")) {
  $("session-list").addEventListener("click", (e) => {
    const editBtn = e.target.closest("[data-act='rename']");
    if (editBtn) {
      e.stopPropagation();
      renameSession(editBtn.dataset.id);
      return;
    }
    const delBtn = e.target.closest("[data-act='delete']");
    if (delBtn) {
      e.stopPropagation();
      deleteSession(delBtn.dataset.id, e);
      return;
    }
    const item = e.target.closest(".session-item");
    if (item && item.dataset.id) {
      switchSession(item.dataset.id);
    }
  });
}

$("export-btn").onclick = () => {
  if (!messages.length) { toast("info", "Nothing to save yet"); return; }
  const activeSess = sessions.find((s) => s.id === activeSessionId);
  const titleSlug = (activeSess && activeSess.title ? activeSess.title.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 30) : "chat");
  const tools = (m) => (m.tools || []).filter((t) => t.result != null).map((t) =>
    `<details><summary>Tool ${t.server ? `${t.server} / ` : ""}${t.tool || t.name}${t.ok ? "" : " (error)"}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(t.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${t.result}\n\`\`\`\n\n</details>\n\n`).join("");
  const md = messages.map((m) => m.role === "user" ? `## You\n\n${m.text}\n` :
    `## ${health.model}\n\n${m.reasoning ? `<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${tools(m)}${m.text || m.error || ""}\n`).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], {type: "text/markdown"}));
  a.download = `strata-${titleSlug}-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};

// pictures and text files: the attach button, dropping them on the chat, or pasting a picture (issue #30)
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
const MAX_TEXT_FILE = 512 * 1024;
function isTextFile(f) {
  return f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
         TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
}
function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      if (!health.images) { toast("warn", "Pictures are off", "This model was set up for text only."); continue; }
      if (f.size > 20e6) { toast("warn", "Picture too large", `${f.name} is over 20 MB.`); continue; }
      const r = new FileReader();
      r.onload = () => { attachments.push({kind: "image", name: f.name || "pasted image", url: r.result}); renderAttachments(); };
      r.readAsDataURL(f);
      continue;
    }
    if (!isTextFile(f)) { toast("warn", "Not a text file", `${f.name}: attach text files (code, notes, logs, data)${health.images ? " or pictures" : ""}.`); continue; }
    if (f.size > MAX_TEXT_FILE) { toast("warn", "File too large", `${f.name} is over 512 KB.`); continue; }
    const r = new FileReader();
    r.onload = () => {
      const text = String(r.result);
      if (text.includes("\u0000")) { toast("warn", "Not a text file", `${f.name} looks like a binary file.`); return; }
      attachments.push({kind: "file", name: f.name, text});
      renderAttachments();
    };
    r.readAsText(f);
  }
}
// a file's text in the message, fenced with more backticks than it contains itself
function fileBlock(f) {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
function userText(m) {
  const files = (m.files || []).filter((f) => f.text != null);
  return [m.text, ...files.map(fileBlock)].filter((s) => s).join("\n\n");
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.innerHTML = "";
  attachments.forEach((a, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = icon(a.kind === "file" ? "attach" : "image", "st-icon st-icon--sm");
    c.append(a.name);
    const x = document.createElement("button");
    x.type = "button"; x.className = "st-btn st-btn--icon"; x.setAttribute("aria-label", "Remove");
    x.innerHTML = icon("trash");
    x.onclick = () => { attachments.splice(i, 1); renderAttachments(); };
    c.appendChild(x);
    box.appendChild(c);
  });
}
$("attach-btn").onclick = () => $("file").click();
// drop files on the chat or the message box
for (const id of ["chat", "composer"]) {
  const el = $(id);
  el.addEventListener("dragover", (e) => {
    if (![...(e.dataTransfer || {}).types || []].includes("Files")) return;
    e.preventDefault();
    $("composer").classList.add("dragging");
  });
  el.addEventListener("dragleave", () => $("composer").classList.remove("dragging"));
  el.addEventListener("drop", (e) => {
    $("composer").classList.remove("dragging");
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    addFiles(e.dataTransfer.files);
    $("input").focus();
  });
}
$("file").onchange = () => { addFiles($("file").files); $("file").value = ""; };
$("input").addEventListener("paste", (e) => {
  if (!health.images) return;
  const files = [...(e.clipboardData || {}).files || []].filter((f) => f.type.startsWith("image/"));
  if (files.length) { e.preventDefault(); addFiles(files); }
});

// ------------------------------------------------------------------ the sampling drawer
function openDrawer(open) {
  $("drawer").dataset.open = String(open);
  $("drawer").setAttribute("aria-hidden", String(!open));
  $("scrim").hidden = !open;
  if (open) { loadDrawer(); loadShared(); }
}
function loadDrawer(s = settings) {
  for (const b of $("s-thinking").children) b.setAttribute("aria-checked", String(b.dataset.v === s.thinking));
  $("s-temp").value = s.temperature; $("s-topp").value = s.top_p; $("s-topk").value = s.top_k;
  $("s-max").value = s.max; $("s-seed").value = s.seed;
  $("s-show").setAttribute("aria-checked", String(!!s.show));
  $("s-esp").setAttribute("aria-checked", String(s.esp !== false));
  $("esp-row").hidden = !projectionLoaded();
  $("s-abliterate").setAttribute("aria-checked", String(!!s.abliterate));
  $("s-abliterate-strength").value = s.abliteration_strength !== undefined ? s.abliteration_strength : 1.0;
  $("o-abliterate-strength").textContent = (+($("s-abliterate-strength").value)).toFixed(2);
  $("abliterate-strength-field").hidden = !s.abliterate;
  $("s-share").setAttribute("aria-checked", String(sharedOn));
  outputs();
}
// "Use for other apps too": the server keeps these settings as every client's defaults (GET/POST /settings)
let sharedOn = false;
async function loadShared() {
  try {
    const r = await fetch("settings", {headers: headers()});
    if (r.ok) sharedOn = !!(await r.json()).shared;
  } catch (e) { /* an older server: the switch just stays off */ }
  $("s-share").setAttribute("aria-checked", String(sharedOn));
}
function sharedDefaults(s) {
  const d = {reasoning_effort: s.thinking, temperature: +s.temperature};
  if (+s.temperature > 0) Object.assign(d, {top_p: +s.top_p, top_k: +s.top_k});
  if (s.seed) d.seed = +s.seed;
  if (s.max) d.max_tokens = +s.max;
  if (projectionLoaded()) d.experimental_speed_projection = s.esp !== false;
  if (s.abliterate) {
    d.abliterate = true;
    if (s.abliteration_strength !== undefined) d.abliteration_strength = +s.abliteration_strength;
  }
  return d;
}
async function saveShared(on, s) {
  const r = await fetch("settings", {method: "POST", headers: headers(true),
                                      body: JSON.stringify({defaults: on ? sharedDefaults(s) : null})});
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  sharedOn = !!(await r.json()).shared;
}
// the engine was started with the experimental-speed-projection control vector (INFO cvec=...)
function projectionLoaded() {
  const c = lastMetrics && lastMetrics.engine ? lastMetrics.engine.cvec : 0;
  return !!c && c !== "0";
}
function outputs() {
  const t = +$("s-temp").value;
  $("o-temp").textContent = t === 0 ? "0 · greedy" : t.toFixed(2);
  $("o-topp").textContent = (+$("s-topp").value).toFixed(2);
  $("o-topk").textContent = $("s-topk").value;
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  $("o-thinking").textContent = sel ? {none: "answers right away", low: "short", medium: "medium", high: "thorough (default)"}[sel.dataset.v] : "";
  for (const id of ["s-topp", "s-topk"]) $(id).disabled = t === 0;
}
for (const b of $("s-thinking").children) b.onclick = () => { for (const x of $("s-thinking").children) x.setAttribute("aria-checked", String(x === b)); outputs(); };
for (const id of ["s-temp", "s-topp", "s-topk"]) $(id).oninput = outputs;
$("s-show").onclick = () => $("s-show").setAttribute("aria-checked", String($("s-show").getAttribute("aria-checked") !== "true"));
$("s-esp").onclick = () => $("s-esp").setAttribute("aria-checked", String($("s-esp").getAttribute("aria-checked") !== "true"));
$("s-abliterate").onclick = () => {
  const on = $("s-abliterate").getAttribute("aria-checked") !== "true";
  $("s-abliterate").setAttribute("aria-checked", String(on));
  $("abliterate-strength-field").hidden = !on;
};
$("s-abliterate-strength").oninput = () => {
  $("o-abliterate-strength").textContent = (+($("s-abliterate-strength").value)).toFixed(2);
};
$("s-share").onclick = () => $("s-share").setAttribute("aria-checked", String($("s-share").getAttribute("aria-checked") !== "true"));
$("s-reset").onclick = () => loadDrawer(DEFAULTS);
$("s-apply").onclick = async () => {
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  settings = {thinking: sel ? sel.dataset.v : "high", temperature: +$("s-temp").value, top_p: +$("s-topp").value,
              top_k: +$("s-topk").value, max: $("s-max").value.trim(), seed: $("s-seed").value.trim(),
              show: $("s-show").getAttribute("aria-checked") === "true",
              esp: $("s-esp").getAttribute("aria-checked") === "true",
              abliterate: $("s-abliterate").getAttribute("aria-checked") === "true",
              abliteration_strength: +$("s-abliterate-strength").value};
  store.set("sampling", settings);
  const share = $("s-share").getAttribute("aria-checked") === "true";
  openDrawer(false);
  if (share || sharedOn) {
    try {
      await saveShared(share, settings);
      toast("success", "Sampling saved", share ? "Other apps (omp, API clients) use these settings from their next request."
                                               : "Other apps use their own settings again.");
    } catch (e) {
      toast("error", "Saved here, but not for other apps", e.message, 6000);
    }
    return;
  }
  toast("success", "Sampling saved", settings.temperature === 0 ? "Greedy: the same question gives the same answer." : "");
};
$("sampling-btn").onclick = () => openDrawer(true);
$("drawer-close").onclick = () => openDrawer(false);
$("scrim").onclick = () => openDrawer(false);
// ------------------------------------------------------------------ Setup & Hardware Configuration
let allSetupModels = [];
let detectedHwSpecs = null;
let activeSetupCat = "all";
let setupSearchQuery = "";
let selectedModelOpt = null;
let setupDlPolling = false;

async function loadSetup() {
  try {
    const r = await fetch("api/setup/models", { headers: headers() });
    if (r.ok) {
      const data = await r.json();
      detectedHwSpecs = data.hardware;
      allSetupModels = data.models || [];
      renderSetupHardware(data.hardware);
      filterAndRenderModels(data.recommended);
    }
  } catch (e) {
    console.error("Failed to load setup models:", e);
  }
  checkSetupStatus();
}

function renderSetupHardware(hw) {
  if (!hw) return;
  if ($("hw-cpu")) $("hw-cpu").textContent = hw.cpu_cores ? `${hw.cpu_cores} Cores` : "–";
  if ($("hw-ram")) $("hw-ram").textContent = hw.ram_total_gb ? `${fmt(hw.ram_total_gb, 1)} GB` : "–";
  if ($("hw-gpu")) $("hw-gpu").textContent = hw.gpu_name || hw.gpu_vendor || "CPU Only";
  if ($("hw-vram")) $("hw-vram").textContent = hw.vram_gb ? `${fmt(hw.vram_gb, 1)} GB` : (hw.gpu_vendor === "CPU" ? "Shared RAM" : "–");
}

function filterAndRenderModels(recommended) {
  const container = $("models-grid");
  if (!container) return;
  container.innerHTML = "";

  const q = setupSearchQuery.toLowerCase().trim();
  const filtered = allSetupModels.filter((m) => {
    // 1. Category Filter
    if (activeSetupCat === "recommended" && !m.recommended) return false;
    if (activeSetupCat === "installed" && !m.installed && !m.has_files && m.source !== "local") return false;
    if (activeSetupCat === "mimo" && m.category !== "mimo" && !(m.name || "").toLowerCase().includes("mimo") && !(m.tags || []).some((t) => t.toLowerCase() === "mimo")) return false;
    if (activeSetupCat === "huggingface" && m.source !== "huggingface") return false;
    if (activeSetupCat === "ollama" && m.source !== "ollama") return false;
    if (activeSetupCat === "local" && m.source !== "local") return false;
    if (activeSetupCat === "coder" && m.category !== "coder") return false;
    if (activeSetupCat === "reasoning" && m.category !== "reasoning" && !(m.tags || []).some((t) => t.toLowerCase() === "reasoning")) return false;
    if (activeSetupCat === "lightweight" && m.category !== "lightweight") return false;

    // 2. Search Query Filter
    if (q) {
      const matchName = (m.name || "").toLowerCase().includes(q);
      const matchDesc = (m.description || "").toLowerCase().includes(q);
      const matchQuant = (m.quantization || "").toLowerCase().includes(q);
      const matchParam = (m.param_size || "").toLowerCase().includes(q);
      const matchTags = (m.tags || []).some((t) => t.toLowerCase().includes(q));
      if (!matchName && !matchDesc && !matchQuant && !matchParam && !matchTags) return false;
    }
    return true;
  });

  if (filtered.length === 0) {
    container.innerHTML = `
      <div style="grid-column: 1 / -1; text-align: center; padding: var(--st-s-6); color: var(--st-ink-muted);">
        <p>No models matching <strong>${esc(setupSearchQuery || activeSetupCat)}</strong>.</p>
        <p class="small">Try changing filters or use the <em>Custom Hugging Face Import</em> bar above.</p>
      </div>
    `;
    return;
  }

  const hostRam = (detectedHwSpecs && detectedHwSpecs.ram_total_gb) || 16.0;

  filtered.forEach((m) => {
    const isRec = recommended && (m.name === recommended.name || m.recommended);
    const card = document.createElement("div");
    card.className = `model-card-item ${isRec ? "recommended" : ""}`;
    card.id = `mcard-${encodeURIComponent(m.name)}`;

    let sourceLabel = "Hugging Face";
    let sourceIcon = "🤗";
    if (m.source === "ollama") {
      sourceLabel = "Ollama";
      sourceIcon = "🦙";
    } else if (m.source === "local") {
      sourceLabel = "Local Pack";
      sourceIcon = "⚡";
    } else if (m.source === "custom") {
      sourceLabel = "Custom";
      sourceIcon = "🌐";
    }

    let ramClass = "ram-fit-ok";
    let ramIcon = "✓";
    if (hostRam < m.min_ram_gb) {
      ramClass = hostRam >= m.min_ram_gb * 0.75 ? "ram-fit-warn" : "ram-fit-err";
      ramIcon = hostRam >= m.min_ram_gb * 0.75 ? "⚠" : "✕";
    }

    const tagsHtml = (m.tags || [])
      .slice(0, 4)
      .map((t) => `<span class="model-tag">${esc(t)}</span>`)
      .join("");

    const hasFiles = Boolean(m.installed || m.has_files || m.source === "local");
    const isInstalled = Boolean(m.installed || m.source === "local");

    card.innerHTML = `
      <div class="model-card-top">
        <span class="model-source-badge" data-source="${esc(m.source || 'huggingface')}">
          ${sourceIcon} ${sourceLabel}
        </span>
        ${isInstalled ? `<span class="model-installed-badge">💾 Installed</span>` : ""}
        ${isRec ? `<span class="model-rec-badge">★ Recommended</span>` : ""}
        ${m.category === "mimo" || (m.tags || []).includes("MiMo") ? `<span class="model-rec-badge" style="background:linear-gradient(135deg,#ff6900,#ff8a3d);color:#fff;">🚀 MiMo RL</span>` : ""}
        ${hasFiles ? `<button class="model-card-delete-btn" title="Delete model files from disk" type="button"><svg class="st-icon st-icon--sm" style="width:14px;height:14px;pointer-events:none;"><use href="web/sprite.svg#i-trash"/></svg></button>` : ""}
      </div>
      <div class="model-card-name">${esc(m.name)}</div>
      <div class="model-tags-row">${tagsHtml}</div>
      <div class="model-card-desc">${esc(m.description || "")}</div>
      <div class="model-card-meta">
        <span class="${ramClass}">${ramIcon} Min RAM: ${fmt(m.min_ram_gb, 0)} GB</span>
        <span>Size: ${m.size_gb ? fmt(m.size_gb, 1) + " GB" : "Instant"}</span>
      </div>
    `;

    card.onclick = () => selectSetupModel(m);
    const delBtn = card.querySelector(".model-card-delete-btn");
    if (delBtn) {
      delBtn.onclick = (e) => {
        e.stopPropagation();
        deleteSetupModel(m);
      };
    }
    container.appendChild(card);
  });

  // Preserve selection or select default
  if (selectedModelOpt) {
    const stillThere = filtered.find((m) => m.name === selectedModelOpt.name);
    if (stillThere) {
      selectSetupModel(stillThere);
    } else if (filtered.length > 0) {
      selectSetupModel(filtered[0]);
    }
  } else {
    const defaultPick = recommended || filtered[0];
    if (defaultPick) {
      selectSetupModel(defaultPick);
    }
  }
}

function selectSetupModel(m) {
  selectedModelOpt = m;
  for (const el of document.querySelectorAll(".model-card-item")) {
    el.classList.remove("selected");
  }
  const chosen = $(`mcard-${encodeURIComponent(m.name)}`);
  if (chosen) chosen.classList.add("selected");

  if ($("setup-custom-url")) $("setup-custom-url").value = m.download_url || "";
  if ($("setup-dest-path")) {
    if (m.filename && (m.filename.startsWith("/") || m.filename.includes("\\"))) {
      $("setup-dest-path").value = m.filename;
    } else {
      $("setup-dest-path").value = `models/${m.filename || (m.name.toLowerCase() + ".gguf")}`;
    }
  }

  // Show or hide delete button based on local/installed status
  const deleteBtn = $("setup-delete-btn");
  if (deleteBtn) {
    deleteBtn.hidden = !(m.installed || m.has_files || m.source === "local");
  }

  // Update download button label depending on local vs remote
  const btn = $("setup-download-btn");
  if (btn) {
    if (m.download_url === "" || m.source === "ollama" || m.source === "local" || m.installed) {
      btn.innerHTML = `<svg class="st-icon st-icon--sm"><use href="web/sprite.svg#i-check"/></svg>Activate Local Model (Instant)`;
    } else {
      btn.innerHTML = `<svg class="st-icon st-icon--sm"><use href="web/sprite.svg#i-download"/></svg>Download & Activate (${fmt(m.size_gb, 1)} GB)`;
    }
  }
}

async function importCustomSetupModel() {
  const inputEl = $("setup-import-input");
  if (!inputEl) return;
  const val = inputEl.value.trim();
  if (!val) {
    toast("warn", "Enter Model", "Please enter a Hugging Face repo ID or direct GGUF URL.");
    return;
  }

  try {
    const r = await fetch("api/setup/resolve", {
      method: "POST",
      headers: headers(true),
      body: JSON.stringify({ input: val }),
    });
    if (r.ok) {
      const resolved = await r.json();
      allSetupModels.unshift(resolved);
      activeSetupCat = "all";
      document.querySelectorAll("#setup-cat-filters .st-pill").forEach((b) => {
        b.classList.toggle("active", b.dataset.cat === "all");
      });
      setupSearchQuery = "";
      if ($("setup-search-input")) $("setup-search-input").value = "";
      filterAndRenderModels(resolved);
      selectSetupModel(resolved);
      toast("success", "Model Resolved", `Added ${resolved.name} from ${resolved.repo_id}`);
    } else {
      const err = await r.json().catch(() => ({}));
      toast("error", "Resolution failed", err.error || "Could not resolve model from Hugging Face.");
    }
  } catch (e) {
    toast("error", "Network error", e.message);
  }
}

async function startSetupDownload() {
  if (!selectedModelOpt) {
    toast("warn", "Select a model", "Please choose a model option first.");
    return;
  }

  // If selecting an existing local pack, activate it directly without downloading
  if (selectedModelOpt.download_url === "" || (selectedModelOpt.name && selectedModelOpt.name.includes("Local"))) {
    await saveSetupConfig();
    toast("success", "Local Model Activated", `${selectedModelOpt.name} is now active.`);
    return;
  }

  const payload = {
    model_name: selectedModelOpt.name,
    download_url: ($("setup-custom-url") && $("setup-custom-url").value.trim()) || selectedModelOpt.download_url,
    dest_path: ($("setup-dest-path") && $("setup-dest-path").value.trim()) || `models/${selectedModelOpt.filename}`,
  };

  try {
    const r = await fetch("api/setup/download", {
      method: "POST",
      headers: headers(true),
      body: JSON.stringify(payload),
    });

    if (r.ok) {
      toast("info", "Download Started", `Downloading ${selectedModelOpt.name}...`);
      pollSetupDownload();
    } else {
      let errMsg = `HTTP ${r.status}`;
      try {
        const err = await r.json();
        if (typeof err.error === "string") errMsg = err.error;
        else if (err.error && err.error.message) errMsg = err.error.message;
        else if (err.message) errMsg = err.message;
      } catch (e) {}
      toast("error", "Download failed to start", errMsg);
    }
  } catch (e) {
    toast("error", "Connection error", e.message);
  }
}

async function saveSetupConfig() {
  const payload = {
    model_name: selectedModelOpt ? selectedModelOpt.name : "",
    model_path: ($("setup-dest-path") && $("setup-dest-path").value.trim()) || "",
    pool_workers: +($("setup-pool-workers") ? $("setup-pool-workers").value : 16),
    pcie_frac: +($("setup-pcie-frac") ? $("setup-pcie-frac").value : 0.5),
    gpus: ($("setup-gpus") && $("setup-gpus").value.trim()) || "",
  };

  try {
    const r = await fetch("api/setup/configure", {
      method: "POST",
      headers: headers(true),
      body: JSON.stringify(payload),
    });
    if (r.ok) {
      toast("success", "Configuration Saved", "Settings updated for active server session.");
      loadHealth();
    } else {
      toast("error", "Failed to save configuration");
    }
  } catch (e) {
    toast("error", "Error saving config", e.message);
  }
}

async function cancelSetupDownload() {
  try {
    await fetch("api/setup/cancel", { method: "POST", headers: headers(true) });
    toast("info", "Download Cancelled");
    if ($("setup-cancel-btn")) $("setup-cancel-btn").hidden = true;
    if ($("setup-dl-card")) $("setup-dl-card").hidden = true;
  } catch (e) {
    toast("error", "Cancel failed", e.message);
  }
}

async function pollSetupDownload() {
  if (setupDlPolling) return;
  setupDlPolling = true;

  const check = async () => {
    try {
      const r = await fetch("api/setup/status", { headers: headers() });
      if (r.ok) {
        const st = await r.json();
        const dl = st.download || {};
        const card = $("setup-dl-card");
        const cancelBtn = $("setup-cancel-btn");

        if (dl.status === "downloading") {
          if (card) card.hidden = false;
          if (cancelBtn) cancelBtn.hidden = false;
          if ($("setup-dl-label")) $("setup-dl-label").textContent = `Downloading ${dl.model_name || "Model"}...`;
          if ($("setup-dl-pct")) $("setup-dl-pct").textContent = `${fmt(dl.percent, 1)}%`;
          if ($("setup-dl-bar")) $("setup-dl-bar").style.width = `${dl.percent}%`;
          if ($("setup-dl-speed")) $("setup-dl-speed").textContent = `Speed: ${fmt(dl.speed_mb_s, 2)} MB/s`;
          if ($("setup-dl-bytes")) $("setup-dl-bytes").textContent = `${gb(dl.downloaded_bytes, 2)} / ${gb(dl.total_bytes, 2)} GB`;
          if ($("setup-dl-err")) $("setup-dl-err").hidden = true;
          setTimeout(check, 1000);
          return;
        } else if (dl.status === "completed") {
          if (card) card.hidden = false;
          if (cancelBtn) cancelBtn.hidden = true;
          if ($("setup-dl-label")) $("setup-dl-label").textContent = `Download Complete (${dl.model_name})`;
          if ($("setup-dl-pct")) $("setup-dl-pct").textContent = "100%";
          if ($("setup-dl-bar")) $("setup-dl-bar").style.width = "100%";
          if ($("setup-dl-err")) $("setup-dl-err").hidden = true;
          toast("success", "Model Ready", `${dl.model_name} is downloaded and activated!`);
          setupDlPolling = false;
          loadHealth();
          return;
        } else if (dl.status === "failed") {
          if (card) card.hidden = false;
          if (cancelBtn) cancelBtn.hidden = true;
          if ($("setup-dl-err")) {
            $("setup-dl-err").hidden = false;
            $("setup-dl-err").textContent = `Error: ${dl.error || "Unknown error"}`;
          }
          setupDlPolling = false;
          return;
        }
      }
    } catch (e) {}
    setupDlPolling = false;
  };
  check();
}

async function checkSetupStatus() {
  try {
    const r = await fetch("api/setup/status", { headers: headers() });
    if (r.ok) {
      const st = await r.json();
      if (st.download && st.download.status === "downloading") {
        pollSetupDownload();
      }
    }
  } catch (e) {}
}

async function deleteSetupModel(m) {
  const modelToDel = m || selectedModelOpt;
  if (!modelToDel) return;
  const name = modelToDel.name || modelToDel.filename || "selected model";
  const confirmed = confirm(`Are you sure you want to delete "${name}"?\n\nThis will remove all downloaded model weights and shard files from disk to free storage space.`);
  if (!confirmed) return;

  const btn = $("setup-delete-btn");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<svg class="st-icon st-icon--sm"><use href="web/sprite.svg#i-refresh"/></svg>Deleting...`;
  }

  try {
    const r = await fetch("api/setup/delete", {
      method: "POST",
      headers: headers(true),
      body: JSON.stringify({
        model_name: modelToDel.name,
        filename: modelToDel.filename,
        path: modelToDel.filename
      })
    });
    const data = await r.json();
    if (r.ok) {
      const freed = data.freed_gb ? `${data.freed_gb} GB` : `${data.freed_bytes || 0} bytes`;
      alert(`Model files deleted successfully.\nFreed ${freed} of disk space.`);
      await loadSetup();
    } else {
      alert(`Failed to delete model: ${(data.error && data.error.message) || "Unknown error"}`);
    }
  } catch (err) {
    alert(`Failed to delete model: ${err.message}`);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<svg class="st-icon st-icon--sm"><use href="web/sprite.svg#i-trash"/></svg>Delete Model Files`;
    }
  }
}

if ($("hw-refresh-btn")) $("hw-refresh-btn").onclick = loadSetup;
if ($("setup-download-btn")) $("setup-download-btn").onclick = startSetupDownload;
if ($("setup-save-btn")) $("setup-save-btn").onclick = saveSetupConfig;
if ($("setup-delete-btn")) $("setup-delete-btn").onclick = () => deleteSetupModel(selectedModelOpt);
if ($("setup-cancel-btn")) $("setup-cancel-btn").onclick = cancelSetupDownload;
if ($("setup-import-btn")) $("setup-import-btn").onclick = importCustomSetupModel;
if ($("setup-import-input")) {
  $("setup-import-input").onkeydown = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      importCustomSetupModel();
    }
  };
}
if ($("setup-search-input")) {
  $("setup-search-input").oninput = (e) => {
    setupSearchQuery = e.target.value;
    filterAndRenderModels();
  };
}
document.querySelectorAll("#setup-cat-filters .st-pill").forEach((pill) => {
  pill.onclick = () => {
    activeSetupCat = pill.dataset.cat || "all";
    document.querySelectorAll("#setup-cat-filters .st-pill").forEach((p) => p.classList.remove("active"));
    pill.classList.add("active");
    filterAndRenderModels();
  };
});

if ($("mon-detail-close")) {
  $("mon-detail-close").onclick = () => {
    if ($("mon-detail-card")) $("mon-detail-card").hidden = true;
    selectedMonReq = null;
    document.querySelectorAll("#req-body tr.clickable").forEach(t => t.classList.remove("selected"));
  };
}

if ($("mon-copy-btn")) {
  $("mon-copy-btn").onclick = () => {
    const text = $("mon-content") ? $("mon-content").textContent : "";
    if (text) copyText(text, null);
  };
}

document.querySelectorAll("[data-mon-pane]").forEach((btn) => {
  btn.onclick = () => {
    monPane = btn.dataset.monPane;
    document.querySelectorAll("[data-mon-pane]").forEach(b => b.setAttribute("aria-selected", String(b === btn)));
    renderMonPaneContent();
  };
});

document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("drawer").dataset.open === "true") openDrawer(false); });

// ------------------------------------------------------------------ start
setBusy(false);
initSessions();
renderChat();
const startQuestion = new URLSearchParams(location.search).get("q");   // /?q=... starts a chat (a shortcut)
if (startQuestion) history.replaceState(null, "", location.pathname + location.hash);
loadHealth().then(() => { if (startQuestion) { $("input").value = startQuestion; send(); } });
showTab(location.hash.slice(1) || "chat");
poll();
