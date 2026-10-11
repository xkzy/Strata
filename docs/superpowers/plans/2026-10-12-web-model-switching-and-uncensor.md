# Implementation Plan: Web Model Switching, Last Session Restore & Web Uncensor Config

**Goal:** Enable seamless model switching on the web UI, automatic startup with the last session's configuration, and an in-web uncensor/abliteration configuration for all models.

---

### Task 1: Last Session Restore in Server Engine (`pkg/server` & `pkg/runconfig`)
- Add `LastConfigFile` to `StrataConfig` in `pkg/runconfig/runconfig.go`.
- In `pkg/server/engine.go`:
  - Check `s.ConfigManager.Get().LastConfigFile` when `s.Config.ConfigFile` is empty.
  - If multiple `strata-*.json` configs exist on disk, pick the latest modified rather than returning an error.
  - Persist `LastConfigFile` whenever a model starts or is switched.
- Test in `pkg/server/engine_test.go`.

---

### Task 2: Backend Switch Endpoint (`POST /api/models/switch`)
- Add `/api/models/switch` in `pkg/server/server.go`.
- Accepts `{"model": "name", "config": "path"}`.
- Stops active engine process (`s.EngineIPC.Stop()`).
- Resolves or generates the target `strata-<model>.json` config.
- Updates `s.Config.ConfigFile`, `s.Config.ModelName`, and `LastConfigFile`.
- Calls `s.StartEngine(context.Background())`.
- Returns `{"status": "loading", "model": name}`.
- Test with unit test in `pkg/server/server_test.go`.

---

### Task 3: Global & Per-Model Uncensor (Abliteration) in Server
- In `pkg/server/generate.go`:
  - When `spec.Sampling.Abliterate` is not explicitly set in the request, fallback to `s.ConfigManager.Get().Abliterate` or the runtime profile's `Abliterate` setting.
- In `pkg/server/server.go`:
  - In `handleSetupConfigure`, accept `abliterate` (`bool`) and `abliteration_strength` (`float64`) and persist them.
- Expose `abliterate` in `/health` and `/api/model` responses.

---

### Task 4: Frontend Web UI Enhancements (`pkg/frontend/web`)
- In `pkg/frontend/web/app.js`:
  - `switchActiveModel(name)` function sending `POST /api/models/switch`.
  - Connect `#model-select` `onchange` to `switchActiveModel`.
  - Add "⚡ Activate" button on installed model cards in `#models-grid`.
  - Add "Uncensor Model (Abliteration)" toggle & strength slider in Model settings form (`#cfg-form` / `#cfg-card`).
- Preserve chat history across model switches.
