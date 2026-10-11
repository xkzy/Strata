# Design: Web-Based Dynamic Model Switching for Strata

**Date:** 2026-10-12  
**Target:** Enable seamless dynamic model switching directly from the Web UI (top header dropdown and Models catalog) with background engine reloading and conversation preservation.

---

## 1. Objectives

1. Allow users to switch between models directly inside the Web UI without stopping the server or modifying CLI flags.
2. Provide two accessible interaction surfaces:
   - **Header Dropdown (`#model-select`)**: Instant switching between all configured / installed models on the machine.
   - **Models Tab Grid (`#models-grid`)**: An "Activate / Switch" action button on every installed model card in the catalog.
3. Keep conversation history intact across model switches so the user can continue their chat with the new model.
4. Provide immediate visual feedback (loading indicator, toast notifications, status badge updates) while the engine unloads the old weights and maps the new model.

---

## 2. Backend Design (`pkg/server`)

### 2.1. Dedicated Endpoint: `POST /api/models/switch`
- **Request Body**:
  ```json
  {
    "model": "GPT-OSS-20B-UD-Q4_K_XL",
    "config": "strata-gpt_oss_20b.json" // optional
  }
  ```
- **Execution Flow**:
  1. Acquire server switch mutex to prevent overlapping switch requests.
  2. Stop active engine process:
     ```go
     if s.EngineIPC != nil {
         _ = s.EngineIPC.Stop()
     }
     ```
  3. Resolve model target:
     - Check for matching existing `strata-<name>.json` config file.
     - If none exists, auto-generate standard run config via `installer.GenerateRunConfig()` and save it.
     - Update `s.Config.ConfigFile` and `s.Config.ModelName`.
  4. Launch new engine asynchronously via `s.StartEngine(context.Background())`.
  5. Return HTTP 200:
     ```json
     {
       "status": "switching",
       "model": "GPT-OSS-20B-UD-Q4_K_XL"
     }
     ```

### 2.2. Health & Models Enumeration
- Update `/health` response `models` array to include all installed models and available `strata-*.json` configs on disk, so the frontend `#model-select` dropdown is always populated with selectable models.

---

## 3. Frontend Web UI Design (`pkg/frontend/web`)

### 3.1. Top Header Model Selector (`#model-select`)
- Hook `onchange`:
  ```javascript
  sel.onchange = async () => {
    const targetModel = sel.value;
    await switchActiveModel(targetModel);
  };
  ```
- `switchActiveModel(targetModel)`:
  - Sends `POST /api/models/switch` with `{ model: targetModel }`.
  - Shows progress toast: `"Switching model to {targetModel}... Loading engine"`.
  - Keeps existing messages array in memory.
  - Polls `/health` every 1.5 seconds until `health.status === "ready" && health.model === targetModel`.
  - Updates sub-banner: `"{targetModel} runs on this PC."`.

### 3.2. Models Catalog Grid (`#models-grid`)
- On cards where `isInstalled === true`:
  - Render an "Activate Model" action button.
  - Clicking "Activate Model" invokes `switchActiveModel(m.name)` and navigates to the Chat view with toast confirmation.
