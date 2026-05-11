# CHORAS Integration Reference for PPFFDTD

**Status:** authoritative reference for integrating PPFFDTD as a CHORAS simulation method, hand-merged at TU Eindhoven.

**Companion docs:** `C:\tmp\choras_research\docs\01_backend.md` (full backend reference, 2187 lines), `C:\tmp\choras_research\docs\03_frontend-v2.md` (full frontend reference, 1554 lines).

---

## 1. CHORAS at a glance

CHORAS is a multi-method room-acoustics platform developed by the Building Acoustics group at TU Eindhoven (Hornikx group; backend maintained by Silvin Willemsen).

**Repos** (all under `https://github.com/choras-org/`):

| Repo | Role | Status |
|---|---|---|
| `CHORAS` | Orchestrator. `docker-compose.yml` only. `pull_policy: always` on prebuilt images. | v0.1 tag at `b868246` (2025-11-04) |
| `backend` | Flask + SQLAlchemy + Celery. **Methods import as Python modules in the Celery worker** (not in separate containers at runtime). | Active, pushed today |
| `frontend-v2` | React 19 + Vite 7 + RTK Query + shadcn/ui. dev branch is the live code; main is empty placeholder. | Active, pushed 3 days ago |
| `simulation-backend` | Method interfaces. Each method has its own dir with `<MethodName>Interface.py` + `Dockerfile` + `requirements.txt`. The Dockerfiles are for headless dev only — production uses in-process imports. | Active, pushed today |
| `template_simulation_method` | Copier scaffold for new methods. **0 adopters; ABC contract incompatibility with `simulation-backend/common/simulation_method_interface.py`. We do NOT use it.** | Pushed 1 day ago |

**Service topology** (`docker-compose up`):

```
postgres:13            (db_service)         :5432
redis:alpine           (redis)              6379 (internal)
silvinwillemsen/choras-frontend:latest      :5173  (Vite dev server)
silvinwillemsen/choras-backend:latest       :5001  (Flask + Celery)
```

`pull_policy: always` + floating `:latest` → reproducibility is by design weak. Hand-merged integrations need to be tested against the live image.

---

## 2. The `SimulationMethod` contract

`simulation-backend/common/simulation_method_interface.py` (8 lines):

```python
from abc import ABC, abstractmethod
from pathlib import Path

class SimulationMethod(ABC):
    @abstractmethod
    def run_simulation(self, json_file_path: str | Path):
        """ Runs the simulation given a json file. """
        pass
```

**Contract:** mutate the JSON file in place. Return value unused. Argument is an absolute path string to a JSON file in the backend's `uploads/` working directory.

**Caller in production:** `app/services/simulation_service.py::run_solver` (Celery task). Methods are imported **inside the task body** (not at module top) to avoid loading heavy solver deps into gunicorn web workers.

**`save_results` quirk:** `simulation-backend/common/HelperFunctions.py::save_results` POSTs the mutated JSON to `http://host.docker.internal:5001/receive`. **The `/receive` endpoint does not exist in the backend.** The POST fails silently. The actual contract is JSON mutation in place — the backend re-reads the file after `run_solver` returns.

---

## 3. Output JSON schema

PPFFDTD must read AND write to the same JSON file. Required output fields (from `tests/edg-acoustics/test_input_edg_acoustics.json` and DG's `DGinterface.py`):

```json
{
  "absorption_coefficients": { "<surface_name>": "0.6, 0.69, 0.71, 0.7, 0.63", ... },
  "msh_path": "<path>",
  "geo_path": "<path>",
  "simulationSettings": { "<id>": <value>, ... },
  "results": [
    {
      "percentage": 100,                          // SET TO 100 ON COMPLETION
      "sourceX": 2, "sourceY": 2, "sourceZ": 1.5,
      "resultType": "PFFDTD",                     // SET TO METHOD ID STRING
      "frequencies": [125, 250, 500, 1000, 2000], // INPUT (5 octave bands typical)
      "responses": [
        {
          "x": 1, "y": 1, "z": 1.5,
          "parameters": {
            "edt":         [<f1>, <f2>, ...],     // PER-BAND ARRAYS (len == len(frequencies))
            "t20":         [<f1>, <f2>, ...],
            "t30":         [<f1>, <f2>, ...],
            "c80":         [<f1>, <f2>, ...],
            "d50":         [<f1>, <f2>, ...],
            "ts":          [<f1>, <f2>, ...],
            "spl_t0_freq": [<f1>, <f2>, ...]
          },
          "receiverResults":            [<ir_sample_0>, <ir_sample_1>, ...],  // RAW IR (DG-style)
          "receiverResultsUncorrected": [<ir_sample_0>, <ir_sample_1>, ...]   // PRE-CORRECTION IR
        }
      ]
    }
  ]
}
```

**f-range gating:** if a frequency band's center exceeds `0.9 * fmax_grid`, write `null` (or `NaN` serialized as null) for that band's metric value. Don't fabricate metrics for unsupported bands.

**Auralization side-effect:** the backend's `auralization_service.auralization_calculation_DG` reads a CSV file at `json_path.replace(".json", "_pressure.csv")` with columns `t, pressure`. PPFFDTD will write this CSV alongside the JSON for the auralization handler to find.

---

## 4. Material spec

CHORAS sends materials as comma-separated strings of α per band (Sabine-style, random-incidence):

```json
"absorption_coefficients": {
  "floor":   "0.6, 0.69, 0.71, 0.7, 0.63",
  "wall1":   "0.6, 0.69, 0.71, 0.7, 0.63",
  "ceiling": "0.6, 0.69, 0.71, 0.7, 0.63",
  ...
}
```

Bands are `[125, 250, 500, 1000, 2000]` Hz (5-band default for DE/DG). PPFFDTD currently expands these to 11 bands (16 Hz to 16 kHz) for the DEF triplet fit — keep that pipeline.

Surface names are physical group names from the `.geo`/`.msh` mesh. CHORAS may suffix with `$<tag>`; split on `$` and use the first part (existing interface already does this in `PFDTDInterface.py:198`).

---

## 5. Geometry input

Backend writes `.geo` and `.msh` paths into the JSON (relative to JSON's parent dir or absolute). Existing `PFDTDInterface.py:80-83` handles both. **`MeasurementRoom.obj` (the canonical test geometry)** segments by `usemtl` markers (M_1/M_2/M_3), not `g` groups; the backend's `obj_to_gmsh_geo_precise` converts OBJ → `.geo` with physical groups named after the `usemtl` tags.

---

## 6. Settings JSON schema (drives the frontend form)

Two registry files. Both must be edited per integration.

### 6.1 `app/models/data/simulation_settings.json` (in backend)

This is the **method registry**. Schema:

```json
{
  "simulationType": "PFFDTD",                   // primary key; MUST equal TaskType.PFFDTD.value
  "name": "PFFDTD_setting.json",                // filename in example_settings/
  "label": "PPFFDTD (PFFDTD + POD-GP ROM)",     // shown in frontend method selector
  "description": "PPFFDTD: a non-intrusive POD+GP reduced-order model wrapper around Brian Hamilton's PFFDTD finite-difference time-domain solver.",
  "repositoryURL": "https://github.com/Burhanuddin98/PPFFDTD",
  "documentationURL": ""
}
```

After editing, run `flask reset-db` to drop+recreate SQLite and re-seed.

### 6.2 `example_settings/PFFDTD_setting.json` (in backend) — UI form

Schema:

```json
{
  "type": "simulationSettings",
  "options": [ <option_1>, <option_2>, ... ]
}
```

Each option is a UI widget descriptor:

| Field | Type | Notes |
|---|---|---|
| `name` | string | Label shown to user |
| `id` | string | **Key into `result_container["simulationSettings"][id]`.** PFFDTD interface reads with this exact key. |
| `type` | `"integer"` \| `"float"` \| `"string"` \| `"list"` | Storage type |
| `display` | `"text"` \| `"slider"` \| `"radio"` \| `"checkbox"` | UI widget |
| `min`, `max` | number | Bounds (nullable) |
| `step` | number | Slider/spinner step |
| `default` | varies | Initial value (`null` allowed) |
| `startAdornment`, `endAdornment` | string | Inline prefix/suffix (e.g. `"Hz"`, `"m"`) |
| `options` | dict {label: value} | For `radio`/`checkbox` |

**Critical:** the `id` strings here MUST match the keys PFFDTDInterface reads from `simulationSettings`. Current interface reads `pffdtd_c0`, `pffdtd_fmax`, `pffdtd_ppw`, `pffdtd_ir_length`, `pffdtd_temperature`, `pffdtd_humidity`, `pffdtd_use_gpu`, `pffdtd_use_rom`, `pffdtd_train_rom`. Match these in the JSON.

---

## 7. Backend integration recipe (precise)

Three places to edit `backend/app/services/simulation_service.py`:

### 7.1 In `start_solver_task` (around lines 219-227)

After the existing MyNewMethod branch, add:

```python
if simulation.taskType.value == TaskType.PFFDTD.value:
    task_statuses.append(create_source_task(TaskType.PFFDTD.value, source["id"]))
    results_container.append(
        create_result_source_object(source, simulation.receivers, TaskType.PFFDTD.value)
    )
```

### 7.2 In `run_solver` (around line 323)

Add the import inside the function body (not at module top):

```python
from simulation_backend.PFFDTDInterface import pffdtd_method
```

Add the match case (around lines 389-405):

```python
case TaskType.PFFDTD:
    pffdtd_method(json_file_path=json_path)
    logger.info("PFFDTD method")
```

### 7.3 In the auralization match block (around lines 433-445)

Reuse DG's path (per integration decision):

```python
case TaskType.PFFDTD:
    imp_tot, fs = auralization_calculation_DG(
        None,
        json_path.replace(".json", "_pressure.csv"),
        json_path.replace(".json", ".wav"),
    )
```

### 7.4 In `app/types/Task.py`

Add the enum value:

```python
class TaskType(Enum):
    GeometryCheck = "GeometryCheck"
    Mesh = "Mesh"
    DE = "DE"
    DG = "DG"
    MyNewMethod = "MyNewMethod"
    BOTH = "BOTH"
    PFFDTD = "PFFDTD"   # ADD THIS
```

### 7.5 In `simulation-backend/simulation_backend/__init__.py`

Append:

```python
from .PFFDTDInterface import pffdtd_method
```

### 7.6 In `simulation-backend/simulation_backend/PFFDTDInterface.py` (new file)

Module-level function `pffdtd_method(json_file_path)` that wraps our class. Mirrors `dg_method` in `DGinterface.py`.

---

## 8. Frontend integration

**The frontend is method-agnostic for the core flow.** No frontend changes required for PPFFDTD to appear in the dropdown. The frontend reads:

- `GET /simulation_settings` — list of available methods (drives the dropdown)
- `GET /simulation_settings/<simulationType>` — the specific UI form schema

Both are populated automatically by the backend after `flask reset-db` reads `simulation_settings.json` + the per-method `<Type>_setting.json`.

**Hard limits in the frontend (PR-blocked):**

- `sourceReceiverSlice.ts:25-29, 46-50` and `useJsonValidation.ts:86-95, 132-141` enforce **1 source / 1 receiver max**. PPFFDTD's existing interface accepts multiple receivers in its `responses` array — the backend will pass them through, but the frontend won't let users add a second receiver via UI. Acceptable for v1.
- `simulationSettingsSlice.ts:11`'s default `selectedMethodType: "DE"` — user must explicitly switch to PPFFDTD in dropdown each session.
- `Parameters` interface in `src/types/simulation.ts:132-140` types only the standard 7 metrics. Our extra fields (`t30_std`, sensitivity) won't render in `ResultParameters.tsx` until a frontend PR. **v1 doesn't add these fields.**

---

## 9. PPFFDTD-side changes

### 9.1 Files to modify in PPFFDTD repo

| File | Change |
|---|---|
| `pffdtd_method/PFDTDInterface.py` | Full refactor (see §10). Rename to `PFFDTDInterface.py` to match `<MethodName>Interface.py` convention. |
| `pffdtd_method/Dockerfile` | Bump base to `python:3.11.13-slim`; restructure for build context = `simulation-backend/` root; mirror `dg_method/Dockerfile`'s COPY pattern. |
| `pffdtd_method/requirements.txt` | Replace local PFFDTD submodule with `git+https://github.com/bsxfun/pffdtd.git@f79192d` (current submodule SHA); add ROM dependencies (numpy, scipy, scikit-learn for GP). |
| `README.md` | Sync claims to code (currently mentions GP+33-Smolyak, but interface ships RBF; we're unifying on GP). |

### 9.2 Files to delete

In `pffdtd_method/PFDTDInterface.py`:
- `_train_rom` method (lines ~399-496) — replace with call to `NonIntrusiveROM.train`
- `_run_rom` method (lines ~498-551) — replace with call to `NonIntrusiveROM.evaluate`

### 9.3 Files to create

| File | Purpose |
|---|---|
| `choras_integration/pffdtd_method/PFFDTDInterface.py` | The drop-in for `simulation-backend/simulation_backend/PFFDTDInterface.py` |
| `choras_integration/pffdtd_method/Dockerfile` | The drop-in for headless dev |
| `choras_integration/pffdtd_method/requirements.txt` | Pip deps |
| `choras_integration/common/exampleInput_PFFDTD.json` | CHORAS-conformant input matching DG/DE shape |
| `choras_integration/backend_patches/Task.py.add` | Snippet to add to `app/types/Task.py` |
| `choras_integration/backend_patches/simulation_service.py.add` | Snippets to add to `app/services/simulation_service.py` |
| `choras_integration/backend_patches/simulation_settings.json.entry` | JSON entry to append to `app/models/data/simulation_settings.json` |
| `choras_integration/backend_patches/methods-config.json.entry` | JSON entry to append to `simulation-backend/methods-config.json` |
| `choras_integration/backend_patches/PFFDTD_setting.json` | Drop into `example_settings/` |
| `choras_integration/MERGE_INSTRUCTIONS.md` | Step-by-step for the TUE team |

### 9.4 Per-band metric extraction (new feature)

PPFFDTD currently writes broadband T30/EDT/C80 only. CHORAS expects per-band arrays. New helper in `pffdtd_method/`:

```python
def _compute_per_band_metrics(ir_48k, fs, frequencies, fmax_grid):
    """Bandpass IR into octave bands, compute ISO 3382 metrics per band.
    
    Bands above 0.9 * fmax_grid get None (JSON null).
    Returns dict of arrays: {edt, t20, t30, c80, d50, ts, spl_t0_freq}.
    """
    # Reuse C:\RoomGUI\room_acoustics\acoustics_metrics.py from parent ROM project
    # Or re-implement: 4th-order Butterworth bandpass per band -> Schroeder integration -> T30/EDT/C80
```

Pull from parent ROM repo at `C:\RoomGUI\room_acoustics\acoustics_metrics.py`. Vendored copy goes into `pffdtd_method/iso_metrics.py`.

### 9.5 Wire to NonIntrusiveROM

`pffdtd_method/PFFDTDInterface.py` becomes:

```python
from ppffdtd.rom import NonIntrusiveROM

class PFFDTDMethod(SimulationMethod):
    def run_simulation(self, json_file_path):
        # Parse JSON, build PFFDTD model directory (existing _build_model)
        # 
        # If simulationSettings["pffdtd_use_rom"] and rom cache exists:
        #     rom = NonIntrusiveROM(...)
        #     rom.load(cache_path)
        #     ir = rom.evaluate(alpha_scales)
        # 
        # Else: full FDTD via existing _run_fdtd
        # 
        # If simulationSettings["pffdtd_train_rom"]:
        #     rom = NonIntrusiveROM(...)
        #     rom.train(baseline_alphas, dim=3, level=2, use_gpu=use_gpu)
        #     rom.save(cache_path)
        #
        # Compute per-band metrics from ir (with f-range gating)
        # Write back to JSON: parameters arrays + receiverResults + percentage=100 + resultType="PFFDTD"
        # Write _pressure.csv for auralization handler
```

---

## 10. Refactored `PFFDTDInterface.py` skeleton

```python
"""CHORAS interface for PPFFDTD (PFFDTD + POD-GP ROM)."""
import os, sys, json, csv
import numpy as np
from pathlib import Path

# (path setup for PFFDTD submodule + common)

from simulation_method_interface import SimulationMethod
from ppffdtd.rom import NonIntrusiveROM
from .iso_metrics import compute_per_band_metrics


class PFFDTDMethod(SimulationMethod):
    def run_simulation(self, json_file_path: str):
        json_path = Path(json_file_path)
        with open(json_path) as f:
            config = json.load(f)
        
        settings = config.get("simulationSettings", {})
        c0       = float(settings.get("pffdtd_c0", 343.0))
        fmax     = float(settings.get("pffdtd_fmax", 1000.0))
        ppw      = int(settings.get("pffdtd_ppw", 6))
        ir_len   = float(settings.get("pffdtd_ir_length", 1.0))
        Tc       = float(settings.get("pffdtd_temperature", 20.0))
        rh       = float(settings.get("pffdtd_humidity", 50.0))
        use_gpu  = bool(settings.get("pffdtd_use_gpu", True))
        use_rom  = bool(settings.get("pffdtd_use_rom", False))
        train_rom = bool(settings.get("pffdtd_train_rom", False))
        
        h = c0 / (fmax * ppw)
        fmax_grid = c0 / (2 * h)
        
        # Parse geometry, materials, source/receivers (existing code)
        # ...
        
        # Cache key: hash of (model, src, recvs, baseline alphas)
        cache_dir = Path("/app/cache") if Path("/app/cache").exists() else json_path.parent / "pffdtd_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        rom_path = cache_dir / f"rom_{cache_key}.npz"
        
        # Train mode
        if train_rom:
            self._build_model(...)  # produce common/pffdtd_data
            rom = NonIntrusiveROM(str(work_dir / "pffdtd_data"), str(work_dir / "pffdtd"))
            rom.train(baseline_alphas, dim=3, level=2, use_gpu=use_gpu)
            rom.save(str(rom_path))
        
        # Use ROM if available
        ir = None
        if use_rom and rom_path.exists():
            rom = NonIntrusiveROM(str(work_dir / "pffdtd_data"), str(work_dir / "pffdtd"))
            rom.load(str(rom_path))
            alpha_scales = self._abs_coeffs_to_scales(abs_coeffs, baseline_alphas)
            ir = rom.evaluate(alpha_scales)[0]  # [0] = mean prediction; [1] = posterior std (v2)
        
        # Otherwise full FDTD
        if ir is None:
            self._build_model(...)
            irs = self._run_fdtd(work_dir, use_gpu=use_gpu)
            ir = irs[0]
        
        # Per-band metrics with f-range gating
        frequencies = config["results"][0].get("frequencies", [125, 250, 500, 1000, 2000])
        metrics = compute_per_band_metrics(ir, fs=48000.0, frequencies=frequencies, fmax_grid=fmax_grid)
        
        # Write results back
        result_block = config["results"][0]
        result_block["resultType"] = "PFFDTD"
        result_block["percentage"] = 100
        for i, resp in enumerate(result_block["responses"]):
            if i < 1:  # currently 1-receiver; loop receivers when multi-receiver lands
                resp["receiverResults"] = ir.tolist()
                resp["receiverResultsUncorrected"] = ir.tolist()
                resp["parameters"] = metrics  # arrays of len(frequencies), with None where gated
        
        with open(json_path, "w") as f:
            json.dump(config, f, indent=4)
        
        # Auralization sidecar CSV (DG-style, two columns: t, pressure)
        csv_path = str(json_path).replace(".json", "_pressure.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "pressure"])
            t = np.arange(len(ir)) / 48000.0
            for ti, pi in zip(t, ir):
                w.writerow([f"{ti:.6f}", f"{pi:.9f}"])


# Module-level entry point per CHORAS pattern (matches dg_method)
def pffdtd_method(json_file_path: str):
    PFFDTDMethod().run_simulation(json_file_path)


if __name__ == "__main__":
    json_file_path = os.environ.get("JSON_PATH")
    if json_file_path is None:
        print("Set JSON_PATH=/path/to/sim.json")
        sys.exit(1)
    pffdtd_method(json_file_path)
    
    # CHORAS pattern: notify backend (no-op when /receive endpoint is missing)
    try:
        from HelperFunctions import save_results
        save_results(json_file_path)
    except ImportError:
        pass
```

---

## 11. methods-config.json entry

Add to `simulation-backend/methods-config.json`:

```json
{
    "simulationType": "PFFDTD",
    "containerImage": "pffdtd_image:latest",
    "envVars": {},
    "label": "PPFFDTD (PFFDTD + POD-GP ROM)",
    "settings": "PFFDTD_setting.json",
    "entryFile": "PFFDTDInterface.py",
    "repositoryURL": "https://github.com/Burhanuddin98/PPFFDTD",
    "documentationURL": ""
}
```

This file is currently used for headless tooling, not for runtime dispatch (which is in-process). Still good citizenship.

---

## 12. Auralization

PPFFDTD writes `<json_path_minus_ext>_pressure.csv` (two columns: `t`, `pressure`). The backend's `auralization_calculation_DG` reads column index 1 (the `pressure` column), normalizes, optionally convolves with anechoic source, writes `<json_path_minus_ext>.wav`. **No new code in the backend** — just the match-block addition in §7.3.

---

## 13. End-to-end test plan

```bash
# 1. Build PPFFDTD container (headless dev test)
cd C:\RoomGUI\choras\backend\simulation-backend
docker build -t pffdtd_method_image -f pffdtd_method/Dockerfile .

# 2. Headless run
mkdir test
cp ../common/MeasurementRoom.{geo,msh} test/
cp <our exampleInput_PFFDTD.json> test/
docker run --rm -v ${PWD}/test:/test \
  -e JSON_PATH=/test/exampleInput_PFFDTD.json \
  pffdtd_method_image
# Verify: JSON now has parameters.t30 = list of 5, receiverResults populated, percentage=100

# 3. Patched-CHORAS test (in-process integration)
# (Apply backend patches to local C:\RoomGUI\choras\backend\)
cd C:\RoomGUI\choras
docker compose down
docker compose up
# Browser → http://localhost:5173
# Upload MeasurementRoom.obj
# In simulation creation: pick "PPFFDTD (PFFDTD + POD-GP ROM)"
# Set materials: "Upholstered concert chairs" all surfaces
# Run, observe Celery logs (docker compose logs -f backend)
# Verify completion: ResultParameters table populates per band, ResultPlots renders IR

# 4. Train -> use ROM round-trip
# First click: pffdtd_train_rom=true -> ~19 min
# Second click (different materials): pffdtd_use_rom=true -> <5 sec
```

---

## 14. Known gotchas

From the deep-read agents:

- `BOTH` taskType is **half-implemented**: `start_solver_task` queues both DE and DG tasks, but `run_solver.delay` is only called once per `simulationRun`, picking up the first `resultType`. PPFFDTD will not be affected unless we try to combine.
- `delete_simulation` deletes `SimulationRun` rows by `simulation.id`, not `simulationRunId` — orphans run rows.
- `datetime.now()` is evaluated at import in `Simulation.createdAt` default — every Simulation row created in a process gets the same timestamp until process restart.
- `eventlet` + SQLite Celery broker → concurrency races. Don't run multiple PPFFDTD jobs in parallel locally; they'll trample each other's pffdtd_data dirs.
- DE writes `parameters["edt"] = results["t20_band"]` (label mismatch; not our problem to fix).
- `MeasurementRoom.obj` uses `usemtl` not `g`. Don't assume `g` groups for surface naming.
- Issue templates link to `Building-acoustics-TU-Eindhoven/CHORAS` (stale; org renamed to `choras-org`).
- `/receive` endpoint **does not exist**. `save_results()`'s POST is a no-op. Backend re-reads JSON after task returns.

---

## 15. Decisions log

| Question | Decision | Rationale |
|---|---|---|
| ROM unification | Use `NonIntrusiveROM` (33-Smolyak GP); delete `_train_rom`/`_run_rom` (13-RBF) | README claims match GP path; RBF was an unvalidated parallel implementation |
| Integration delivery | Local patch files; hand-merge at TUE | No fork PRs needed; cleaner for live collaboration |
| Auralization | Reuse `auralization_calculation_DG` | PPFFDTD's IR shape mirrors DG's; no new auralization code |
| v1 scope | Per-band metrics + f-range gating only | Schema conformance first; differentiation features (GP-MC σ, Sobol) follow in v2 |
| Copier template | Do NOT adopt | 1-day-old, 0 adopters, ABC contract incompatibility |
| Frontend changes | None for v1 | Frontend is method-agnostic; existing `Parameters` interface accommodates our 7 fields |
