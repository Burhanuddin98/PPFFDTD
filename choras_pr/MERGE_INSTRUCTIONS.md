# Merging PPFFDTD into CHORAS via the Copier scaffold

This is the **Copier-scaffolded** integration package — the path Silvin promotes in the official CHORAS Developer Workshop (slides 5–11). It replaces the earlier `choras_integration/` tree (which followed the older DG/DE pattern); both work, but this one matches the documented contribution workflow.

Repository layout produced here:

```
choras_pr/
├── pffdtd_method/                              # Copier scaffold output — drops into
│   │                                             simulation-backend/pffdtd_method/
│   ├── Dockerfile                              # python:3.11.13-slim, full deps
│   ├── pyproject.toml                          # PFFDTD + ppffdtd via git+https
│   ├── LICENSE
│   ├── pffdtd_interface/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── __cli__.py                          # reads JSON_PATH env var
│   │   ├── definition.py                       # ABC: run_simulation() parameterless
│   │   ├── pffdtd_interface.py                 # PFFDTDMethod class (our refactor)
│   │   └── iso_metrics.py                      # per-band Butterworth + Schroeder
│   └── tests/
│       ├── conftest.py
│       ├── test_definition.py                  # tests the ABC
│       ├── test_pffdtd_cli.py                  # CLI smoke test, skips if heavy deps missing
│       ├── test_fixtures.py
│       ├── test_input_pffdtd.json              # CHORAS-conformant input
│       ├── test_room_pffdtd.geo
│       └── test_room_pffdtd.msh
│
├── backend_patches/                            # diffs to apply in choras-org/backend
│   ├── Task.py.add                             # PFFDTD enum value
│   ├── simulation_service.py.add               # 5 dispatch points
│   ├── simulation_settings.json.entry          # method registry
│   └── PFFDTD_setting.json                     # UI form (9 controls)
│
└── orchestrator_patches/                       # diffs to apply in choras-org/CHORAS
    ├── methods-config.json.entry               # simulation-backend method registry
    ├── docker-compose.yml.snippet              # pffdtd_method service + pffdtd_cache volume
    └── CHORAS_BUILD.sh.snippet                 # docker build line for pffdtd_image
```

## Prerequisites

- CHORAS clone at `<your-CHORAS-dir>` with `backend/` and `frontend-v2/` submodules initialized (`git submodule update --init --recursive`).
- Docker Desktop running.

## Step 1 — Drop the Copier scaffold into simulation-backend

```bash
cd <your-CHORAS-dir>/backend/simulation-backend
cp -r <ppffdtd>/choras_pr/pffdtd_method ./
```

## Step 2 — Append to `simulation-backend/methods-config.json`

Open the file and add the entry from [orchestrator_patches/methods-config.json.entry](orchestrator_patches/methods-config.json.entry) to the JSON array.

## Step 3 — Apply backend patches

```bash
cd <your-CHORAS-dir>/backend
```

Then:
- Edit `app/types/Task.py` — add `PFFDTD = "PFFDTD"` (see [backend_patches/Task.py.add](backend_patches/Task.py.add))
- Edit `app/services/simulation_service.py` — apply the 5 additions in [backend_patches/simulation_service.py.add](backend_patches/simulation_service.py.add). The import line inside `run_solver` is:
  ```python
  from simulation_backend.pffdtd_method.pffdtd_interface import pffdtd_method
  ```
- Append to `app/models/data/simulation_settings.json` per [backend_patches/simulation_settings.json.entry](backend_patches/simulation_settings.json.entry).
- Copy `backend_patches/PFFDTD_setting.json` into `example_settings/`.

## Step 4 — Apply orchestrator patches

```bash
cd <your-CHORAS-dir>
```

Then:
- Edit `docker-compose.yml` — add the `pffdtd_method` service per [orchestrator_patches/docker-compose.yml.snippet](orchestrator_patches/docker-compose.yml.snippet), and add `pffdtd_cache:` to the `volumes:` section.
- Edit `CHORAS_BUILD.sh` — append the docker-build stanza from [orchestrator_patches/CHORAS_BUILD.sh.snippet](orchestrator_patches/CHORAS_BUILD.sh.snippet).

## Step 5 — Build the image and seed the DB

```bash
bash CHORAS_BUILD.sh          # builds pffdtd_image:latest (plus existing dg/de)
cd backend
flask reset-db                # or `flask update-setting` if you want to keep existing data
cd ..
docker compose down
docker compose up
```

## Step 6 — Verify in the UI

Open http://localhost:5173 → create a project → upload `example_geometries/MeasurementRoom.obj` → pick **PPFFDTD (PFFDTD + POD-GP ROM)** as the method → pick materials → Run.

End-to-end success criteria:

- Method dropdown contains PPFFDTD ✓
- Settings panel shows 9 controls (c0, fmax, ppw, ir_length, temperature, humidity, three Yes/No radios for use_gpu/train_rom/use_rom) ✓
- Default run (Train=No, Use=No) takes ~1 minute of FDTD on CPU; result has per-band metrics, EDC curves under `receiverResultsEDC`, raw IR under `receiverResults` ✓
- Train run (Train=Yes) shows progress bar advancing in 33 steps; ~20–35 min wall time depending on CPU ✓
- Cached use (Train=No, Use=Yes) on the same model returns in <10 s solver time ✓
- Auralization Play button works ✓

## Notes for the maintainer

- **ABC contract**: this scaffold uses the *new* contract from `template_simulation_method` — `__init__(input_json_path)` + parameterless `run_simulation()` + inherited `save_results()`. A backward-compatible `pffdtd_method(json_file_path)` wrapper is exported from `pffdtd_interface.__init__` so the existing `simulation_service.run_solver` dispatch pattern keeps working unchanged.
- **`receiverResults`** is the raw IR per slide 10 ("It's best if a Room Impulse Response is passed"). EDC curves go into a separate `receiverResultsEDC` field for the Plots tab to consume. If the frontend selector still expects EDC dicts in `receiverResults`, we can flip the assignment in [pffdtd_interface.py](pffdtd_method/pffdtd_interface/pffdtd_interface.py).
- **ROM cache key** excludes materials, so Run 2 with different absorption coefficients hits the same trained ROM and only re-evaluates the GPs.
- **Operating window**: the ROM is valid for materials in [0.3, 3.0] × the training-time baseline. Outside that we clip to the boundary.
- **Validation**: see `docs/ROM_VERIFICATION_SUMMARY.md` and `docs/figures/` for the 5-step verification suite (POD spectrum, LOO, unseen-point, GP calibration, Smolyak L1→L2 convergence) with reproducible scripts under `validation/`.
