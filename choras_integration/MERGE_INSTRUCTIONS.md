# Merging PPFFDTD into CHORAS — step-by-step

This directory contains every file needed to integrate PPFFDTD as a CHORAS simulation method. All edits are additive (no destructive changes to existing methods). Total integration time: ~30 minutes if everything else is in place.

## Prerequisites

- CHORAS clone at `~/CHORAS` (with `backend/` and `frontend-v2/` submodules initialized)
- Local CHORAS running via `docker compose up`, or the dev environment described in `backend/docs/source/includes/installing_choras.md`
- The PPFFDTD repo: https://github.com/Burhanuddin98/PPFFDTD

## Step 1 — Drop files into `simulation-backend/`

```bash
cd <CHORAS>/backend/simulation-backend

# Method directory
mkdir -p pffdtd_method
cp <ppffdtd>/choras_integration/pffdtd_method/PFFDTDInterface.py  pffdtd_method/
cp <ppffdtd>/choras_integration/pffdtd_method/iso_metrics.py      pffdtd_method/
cp <ppffdtd>/choras_integration/pffdtd_method/Dockerfile          pffdtd_method/
cp <ppffdtd>/choras_integration/pffdtd_method/requirements.txt    pffdtd_method/

# Headless input example
cp <ppffdtd>/choras_integration/common/exampleInput_PFFDTD.json   common/
```

## Step 2 — Append to `simulation-backend/methods-config.json`

Open the file and add the entry from `backend_patches/methods-config.json.entry` to the JSON array. Final array should have DG, DE, MyNewMethod, and PFFDTD.

## Step 3 — Register module in `simulation_backend/__init__.py`

Add the import:

```python
from .PFFDTDInterface import pffdtd_method
```

(If the existing methods are imported here too. If not, the backend pulls them via direct import — check `app/services/simulation_service.py`'s import block to confirm pattern.)

## Step 4 — Drop files into `backend/`

```bash
cd <CHORAS>/backend

cp <ppffdtd>/choras_integration/backend_patches/PFFDTD_setting.json example_settings/
```

## Step 5 — Edit `backend/app/types/Task.py`

Add a single line per `backend_patches/Task.py.add`:

```python
class TaskType(Enum):
    GeometryCheck = "GeometryCheck"
    Mesh = "Mesh"
    DE = "DE"
    DG = "DG"
    MyNewMethod = "MyNewMethod"
    BOTH = "BOTH"
    PFFDTD = "PFFDTD"   # <-- ADD
```

## Step 6 — Edit `backend/app/services/simulation_service.py`

Apply the **5 additions** from `backend_patches/simulation_service.py.add`. They mirror the existing DG / DE / MyNewMethod cases:

1. In `start_solver_task`: PFFDTD branch alongside MyNewMethod
2. In `run_solver` body: `from simulation_backend.PFFDTDInterface import pffdtd_method`
3. In the solver `match taskType`: `case TaskType.PFFDTD: pffdtd_method(...)`
4. In the in-line auralization `match taskType` (DG-style): `case TaskType.PFFDTD: ... auralization_calculation_DG(...)`
5. In the post-hoc auralization service's `match taskType.value` (`auralization_service.run_auralization`): same DG-style auralization call

## Step 7 — Append to `backend/app/models/data/simulation_settings.json`

Add the entry from `backend_patches/simulation_settings.json.entry` to the JSON array. Final array should have DE, DG, and PFFDTD.

## Step 8 — Append to `backend/requirements.txt`

```
git+https://github.com/Burhanuddin98/PPFFDTD.git@main#egg=ppffdtd
```

(So that `from ppffdtd.rom import NonIntrusiveROM` resolves inside the backend container.)

## Step 9 — Reset DB and restart

```bash
cd <CHORAS>/backend
flask reset-db        # picks up the new TaskType + simulation_settings entry
cd ..
docker compose down
docker compose up
```

## Step 10 — Verify in browser

1. Open http://localhost:5173
2. Upload a geometry (e.g. `example_geometries/MeasurementRoom.obj`)
3. Create a simulation; the method dropdown should now include **PPFFDTD (PFFDTD + POD-GP ROM)**
4. Pick "Upholstered concert chairs" for all surfaces
5. Click Run; observe the Celery worker log: `[INFO] PFFDTD method` then `PFFDTD: c0=..., fmax=...`
6. After the run completes:
   - **ResultParameters table** populates per band (T30/EDT/C80/D50/TS/TS) with `null` above 0.9·fmax_grid (default = 900 Hz, so the 1 kHz and 2 kHz bands will be null at default settings — this is correct behavior)
   - **ResultPlots** renders the IR
   - **ImpulseResponsePlayer** plays the auralization WAV
7. To test the ROM round-trip:
   - Change `pffdtd_train_rom: true` in the settings panel; click Run. Wait ~19 min.
   - Then in a fresh simulation against the same geometry, set `pffdtd_use_rom: true` and change material absorptions. Click Run. <5 seconds end-to-end.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Method dropdown does not include PPFFDTD | DB not reset, or `simulation_settings.json` entry malformed | `flask reset-db`, then check the JSON file is valid |
| `ImportError: PFFDTDInterface` in Celery log | `simulation-backend` not pip-reinstalled into backend image | Rebuild the backend image: `docker compose build backend` |
| `ImportError: ppffdtd.rom` | `ppffdtd` not on requirements.txt of the backend | Confirm Step 8; rebuild backend image |
| Run hangs forever | TDR triggered on Windows | Apply TdrDelay=30 in registry per `Wayverb` notes |
| Per-band metrics all null | `pffdtd_fmax` too low — no band's center sits below 0.9·fmax_grid | Raise `pffdtd_fmax` to 1000+ Hz |
| Auralization CSV not found | PFFDTDInterface failed before writing the sidecar | Check Celery log for the actual exception above the auralization step |

## Roll-back

Every change is additive. To remove PPFFDTD:

1. Delete `simulation-backend/pffdtd_method/`
2. Delete `simulation-backend/common/exampleInput_PFFDTD.json`
3. Remove the PFFDTD entry from `simulation-backend/methods-config.json`
4. Remove `from .PFFDTDInterface import pffdtd_method` from `simulation_backend/__init__.py`
5. Revert the 5 additions in `app/services/simulation_service.py`
6. Remove `PFFDTD = "PFFDTD"` from `app/types/Task.py`
7. Remove the PFFDTD entry from `app/models/data/simulation_settings.json`
8. Delete `example_settings/PFFDTD_setting.json`
9. Remove the `git+https://...PPFFDTD` line from `requirements.txt`
10. `flask reset-db && docker compose down && docker compose up`
