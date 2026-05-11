"""CHORAS interface for PPFFDTD — PFFDTD with a non-intrusive POD+GP ROM.

PPFFDTD wraps Brian Hamilton's PFFDTD wave solver with a reduced-order model
built on a Smolyak sparse grid in material-scale space, POD compression, and
Gaussian-Process interpolation. Three modes:

    Mode                                        | Time         | Trigger
    ------------------------------------------- | ------------ | ----------------------------------
    Full FDTD                                   | minutes      | default (pffdtd_use_rom = "no")
    Train ROM + run                             | ~20 min      | pffdtd_train_rom = "yes"
    Use cached ROM                              | <5 sec       | pffdtd_use_rom = "yes"

Implements the CHORAS SimulationMethod ABC (new contract: path passed at
__init__, run_simulation() is parameterless). Mutates the input JSON in place
and writes a sidecar CSV for the backend's DG-style auralization handler.
"""
import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

from .definition import SimulationMethod

# Production-container install paths for PFFDTD's Python sources and our ROM package.
# When the backend image is built per Silvin's contribution workflow, these resolve via
# pip from pyproject.toml. When deployed alongside the scaffold (PR drop-in), they
# come from the simulation-backend container's pip install of our package + its deps.
_PROD_PFFDTD_PY = Path("/app/pffdtd_python")
_PROD_PPFFDTD   = Path("/app/ppffdtd_pkg")
for _cand in (_PROD_PFFDTD_PY, _PROD_PPFFDTD):
    if _cand.is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

try:
    from .iso_metrics import compute_per_band_metrics, _bandpass, _edc_db
except ImportError:
    from iso_metrics import compute_per_band_metrics, _bandpass, _edc_db


def _truthy(v, default=False):
    """Accept multiple truthy shapes the frontend may serialize."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("yes", "true", "on", "1")
    if isinstance(v, (list, tuple)):
        return bool(v)
    return bool(v)


class PFFDTDMethod(SimulationMethod):
    """PPFFDTD CHORAS interface (new ABC contract)."""

    def __init__(self, input_json_path):
        super().__init__(input_json_path)

    def run_simulation(self):
        """Entry point invoked by the CHORAS backend / CLI.

        Path is already validated by the base class and stored at
        ``self.input_json_path``. Mutates that JSON in place.
        """
        json_path = Path(self.input_json_path)
        with open(json_path, "r") as f:
            config = json.load(f)

        # ── Settings ──
        s = config.get("simulationSettings", {})
        c0        = float(s.get("pffdtd_c0", 343.0))
        fmax      = float(s.get("pffdtd_fmax", 1000.0))
        ppw       = int(s.get("pffdtd_ppw", 6))
        ir_length = float(s.get("pffdtd_ir_length", 1.0))
        Tc        = float(s.get("pffdtd_temperature", 20.0))
        rh        = float(s.get("pffdtd_humidity", 50.0))
        use_gpu   = _truthy(s.get("pffdtd_use_gpu"),   default=True)
        use_rom   = _truthy(s.get("pffdtd_use_rom"),   default=False)
        train_rom = _truthy(s.get("pffdtd_train_rom"), default=False)

        h = c0 / (fmax * ppw)
        fmax_grid = c0 / (2 * h)
        # Gate metrics above 0.9 * USER-REQUESTED fmax (not grid Nyquist) — bands in
        # (fmax, fmax_grid) have severe FDTD numerical dispersion.
        f_gate = 0.9 * fmax

        # ── Geometry ──
        geo_path = config.get("geo_path", "")
        msh_path = config.get("msh_path", "")
        if geo_path and not os.path.isabs(geo_path):
            geo_path = str(json_path.parent / geo_path)
        if msh_path and not os.path.isabs(msh_path):
            msh_path = str(json_path.parent / msh_path)

        # ── Materials ──
        abs_coeffs = config.get("absorption_coefficients", {})

        # ── Source / receivers ──
        result_block = config["results"][0]
        src_xyz = np.array([
            float(result_block["sourceX"]),
            float(result_block["sourceY"]),
            float(result_block["sourceZ"]),
        ])
        receivers = []
        for resp in result_block["responses"]:
            receivers.append(np.array([float(resp["x"]), float(resp["y"]), float(resp["z"])]))

        frequencies = result_block.get("frequencies", [125, 250, 500, 1000, 2000])

        print(f"PFFDTD: c0={c0}, fmax={fmax}, h={h:.4f}m, fmax_grid={fmax_grid:.1f}Hz, "
              f"IR={ir_length}s, GPU={use_gpu}, ROM(use={use_rom}, train={train_rom})")
        print(f"PFFDTD: {len(abs_coeffs)} surfaces, src={src_xyz.tolist()}, "
              f"{len(receivers)} receivers, gate at {f_gate:.0f} Hz")

        # ── ROM cache key (geometry + source + receivers + grid params only) ──
        cache_dir = self._cache_dir(json_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._cache_key(geo_path, msh_path, src_xyz, receivers, c0, fmax, ppw, ir_length)
        rom_path = cache_dir / f"rom_{cache_key}.npz"

        work_dir = json_path.parent / "pffdtd_data"
        need_fdtd_setup = train_rom or not (use_rom and rom_path.exists())
        if need_fdtd_setup:
            self._build_pffdtd_setup(
                msh_path, geo_path, abs_coeffs, src_xyz, receivers,
                h, c0, Tc, rh, ir_length, fmax, work_dir,
            )

        # ── Train ROM if requested ──
        if train_rom:
            print("PFFDTD: training ROM (33-point Smolyak L2 in 3D)...")
            self._train_rom_via_validated_rom(work_dir, abs_coeffs, rom_path, use_gpu, json_path)

        # ── Inference ──
        ir = None
        if use_rom and rom_path.exists():
            print(f"PFFDTD: using cached ROM at {rom_path.name}")
            baseline_sidecar = Path(str(rom_path) + ".baseline.json")
            baseline_alphas = None
            if baseline_sidecar.exists():
                try:
                    with open(baseline_sidecar) as f:
                        baseline_alphas = json.load(f).get("baseline_alphas")
                    print(f"PFFDTD: loaded baseline materials from {baseline_sidecar.name}")
                except Exception as e:
                    print(f"PFFDTD: sidecar unreadable ({e}); assuming current==baseline")
            ir, fs_out = self._evaluate_rom(rom_path, abs_coeffs, baseline_alphas)
        else:
            print("PFFDTD: running full FDTD")
            ir, fs_out = self._run_fdtd(work_dir, use_gpu=use_gpu)

        # ── Per-band ISO 3382 metrics + per-band EDC curves ──
        per_band = compute_per_band_metrics(ir, fs=fs_out, frequencies=frequencies, fmax_grid=fmax / 0.9)
        n_gated = sum(1 for v in per_band["t30"] if v is None)
        print(f"PFFDTD: metrics computed for {len(frequencies)} bands ({n_gated} gated above {f_gate:.0f} Hz)")

        # Per-band EDC for the Plots tab — extension field separate from receiverResults.
        per_band_edc = []
        for fc in frequencies:
            if fc > f_gate:
                continue
            ir_band = _bandpass(np.asarray(ir, dtype=np.float64), fs_out, fc)
            edc = _edc_db(ir_band)
            stride = max(1, int(round(fs_out * 0.005)))
            t_full = np.arange(len(edc)) / fs_out
            per_band_edc.append({
                "data": edc[::stride].astype(np.float32).tolist(),
                "t":    t_full[::stride].astype(np.float32).tolist(),
                "frequency": int(fc),
                "type": "edc",
            })

        # ── Write results back to JSON ──
        # Per workshop slide 10: "It's best if a Room Impulse Response is passed
        # (receiverResults)." So receiverResults = raw IR (canonical). EDCs go in
        # an extension field.
        result_block["resultType"] = "PFFDTD"
        result_block["percentage"] = 100
        for i, resp in enumerate(result_block["responses"]):
            if i == 0:
                resp["receiverResults"] = ir.tolist()
                resp["receiverResultsUncorrected"] = ir.tolist()
                resp["parameters"] = per_band
                resp["receiverResultsEDC"] = per_band_edc
            else:
                resp["receiverResults"] = []
                resp["receiverResultsUncorrected"] = []
                resp["parameters"] = {k: [None] * len(frequencies) for k in per_band}
                resp["receiverResultsEDC"] = []

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        # ── Auralization sidecar (DG-style two-column CSV) ──
        csv_path = str(json_path).replace(".json", "_pressure.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "pressure"])
            t = np.arange(len(ir)) / fs_out
            for ti, pi in zip(t, ir):
                w.writerow([f"{ti:.6f}", f"{pi:.9f}"])
        print(f"PFFDTD: results written to {json_path}")
        print(f"PFFDTD: auralization CSV written to {csv_path}")

    # ──────────────────────────────────────────────────────────────────────
    # PFFDTD model build pipeline
    # ──────────────────────────────────────────────────────────────────────

    def _build_pffdtd_setup(self, msh_path, geo_path, abs_coeffs, src_xyz, receivers,
                            h, c0, Tc, rh, ir_length, fmax, work_dir):
        import gmsh

        save_dir = Path(work_dir)
        save_dir.mkdir(exist_ok=True)

        gmsh.initialize()
        try:
            if msh_path and os.path.exists(msh_path):
                gmsh.open(msh_path)
            elif geo_path and os.path.exists(geo_path):
                gmsh.open(geo_path)
                gmsh.model.mesh.generate(2)
                gmsh.write(str(save_dir / "room.msh"))
            else:
                raise FileNotFoundError(f"Neither msh_path nor geo_path exists: {msh_path}, {geo_path}")
            mats_hash = self._extract_gmsh_model(abs_coeffs)
        finally:
            gmsh.finalize()

        model_json = {
            "mats_hash": mats_hash,
            "sources": [{"xyz": src_xyz.tolist()}],
            "receivers": [{"xyz": r.tolist()} for r in receivers],
        }
        model_json_path = save_dir / "model_export.json"
        with open(model_json_path, "w") as f:
            json.dump(model_json, f, indent=2)

        mat_dir = save_dir / "materials"
        mat_dir.mkdir(exist_ok=True)
        mat_files = self._fit_materials(abs_coeffs, mat_dir)

        from sim_setup import sim_setup

        mat_files_dict = {name: os.path.basename(path) for name, path in mat_files.items()}
        Nprocs = 1 if platform.system() == "Windows" else None
        PPW = round(c0 / (fmax * h))

        sim_setup(
            model_json_file=str(model_json_path),
            mat_folder=str(mat_dir),
            mat_files_dict=mat_files_dict,
            source_num=1,
            insig_type="dhann30",
            diff_source=False,
            duration=ir_length,
            Tc=Tc, rh=rh,
            fcc_flag=False, PPW=PPW, fmax=fmax,
            save_folder=str(save_dir),
            compress=0, draw_vox=False, Nprocs=Nprocs,
        )

    def _extract_gmsh_model(self, abs_coeffs):
        import gmsh
        phys_groups = gmsh.model.getPhysicalGroups(2)
        mats_hash = {}
        default_colors = [
            [180, 180, 180], [200, 100, 100], [100, 200, 100], [100, 100, 200],
            [200, 200, 100], [200, 100, 200], [100, 200, 200], [150, 150, 150],
        ]
        for idx, (dim, phys_tag) in enumerate(phys_groups):
            name = gmsh.model.getPhysicalName(dim, phys_tag).split("$")[0]
            entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
            all_nodes = {}
            pts, tris = [], []
            for entity_tag in entity_tags:
                elem_types, _, node_tags_list = gmsh.model.mesh.getElements(dim, entity_tag)
                for et, ntags in zip(elem_types, node_tags_list):
                    if et != 2:
                        continue
                    for tri_nodes in ntags.reshape(-1, 3):
                        tri = []
                        for nid in tri_nodes:
                            if nid not in all_nodes:
                                coord, *_ = gmsh.model.mesh.getNode(int(nid))
                                all_nodes[nid] = len(pts)
                                pts.append(coord.tolist())
                            tri.append(all_nodes[nid])
                        tris.append(tri)
            if tris:
                mats_hash[name] = {
                    "pts": pts, "tris": tris,
                    "color": default_colors[idx % len(default_colors)],
                    "sides": [1] * len(tris),
                }
        return mats_hash

    def _fit_materials(self, abs_coeffs, mat_dir):
        from materials.adm_funcs import fit_to_Sabs_oct_11
        mat_files = {}
        for i, (name, alpha_str) in enumerate(abs_coeffs.items()):
            alphas_5 = [float(x.strip()) for x in alpha_str.split(",")]
            alphas_11 = self._expand_5_to_11_bands(alphas_5)
            alphas_11 = np.clip(alphas_11, 0.01, 0.99)
            mat_file = str(mat_dir / f"mat_{i:02d}.h5")
            fit_to_Sabs_oct_11(alphas_11, mat_file)
            mat_files[name] = mat_file
        return mat_files

    @staticmethod
    def _expand_5_to_11_bands(alphas_5):
        a = np.asarray(alphas_5, dtype=float)
        out = np.zeros(11)
        out[0:4] = a[0]; out[4] = a[1]; out[5] = a[2]
        out[6] = a[3]; out[7] = a[4]; out[8:11] = a[4]
        return out

    # ──────────────────────────────────────────────────────────────────────
    # ROM: train + evaluate
    # ──────────────────────────────────────────────────────────────────────

    def _train_rom_via_validated_rom(self, work_dir, abs_coeffs_baseline, rom_path, use_gpu, json_path):
        """Train 33-Smolyak GP ROM. Writes percentage to JSON per fold for the CHORAS
        progress-bar (workshop slide 10)."""
        from rom import NonIntrusiveROM

        pffdtd_python = str(_PROD_PFFDTD_PY) if _PROD_PFFDTD_PY.is_dir() else ""
        rom = NonIntrusiveROM(str(work_dir), pffdtd_python)

        TOTAL_FOLDS = 33
        fold_idx = [0]
        orig = rom._run_single_fdtd

        def _wrapped(DEF, use_gpu_inner=False):
            ir = orig(DEF, use_gpu=use_gpu_inner)
            fold_idx[0] += 1
            if json_path is not None:
                pct = min(95, int(95 * fold_idx[0] / TOTAL_FOLDS))
                try:
                    with open(json_path, "r") as f:
                        cfg = json.load(f)
                    if cfg.get("results"):
                        cfg["results"][0]["percentage"] = pct
                    with open(json_path, "w") as f:
                        json.dump(cfg, f, indent=4)
                    print(f"PFFDTD: progress {pct}% ({fold_idx[0]}/{TOTAL_FOLDS} folds)")
                except Exception as e:
                    print(f"PFFDTD: progress-write failed ({e})")
            return ir

        rom._run_single_fdtd = _wrapped
        try:
            rom.train(abs_coeffs_baseline, dim=3, level=2, use_gpu=use_gpu)
        finally:
            rom._run_single_fdtd = orig

        rom.save(str(rom_path))
        baseline_sidecar = Path(str(rom_path) + ".baseline.json")
        with open(baseline_sidecar, "w") as f:
            json.dump({"baseline_alphas": abs_coeffs_baseline}, f, indent=2)
        print(f"PFFDTD: ROM saved to {rom_path}")
        print(f"PFFDTD: baseline saved to {baseline_sidecar.name}")

    def _evaluate_rom(self, rom_path, abs_coeffs, baseline_alphas=None):
        from rom import NonIntrusiveROM
        rom = NonIntrusiveROM.__new__(NonIntrusiveROM)
        rom.data_dir = Path(rom_path).parent
        rom.pffdtd_dir = Path("")
        rom.load(str(rom_path))

        scales = self._mean_alpha_scale(abs_coeffs, baseline_alphas, int(rom.dim))
        print(f"PFFDTD: ROM evaluate at scales = {scales.tolist()}")
        ir, unc = rom.evaluate(scales)
        print(f"PFFDTD: ROM eval done; mean GP uncertainty = {unc:.4f}")
        return np.asarray(ir, dtype=np.float64), float(getattr(rom, "fs_out", 48000.0))

    @staticmethod
    def _mean_alpha_scale(current_abs, baseline_alphas, n_dim):
        def _mean_of(d):
            vals = []
            for v in d.values():
                if isinstance(v, str):
                    vals.extend(float(x.strip()) for x in v.split(","))
                elif isinstance(v, (list, tuple)):
                    vals.extend(float(x) for x in v)
            return float(np.mean(vals)) if vals else 1.0
        cur = _mean_of(current_abs)
        base = _mean_of(baseline_alphas) if baseline_alphas else cur
        if base <= 1e-6:
            return np.ones(n_dim)
        return np.full(n_dim, float(np.clip(cur / base, 0.30, 3.00)))

    # ──────────────────────────────────────────────────────────────────────
    # FDTD execution
    # ──────────────────────────────────────────────────────────────────────

    def _run_fdtd(self, save_dir, use_gpu=True):
        from fdtd.sim_fdtd import SimEngine
        from fdtd.process_outputs import ProcessOutputs

        engine = SimEngine(str(save_dir), energy_on=False)
        engine.load_h5_data()
        engine.setup_mask()
        engine.allocate_mem()
        engine.set_coeffs()
        engine.checks()

        gpu_ok = False
        if use_gpu:
            try:
                import gpu_engine
                if gpu_engine.HAS_GPU:
                    print("PFFDTD: using GPU engine (CuPy RawKernel)")
                    gpu_engine.run_gpu(engine)
                    gpu_ok = True
            except Exception as e:
                print(f"PFFDTD: GPU init failed ({e}), falling back to CPU")

        if not gpu_ok:
            print("PFFDTD: using CPU engine (numba)")
            engine.run_all()
        engine.save_outputs()

        po = ProcessOutputs(str(save_dir))
        po.initial_process(fcut=10.0, N_order=4)
        fmax_grid = engine.c / (2 * engine.h)
        po.apply_lowpass(fcut=0.9 * fmax_grid, N_order=8)
        po.resample(Fs_f=48000)

        r_out = po.r_out_f
        ir = r_out if r_out.ndim == 1 else r_out[0]
        return np.asarray(ir, dtype=np.float64), 48000.0

    # ──────────────────────────────────────────────────────────────────────
    # Cache plumbing
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_dir(json_path):
        return Path("/app/cache") if os.path.isdir("/app/cache") else json_path.parent / "pffdtd_cache"

    @staticmethod
    def _cache_key(geo_path, msh_path, src_xyz, receivers, c0, fmax, ppw, ir_length):
        """ROM cache key — geometry + source + receivers + grid params ONLY.
        Materials excluded so Run 2 with different α hits the same cached ROM."""
        h = hashlib.sha256()
        h.update(str(geo_path).encode())
        h.update(str(msh_path).encode())
        h.update(np.asarray(src_xyz).tobytes())
        for r in receivers:
            h.update(np.asarray(r).tobytes())
        h.update(f"{c0}|{fmax}|{ppw}|{ir_length}".encode())
        return h.hexdigest()[:16]


# Backward-compat wrapper kept so the existing simulation_service dispatch pattern
# (`case TaskType.PFFDTD: pffdtd_method(json_file_path=json_path)`) keeps working.
def pffdtd_method(json_file_path):
    PFFDTDMethod(input_json_path=json_file_path).run_simulation()
