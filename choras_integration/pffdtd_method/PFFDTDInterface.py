"""CHORAS interface for PPFFDTD.

PPFFDTD wraps Brian Hamilton's PFFDTD finite-difference time-domain solver
with a non-intrusive POD+Gaussian Process reduced-order model. Three modes:

    Mode                                        | Time         | Trigger
    ------------------------------------------- | ------------ | ----------------------------------
    Full FDTD                                   | minutes      | (default)
    Train ROM + run                             | ~19 min      | simulationSettings.pffdtd_train_rom
    Use cached ROM                              | <5 sec       | simulationSettings.pffdtd_use_rom

Implements the CHORAS SimulationMethod ABC. Mutates the input JSON in place
and writes a sidecar CSV for the backend's DG-style auralization handler.

Standalone:
    JSON_PATH=/path/to/sim.json python PFFDTDInterface.py

Production (in-process from CHORAS Celery worker):
    from simulation_backend.PFFDTDInterface import pffdtd_method
    pffdtd_method(json_file_path)
"""
import os
import sys
import json
import csv
import hashlib
import platform
from pathlib import Path

import numpy as np

# ── Path setup ──
# When installed via pip in the backend container, ppffdtd/ and pffdtd/ are
# importable by name. When running standalone or in a method-specific container,
# we add them to sys.path explicitly.
_THIS = Path(__file__).resolve().parent
for _candidate in [
    _THIS.parent / "pffdtd" / "python",     # PFFDTD's Python sources
    _THIS.parent / "ppffdtd",               # our ROM package
    _THIS.parent / "common",                # CHORAS SimulationMethod ABC
    _THIS,                                  # iso_metrics local module
]:
    if _candidate.is_dir():
        p = str(_candidate)
        if p not in sys.path:
            sys.path.insert(0, p)

try:
    from simulation_method_interface import SimulationMethod
except ImportError:
    class SimulationMethod:
        def run_simulation(self, json_file_path):
            raise NotImplementedError

try:
    from iso_metrics import compute_per_band_metrics
except ImportError:
    from .iso_metrics import compute_per_band_metrics


class PFFDTDMethod(SimulationMethod):
    """PPFFDTD CHORAS interface."""

    def run_simulation(self, json_file_path):
        json_path = Path(json_file_path)
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
        use_gpu   = bool(s.get("pffdtd_use_gpu", True))
        use_rom   = bool(s.get("pffdtd_use_rom", False))
        train_rom = bool(s.get("pffdtd_train_rom", False))

        h = c0 / (fmax * ppw)
        fmax_grid = c0 / (2 * h)

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
        print(f"PFFDTD: {len(abs_coeffs)} surfaces, src={src_xyz.tolist()}, {len(receivers)} receivers")

        # ── Cache key for ROM (geometry + source + receivers + baseline materials) ──
        cache_dir = self._cache_dir(json_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = self._cache_key(geo_path, msh_path, src_xyz, receivers, abs_coeffs, c0, fmax, ppw, ir_length)
        rom_path = cache_dir / f"rom_{cache_key}.npz"

        # ── Build PFFDTD model dir if we'll need to run FDTD or train ROM ──
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
            self._train_rom_via_validated_rom(work_dir, abs_coeffs, rom_path, use_gpu)

        # ── Inference ──
        ir = None
        if use_rom and rom_path.exists():
            print(f"PFFDTD: using cached ROM at {rom_path.name}")
            ir, fs_out = self._evaluate_rom(rom_path, abs_coeffs)
        else:
            print("PFFDTD: running full FDTD")
            ir, fs_out = self._run_fdtd(work_dir, use_gpu=use_gpu)

        # ── Per-band ISO 3382 metrics with f-range gating ──
        per_band = compute_per_band_metrics(ir, fs=fs_out, frequencies=frequencies, fmax_grid=fmax_grid)
        n_gated = sum(1 for v in per_band["t30"] if v is None)
        print(f"PFFDTD: metrics computed for {len(frequencies)} bands ({n_gated} gated above {0.9*fmax_grid:.0f} Hz)")

        # ── Write results back to JSON ──
        result_block["resultType"] = "PFFDTD"
        result_block["percentage"] = 100
        # For now, single receiver (frontend's hard 1-source/1-receiver limit). Loop when relaxed.
        for i, resp in enumerate(result_block["responses"]):
            if i == 0:
                resp["receiverResults"] = ir.tolist()
                resp["receiverResultsUncorrected"] = ir.tolist()
                resp["parameters"] = per_band
            else:
                resp["receiverResults"] = []
                resp["receiverResultsUncorrected"] = []
                resp["parameters"] = {k: [None] * len(frequencies) for k in per_band}

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
    # PFFDTD model build pipeline (ported from existing PFDTDInterface.py)
    # ──────────────────────────────────────────────────────────────────────

    def _build_pffdtd_setup(self, msh_path, geo_path, abs_coeffs, src_xyz, receivers,
                            h, c0, Tc, rh, ir_length, fmax, work_dir):
        """Convert CHORAS inputs to PFFDTD H5 data ready for SimEngine."""
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
                msh_out = save_dir / "room.msh"
                gmsh.write(str(msh_out))
            else:
                raise FileNotFoundError(f"Neither msh_path nor geo_path exists: {msh_path}, {geo_path}")

            mats_hash = self._extract_gmsh_model(abs_coeffs)
        finally:
            gmsh.finalize()

        # Write PFFDTD JSON model
        model_json = {
            "mats_hash": mats_hash,
            "sources": [{"xyz": src_xyz.tolist()}],
            "receivers": [{"xyz": r.tolist()} for r in receivers],
        }
        model_json_path = save_dir / "model_export.json"
        with open(model_json_path, "w") as f:
            json.dump(model_json, f, indent=2)

        # Fit materials -> DEF triplets
        mat_dir = save_dir / "materials"
        mat_dir.mkdir(exist_ok=True)
        mat_files = self._fit_materials(abs_coeffs, mat_dir)

        # PFFDTD setup
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
            Tc=Tc,
            rh=rh,
            fcc_flag=False,
            PPW=PPW,
            fmax=fmax,
            save_folder=str(save_dir),
            compress=0,
            draw_vox=False,
            Nprocs=Nprocs,
        )

    def _extract_gmsh_model(self, abs_coeffs):
        """Extract triangle mesh into PFFDTD-compatible mats_hash."""
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
            pts = []
            tris = []
            for entity_tag in entity_tags:
                elem_types, _, node_tags_list = gmsh.model.mesh.getElements(dim, entity_tag)
                for et, ntags in zip(elem_types, node_tags_list):
                    if et != 2:  # triangles only
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
                sides = [1] * len(tris)  # front face is the lossy side (interior normals point inward)
                color = default_colors[idx % len(default_colors)]
                mats_hash[name] = {
                    "pts": pts, "tris": tris, "color": color, "sides": sides,
                }
        return mats_hash

    def _fit_materials(self, abs_coeffs, mat_dir):
        """Fit comma-separated 5-band α to PFFDTD's 11-band DEF triplets."""
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
        """[125,250,500,1k,2k] -> [16,31.5,63,125,250,500,1k,2k,4k,8k,16k] by extrapolation."""
        a = np.asarray(alphas_5, dtype=float)
        out = np.zeros(11)
        out[0:4] = a[0]      # 16,31.5,63,125 -> use 125
        out[4] = a[1]        # 250
        out[5] = a[2]        # 500
        out[6] = a[3]        # 1000
        out[7] = a[4]        # 2000
        out[8:11] = a[4]     # 4k,8k,16k -> use 2000
        return out

    # ──────────────────────────────────────────────────────────────────────
    # ROM: wire to validated ppffdtd.rom.NonIntrusiveROM
    # ──────────────────────────────────────────────────────────────────────

    def _train_rom_via_validated_rom(self, work_dir, abs_coeffs_baseline, rom_path, use_gpu):
        """Train the validated 33-Smolyak GP ROM (replaces the deprecated 13-RBF path)."""
        from rom import NonIntrusiveROM

        # NonIntrusiveROM expects (data_dir, pffdtd_python_dir).
        # data_dir = the PFFDTD save_dir; pffdtd_python_dir = path to PFFDTD python sources
        pffdtd_python = str(Path(__file__).resolve().parent.parent / "pffdtd" / "python")
        rom = NonIntrusiveROM(str(work_dir), pffdtd_python)

        # Convert CHORAS comma-string -> 5-band float lists for the baseline
        baseline_alphas = {
            name: [float(x.strip()) for x in alpha_str.split(",")]
            for name, alpha_str in abs_coeffs_baseline.items()
        }
        rom.train(baseline_alphas, dim=3, level=2, use_gpu=use_gpu)
        rom.save(str(rom_path))
        print(f"PFFDTD: ROM trained and saved to {rom_path}")

    def _evaluate_rom(self, rom_path, abs_coeffs):
        """Evaluate the cached ROM at the requested materials."""
        from rom import NonIntrusiveROM

        rom = NonIntrusiveROM(str(rom_path.parent), "")  # paths not needed for evaluate after load
        rom.load(str(rom_path))

        # NonIntrusiveROM.evaluate expects scaling factors relative to baseline.
        # The baseline alphas were captured at train-time inside the ROM;
        # we compute scales = current_alphas / baseline_alphas (per surface, per band, then mean).
        # For v1 the parameter axis is 3D (floor / ceiling / walls); the helper inside
        # the ROM project handles this contraction.
        alpha_scales = self._abs_coeffs_to_3d_scales(abs_coeffs, rom)

        ir, _unc = rom.evaluate(alpha_scales)
        fs_out = float(getattr(rom, "fs_out", 48000.0))
        return np.asarray(ir, dtype=np.float64), fs_out

    @staticmethod
    def _abs_coeffs_to_3d_scales(abs_coeffs, rom):
        """Map current per-surface α (5-band CSV strings) to 3D Smolyak scales (floor, ceiling, walls).

        The training Smolyak grid is in log(scale) space relative to baseline. Surfaces with
        names containing 'floor' map to dim 0, 'ceil' to dim 1, anything else to dim 2.
        """
        baseline = getattr(rom, "training_params", None)
        if baseline is None:
            return np.array([1.0, 1.0, 1.0])

        scales = []
        for surf_name, alpha_str in abs_coeffs.items():
            alphas = np.array([float(x.strip()) for x in alpha_str.split(",")])
            mean_alpha = float(np.mean(alphas))
            scales.append((surf_name.lower(), mean_alpha))

        # Group means
        floor_vals = [a for n, a in scales if "floor" in n]
        ceil_vals  = [a for n, a in scales if ("ceil" in n) or ("plafond" in n)]
        wall_vals  = [a for n, a in scales if (("floor" not in n) and ("ceil" not in n) and ("plafond" not in n))]

        def _safe_mean(xs, default=1.0):
            return float(np.mean(xs)) if xs else default

        floor_alpha = _safe_mean(floor_vals)
        ceil_alpha  = _safe_mean(ceil_vals)
        wall_alpha  = _safe_mean(wall_vals)

        # Baseline mean per-group (rom does the actual scaling internally).
        # We pass a 3-vector of "current_alpha_per_group" — the rom stored its baseline
        # at train time and computes ratios.
        return np.array([floor_alpha, ceil_alpha, wall_alpha])

    # ──────────────────────────────────────────────────────────────────────
    # FDTD execution
    # ──────────────────────────────────────────────────────────────────────

    def _run_fdtd(self, save_dir, use_gpu=True):
        """Run PFFDTD's SimEngine with optional CuPy GPU path; returns (ir, fs_out)."""
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
                gpu_engine = self._import_gpu_engine()
                if gpu_engine and gpu_engine.HAS_GPU:
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

    @staticmethod
    def _import_gpu_engine():
        try:
            import gpu_engine  # type: ignore
            return gpu_engine
        except ImportError:
            return None

    # ──────────────────────────────────────────────────────────────────────
    # Cache plumbing
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _cache_dir(json_path):
        # /app/cache exists in the backend container if a volume is mounted; else fall back to per-uploads dir.
        cache_root = Path("/app/cache") if os.path.isdir("/app/cache") else json_path.parent / "pffdtd_cache"
        return cache_root

    @staticmethod
    def _cache_key(geo_path, msh_path, src_xyz, receivers, abs_coeffs, c0, fmax, ppw, ir_length):
        h = hashlib.sha256()
        h.update(str(geo_path).encode())
        h.update(str(msh_path).encode())
        h.update(np.asarray(src_xyz).tobytes())
        for r in receivers:
            h.update(np.asarray(r).tobytes())
        for name in sorted(abs_coeffs.keys()):
            h.update(name.encode())
            h.update(abs_coeffs[name].encode())
        h.update(f"{c0}|{fmax}|{ppw}|{ir_length}".encode())
        return h.hexdigest()[:16]


# ── Module-level entry function (matches CHORAS dg_method / de_method pattern) ──

def pffdtd_method(json_file_path):
    """Entry point imported by simulation_service.run_solver."""
    PFFDTDMethod().run_simulation(json_file_path)


if __name__ == "__main__":
    json_file_path = os.environ.get("JSON_PATH")
    if json_file_path is None:
        print("Set JSON_PATH=/path/to/sim.json to run.")
        sys.exit(1)
    print(f"Running PFFDTD method with JSON_PATH={json_file_path}")
    pffdtd_method(json_file_path)

    # CHORAS pattern: notify backend (no-op when /receive endpoint is missing).
    try:
        from HelperFunctions import save_results
        save_results(json_file_path)
    except ImportError:
        print("(HelperFunctions not on path — standalone mode.)")
