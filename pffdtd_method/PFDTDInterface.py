"""
CHORAS interface for PFFDTD (Pretty Fast FDTD).

Implements the SimulationMethod interface required by CHORAS.
Reads the CHORAS JSON, converts geometry + materials to PFFDTD format,
runs the FDTD simulation, and writes the impulse response back to JSON.

Usage (standalone):
    python PFDTDInterface.py  # reads JSON_PATH env var

Usage (from CHORAS backend):
    method = PFDTDMethod()
    method.run_simulation("path/to/simulation.json")
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add PFFDTD submodule to path
PFFDTD_PYTHON = str(Path(__file__).parent.parent / "pffdtd" / "python")
if PFFDTD_PYTHON not in sys.path:
    sys.path.insert(0, PFFDTD_PYTHON)

# Add common (for SimulationMethod base class)
COMMON_DIR = str(Path(__file__).parent.parent / "common")
if os.path.isdir(COMMON_DIR):
    sys.path.insert(0, COMMON_DIR)

try:
    from simulation_method_interface import SimulationMethod
except ImportError:
    # Standalone mode — define stub
    class SimulationMethod:
        def run_simulation(self, json_file_path):
            raise NotImplementedError


class PFDTDMethod(SimulationMethod):
    """PFFDTD solver for CHORAS."""

    def __init__(self):
        super().__init__()

    def run_simulation(self, json_file_path: str):
        """Run PFFDTD simulation from CHORAS JSON.

        1. Parse JSON (geometry, materials, source/receiver positions)
        2. Mesh with Gmsh → export triangles
        3. Voxelize for FDTD
        4. Fit absorption → DEF triplets
        5. Run PFFDTD engine
        6. Post-process (filter, resample)
        7. Write IR back to JSON
        """
        print("PFFDTD: starting simulation")
        json_path = Path(json_file_path)

        with open(json_path, "r") as f:
            config = json.load(f)

        # ── Parse settings ──
        settings = config.get("simulationSettings", {})
        c0 = settings.get("pffdtd_c0", 343.0)
        fmax = settings.get("pffdtd_fmax", 1000.0)
        ppw = settings.get("pffdtd_ppw", 6)  # points per wavelength
        ir_length = settings.get("pffdtd_ir_length", 1.0)  # seconds
        Tc = settings.get("pffdtd_temperature", 20.0)
        rh = settings.get("pffdtd_humidity", 50.0)

        # Grid spacing from PPW and fmax
        h = c0 / (fmax * ppw)

        # ── Geometry ──
        geo_path = config.get("geo_path", "")
        msh_path = config.get("msh_path", "")

        if not os.path.isabs(msh_path):
            msh_path = str(json_path.parent / msh_path)
        if not os.path.isabs(geo_path):
            geo_path = str(json_path.parent / geo_path)

        # ── Materials (absorption per surface) ──
        abs_coeffs = config.get("absorption_coefficients", {})
        # CHORAS format: {"floor": "0.6, 0.69, 0.71, 0.7, 0.63", ...}
        # 5 values = octave bands [125, 250, 500, 1000, 2000] Hz

        # ── Source / receivers ──
        result_block = config["results"][0]
        src_xyz = np.array([
            result_block["sourceX"],
            result_block["sourceY"],
            result_block["sourceZ"]
        ])
        receivers = []
        for resp in result_block["responses"]:
            receivers.append(np.array([resp["x"], resp["y"], resp["z"]]))

        print(f"PFFDTD: c0={c0}, fmax={fmax}, h={h:.4f}m, "
              f"IR={ir_length}s, src={src_xyz}")
        print(f"PFFDTD: {len(abs_coeffs)} surfaces, "
              f"{len(receivers)} receivers")

        # ── Step 1: Build PFFDTD model from Gmsh mesh ──
        model = self._build_model(
            msh_path, geo_path, abs_coeffs, src_xyz, receivers,
            h, c0, Tc, rh, ir_length, fmax, json_path.parent
        )

        # ── Step 2: Run PFFDTD ──
        irs = self._run_fdtd(model)

        # ── Step 3: Write results back to JSON ──
        self._write_results(config, irs, json_file_path)

        print("PFFDTD: simulation done!")

    def _build_model(self, msh_path, geo_path, abs_coeffs, src_xyz,
                     receivers, h, c0, Tc, rh, ir_length, fmax, work_dir):
        """Convert CHORAS inputs to PFFDTD H5 data."""
        import tempfile
        import gmsh

        save_dir = Path(work_dir) / "pffdtd_data"
        save_dir.mkdir(exist_ok=True)

        # ── Mesh the geometry with Gmsh ──
        gmsh.initialize()
        if os.path.exists(msh_path):
            gmsh.open(msh_path)
        elif os.path.exists(geo_path):
            gmsh.open(geo_path)
            gmsh.model.mesh.generate(2)
            gmsh.write(str(save_dir / "room.msh"))
            msh_path = str(save_dir / "room.msh")
        else:
            gmsh.finalize()
            raise FileNotFoundError(f"Neither {msh_path} nor {geo_path} found")

        # Extract triangles and physical groups
        model_data = self._extract_gmsh_model(abs_coeffs)
        gmsh.finalize()

        # ── Build PFFDTD JSON model (same format as SketchUp export) ──
        pffdtd_json = self._to_pffdtd_json(
            model_data, src_xyz, receivers, abs_coeffs)
        json_model_path = save_dir / "model_export.json"
        with open(json_model_path, "w") as f:
            json.dump(pffdtd_json, f, indent=2)

        # ── Fit materials to DEF triplets ──
        mat_dir = save_dir / "materials"
        mat_dir.mkdir(exist_ok=True)
        mat_files = self._fit_materials(abs_coeffs, mat_dir)

        # ── Run PFFDTD setup (voxelizer + signals) ──
        self._run_setup(
            json_model_path, mat_files, save_dir,
            h, c0, Tc, rh, ir_length, fmax
        )

        return save_dir

    def _extract_gmsh_model(self, abs_coeffs):
        """Extract triangle mesh from Gmsh into PFFDTD-compatible format.

        Returns mats_hash with pts, tris, color, sides per surface.
        The 'sides' field controls PFFDTD's one-sided boundary model:
          +1 = front face is lossy (normal pointing inward)
          -1 = back face is lossy
           0 = rigid (unmarked)
        """
        import gmsh

        phys_groups = gmsh.model.getPhysicalGroups(2)
        mats_hash = {}

        # Default colors for surfaces
        default_colors = [
            [180, 180, 180], [200, 100, 100], [100, 200, 100],
            [100, 100, 200], [200, 200, 100], [200, 100, 200],
            [100, 200, 200], [150, 150, 150],
        ]

        for idx, (dim, phys_tag) in enumerate(phys_groups):
            name = gmsh.model.getPhysicalName(dim, phys_tag)
            name = name.split("$")[0]

            # Get entities in this physical group
            entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)

            all_nodes = {}
            pts = []
            tris = []

            for entity_tag in entity_tags:
                elem_types, elem_tags_list, node_tags_list = \
                    gmsh.model.mesh.getElements(dim, entity_tag)

                for et, ntags in zip(elem_types, node_tags_list):
                    if et != 2:  # triangles only
                        continue
                    ntags = ntags.reshape(-1, 3)
                    for tri_nodes in ntags:
                        tri = []
                        for nid in tri_nodes:
                            if nid not in all_nodes:
                                coord, _, _, _ = gmsh.model.mesh.getNode(int(nid))
                                all_nodes[nid] = len(pts)
                                pts.append(coord.tolist())
                            tri.append(all_nodes[nid])
                        tris.append(tri)

            if tris:
                # Sides: +1 means the front face (outward normal) is the lossy side
                # For room interiors, normals point inward, so sides = -1
                n_tris = len(tris)
                sides = np.ones(n_tris, dtype=int).tolist()  # default: front lossy
                color = default_colors[idx % len(default_colors)]

                mats_hash[name] = {
                    "pts": pts,
                    "tris": tris,
                    "color": color,
                    "sides": sides,
                }

        return mats_hash

    def _to_pffdtd_json(self, mats_hash, src_xyz, receivers, abs_coeffs):
        """Build PFFDTD-format JSON model (matches SketchUp export format)."""
        sources = [{"xyz": src_xyz.tolist()}]
        recs = [{"xyz": r.tolist()} for r in receivers]

        return {
            "mats_hash": mats_hash,
            "sources": sources,
            "receivers": recs,
        }

    def _fit_materials(self, abs_coeffs, mat_dir):
        """Fit CHORAS absorption coefficients to PFFDTD DEF triplets."""
        from materials.adm_funcs import fit_to_Sabs_oct_11

        # CHORAS gives 5 bands: 125, 250, 500, 1000, 2000 Hz
        # PFFDTD wants 11 bands: 16, 31.5, 63, 125, 250, 500, 1k, 2k, 4k, 8k, 16k Hz
        choras_freqs = [125, 250, 500, 1000, 2000]

        mat_files = {}
        for i, (name, alpha_str) in enumerate(abs_coeffs.items()):
            alphas_5 = [float(x.strip()) for x in alpha_str.split(",")]

            # Expand 5-band to 11-band by interpolation/extrapolation
            alphas_11 = self._expand_to_11_bands(alphas_5)

            # Clamp
            alphas_11 = np.clip(alphas_11, 0.01, 0.99)

            # Fit DEF triplets
            mat_file = str(mat_dir / f"mat_{i:02d}.h5")
            fit_to_Sabs_oct_11(alphas_11, mat_file)
            mat_files[name] = mat_file
            print(f"  Material {i} ({name}): alpha_500={alphas_5[2]:.2f}")

        return mat_files

    @staticmethod
    def _expand_to_11_bands(alphas_5):
        """Expand 5 octave-band values to 11 (16 Hz to 16 kHz)."""
        # CHORAS: [125, 250, 500, 1000, 2000]
        # PFFDTD: [16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        a = np.array(alphas_5)
        out = np.zeros(11)
        # Below 125 Hz: extrapolate flat from 125
        out[0:3] = a[0]
        # 125-2000 Hz: direct mapping
        out[3] = a[0]   # 125
        out[4] = a[1]   # 250
        out[5] = a[2]   # 500
        out[6] = a[3]   # 1000
        out[7] = a[4]   # 2000
        # Above 2000 Hz: extrapolate flat from 2000
        out[8:] = a[4]
        return out

    def _run_setup(self, json_model_path, mat_files, save_dir,
                   h, c0, Tc, rh, ir_length, fmax):
        """Run PFFDTD setup via Hamilton's sim_setup() function."""
        import platform
        from sim_setup import sim_setup

        # Build mat_files_dict: {surface_name: filename.h5}
        mat_files_dict = {}
        for name, path in mat_files.items():
            mat_files_dict[name] = os.path.basename(path)

        mat_folder = str(save_dir / "materials")
        PPW = round(c0 / (fmax * h))

        Nprocs = 1 if platform.system() == 'Windows' else None

        sim_setup(
            model_json_file=str(json_model_path),
            mat_folder=mat_folder,
            mat_files_dict=mat_files_dict,
            source_num=1,
            insig_type='impulse',
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

        print(f"PFFDTD setup complete: {save_dir}")

    def _run_fdtd(self, save_dir):
        """Run PFFDTD simulation engine."""
        from fdtd.sim_fdtd import SimEngine

        engine = SimEngine(str(save_dir), energy_on=False)
        engine.load_h5_data()
        engine.setup_mask()
        engine.allocate_mem()
        engine.set_coeffs()
        engine.checks()

        # Try GPU first, fall back to CPU
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "ppffdtd"))
            from gpu_engine import run_gpu
            print("PFFDTD: using GPU engine")
            run_gpu(engine)
        except (ImportError, RuntimeError):
            print("PFFDTD: using CPU engine (numba)")
            engine.run_all()

        engine.save_outputs()

        # Post-process
        irs = self._postprocess(save_dir, engine)
        return irs

    def _postprocess(self, save_dir, engine):
        """Post-process FDTD output: filter + resample."""
        from fdtd.process_outputs import ProcessOutputs

        po = ProcessOutputs(str(save_dir))
        po.initial_process(fcut=10.0, N_order=4)

        # Low-pass at grid Nyquist (remove numerical dispersion)
        fmax = engine.c / (2 * engine.h)
        po.apply_lowpass(fcut=0.9 * fmax, N_order=8)

        # Resample to 48 kHz
        po.resample(Fs_f=48000)

        # Return IRs as list of lists (one per receiver)
        r_out = po.r_out_f
        if r_out.ndim == 1:
            irs = [r_out.tolist()]
        else:
            irs = [r_out[r].tolist() for r in range(r_out.shape[0])]
        print(f"PFFDTD post: {len(irs)} IRs, {len(irs[0])} samples each")
        return irs

    def _write_results(self, config, irs, json_file_path):
        """Write impulse responses back to CHORAS JSON."""
        result_block = config["results"][0]
        result_block["resultType"] = "PFFDTD"

        for i, resp in enumerate(result_block["responses"]):
            if i < len(irs):
                resp["receiverResults"] = irs[i]
                resp["receiverResultsUncorrected"] = irs[i]
            else:
                resp["receiverResults"] = []
                resp["receiverResultsUncorrected"] = []

        # Update percentage
        result_block["percentage"] = 100

        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        print(f"PFFDTD: results written to {json_file_path}")


if __name__ == "__main__":
    json_file_path = os.environ.get("JSON_PATH")

    if json_file_path is None:
        # Standalone test mode
        common_dir = Path(__file__).parent.parent / "common"
        json_file_path = str(common_dir / "exampleInput_DG.json")
        if not os.path.exists(json_file_path):
            print("No JSON_PATH set and no example found. Usage:")
            print("  JSON_PATH=/path/to/sim.json python PFDTDInterface.py")
            sys.exit(1)

    print(f"Running PFFDTD method with JSON_PATH={json_file_path}")

    method = PFDTDMethod()
    method.run_simulation(json_file_path)

    # Save results (CHORAS pattern)
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "common"))
        from HelperFunctions import save_results
        save_results(json_file_path)
    except ImportError:
        print("(HelperFunctions not available — standalone mode)")
