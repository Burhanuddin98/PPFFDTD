"""
Load PFFDTD H5 simulation data into PPFFDTD engine.

Bridges the gap: reads Hamilton's voxelizer output (H5 files)
and configures a PPFFDTD FDTDEngine with the same grid, materials,
sources, and receivers.
"""

import numpy as np
from pathlib import Path
import h5py
from ppffdtd.engine import FDTDEngine, MMb


def load_from_pffdtd(data_dir):
    """Load PFFDTD H5 data and return configured FDTDEngine.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing vox_out.h5, comms_out.h5,
        sim_consts.h5, sim_mats.h5

    Returns
    -------
    engine : FDTDEngine
    Nt : int — number of time steps
    """
    d = Path(data_dir)

    # Grid constants
    with h5py.File(d / 'sim_consts.h5', 'r') as f:
        c = float(f['c'][()])
        h = float(f['h'][()])
        Ts = float(f['Ts'][()])
        l = float(f['l'][()])
        l2 = float(f['l2'][()])

    # Voxel data
    with h5py.File(d / 'vox_out.h5', 'r') as f:
        Nx = int(f['Nx'][()])
        Ny = int(f['Ny'][()])
        Nz = int(f['Nz'][()])
        bn_ixyz = f['bn_ixyz'][...]
        adj_bn = f['adj_bn'][...]
        mat_bn = f['mat_bn'][...]
        saf_bn = f['saf_bn'][...]

    # Communications (sources, receivers, signals)
    with h5py.File(d / 'comms_out.h5', 'r') as f:
        in_ixyz = f['in_ixyz'][...]
        out_ixyz = f['out_ixyz'][...]
        in_sigs = f['in_sigs'][...]
        out_reorder = f['out_reorder'][...]
        Nt = int(f['Nt'][()])

    # Materials
    with h5py.File(d / 'sim_mats.h5', 'r') as f:
        Nmat = int(f['Nmat'][()])
        Mb = f['Mb'][...]
        DEF = np.zeros((Nmat, MMb, 3))
        for i in range(Nmat):
            ds = f[f'mat_{i:02d}_DEF'][...]
            DEF[i, :Mb[i]] = ds

    # Create engine
    engine = FDTDEngine(Nx, Ny, Nz, h, c)

    # Override Ts/l/l2 to match PFFDTD exactly (may differ from our CFL default)
    engine.Ts = Ts
    engine.l = l
    engine.l2 = l2

    # Set boundary
    engine.set_boundary(bn_ixyz, adj_bn, mat_bn, saf_bn)

    # Set materials
    engine.set_materials(DEF, Mb)

    # Add sources
    for s in range(len(in_ixyz)):
        engine.add_source(int(in_ixyz[s]), in_sigs[s])

    # Add receivers (in reordered order)
    for r in range(len(out_ixyz)):
        engine.add_receiver(int(out_ixyz[r]))

    # Store reorder map for output
    engine.out_reorder = out_reorder

    print(f"PPFFDTD loaded: {Nx}x{Ny}x{Nz} = {Nx*Ny*Nz:,} voxels")
    print(f"  BN={bn_ixyz.size}, BNL={engine.Nbl}, ABC={engine.Nba}")
    print(f"  Materials: {Nmat}, Sources: {len(in_ixyz)}, Receivers: {len(out_ixyz)}")
    print(f"  Ts={Ts:.6e}, l={l:.6f}, l2={l2:.6f}, Nt={Nt}")
    print(f"  Room: {engine.room_dims[0]:.2f} x {engine.room_dims[1]:.2f} x "
          f"{engine.room_dims[2]:.2f} m")

    return engine, Nt
