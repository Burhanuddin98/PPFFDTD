# PPFFDTD (Python Pretty Fast FDTD)

Clean-room Python implementation of the PFFDTD room acoustics algorithm (Brian Hamilton, MIT license).

Based on: https://github.com/bsxfun/pffdtd

## What is this?

A pure-Python FDTD wave solver for 3D room acoustics with:
- 7-point Cartesian stencil (6-point Laplacian + leapfrog time stepping)
- Frequency-dependent impedance boundaries via parallel RLC IIR filters
- Sided materials (one-sided lossy, other side rigid)
- Staircase surface-area correction
- Absorbing boundary conditions (ABC) at grid edges
- GPU acceleration via CuPy (optional)

## Why rewrite?

PFFDTD's Python engine is a reference implementation with numba JIT. This rewrite:
1. Makes every step explicit and readable (no JIT magic)
2. Runs on Windows without MSYS2/HDF5 dependencies
3. Accepts numpy arrays directly (no HDF5 required)
4. Clean API for integration with ROM frameworks
5. GPU via CuPy RawKernels (single kernel per step)

## Algorithm

See `docs/algorithm.md` for the complete step-by-step specification.

## License

MIT (same as PFFDTD)

## Citation

If this contributes to academic work, cite the original PFFDTD:

```bibtex
@misc{hamilton2021pffdtd,
  title = {PFFDTD Software},
  author = {Brian Hamilton},
  note = {https://github.com/bsxfun/pffdtd},
  year = {2021}
}
```
