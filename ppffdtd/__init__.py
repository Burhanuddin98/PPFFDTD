"""
PPFFDTD — Python Pretty Fast FDTD
==================================
Clean-room Python implementation of the PFFDTD room acoustics algorithm.
Based on Brian Hamilton's PFFDTD (MIT license).

Usage:
    from ppffdtd import FDTDEngine
    engine = FDTDEngine(grid, materials, sources, receivers)
    engine.run()
    ir = engine.get_ir(0)
"""

__version__ = '0.1.0'
