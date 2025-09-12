"""
Wave function data structure.
"""
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
from ..multislice.multislice import Probe

@dataclass
class WFData:
    """
    Data structure for wave function data with format: probe_positions, frame, kx, ky, layer.

    Attributes:
        probe_positions: List of (x,y) probe positions in Angstroms.
        frame: Time array (frame # * timestep) in picoseconds.
        kx: kx sampling vectors.
        ky: ky sampling vectors.
        layer: Layer indices for multi-layer calculations.
        wavefunction_data: Complex wavefunction array with shape (probe_positions, time, kx, ky, layer).
    """
    probe_positions: List[Tuple[float, float]]
    time: np.ndarray  # Time in picoseconds (frame # * timestep)
    kxs: np.ndarray    # kx sampling vectors
    kys: np.ndarray    # ky sampling vectors
    layer: np.ndarray # Layer indices
    wavefunction_data: np.ndarray  # Complex wavefunction array (probe_positions, time, kx, ky, layer)
    probe: Probe



