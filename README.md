# TACAW

TACAW is a Python package for simulating and analyzing time-resolved electron diffraction patterns from molecular dynamics trajectories using the abtem library.

## Overview

This package implements the JACR method for converting time-domain electron scattering data into frequency-domain spectra. It provides a complete workflow from LAMMPS trajectory loading to TACAW data analysis and visualization.

## Main Data Structures

### Trajectory
```python
@dataclass
class Trajectory:
    atom_types: np.ndarray      # Atomic numbers or LAMMPS types (n_atoms,)
    positions: np.ndarray       # Atomic positions (n_frames, n_atoms, 3)
    velocities: np.ndarray      # Atomic velocities (n_frames, n_atoms, 3)
    box_matrix: np.ndarray      # Simulation box matrix (3, 3)
    timestep: float            # Timestep in picoseconds
```

The `Trajectory` class stores molecular dynamics data loaded from LAMMPS dump files. It supports:
- Loading from LAMMPS dump files via OVITO
- Automatic caching of trajectory data
- Optional atomic number mapping from LAMMPS types to actual atomic numbers
- Coordinate transformations and analysis

### WFData (Wave Function Data)
```python
@dataclass
class WFData:
    probe_positions: List[Tuple[float, float]]  # Probe positions in Ångstroms
    time: np.ndarray                           # Time array in picoseconds
    kx: np.ndarray                            # kx sampling vectors
    ky: np.ndarray                            # ky sampling vectors
    layer: np.ndarray                         # Layer indices
    wavefunction_data: np.ndarray             # Complex wavefunctions (probe_positions, time, kx, ky, layer)
```

The `WFData` class contains the raw wave function data from multislice simulations. It provides:
- FFT conversion to TACAW data via `fft_to_tacaw_data()`
- Multi-layer support for complex sample structures
- Efficient storage of complex wave function arrays

### TACAWData (TACAW Analysis Data)
```python
@dataclass
class TACAWData:
    probe_positions: List[Tuple[float, float]]  # Probe positions in Ångstroms
    frequency: np.ndarray                      # Frequencies in THz
    kx: np.ndarray                            # kx sampling vectors (Å⁻¹)
    ky: np.ndarray                            # ky sampling vectors (Å⁻¹)
    intensity: np.ndarray                     # Intensity data (probe_positions, frequency, kx, ky)
```

The `TACAWData` class contains the frequency-domain intensity data and provides comprehensive analysis methods:

#### Key Methods:
- **`spectrum(probe_index=0)`**: Extract frequency spectrum for a specific probe
- **`diffraction(probe_index=0)`**: Extract diffraction pattern (summed over frequencies)
- **`spectral_diffraction(frequency, probe_index=0)`**: Extract diffraction pattern at specific frequency
- **`spectrum_image(frequency, probe_indices=None)`**: Real-space intensity map across probe positions
- **`masked_spectrum(mask, probe_index=0)`**: Apply k-space masks for region-specific analysis

## Abtem-Based Multislice Calculator

### MultisliceCalculatorAbtem

The `MultisliceCalculatorAbtem` class performs multislice electron microscopy simulations using the abtem library with ASE (Atomic Simulation Environment) integration.

```python
class MultisliceCalculatorAbtem:
    def __init__(self):
        # Element mapping: atomic_number -> element_symbol
        self.element_map = {
            5: 'B', 6: 'C', 7: 'N', 8: 'O',  # Boron, Carbon, Nitrogen, Oxygen
            # ... additional elements
        }

    def run_simulation(
        self,
        trajectory: Trajectory,
        aperture: float,              # Probe aperture in milliradians
        voltage_kv: float,            # Accelerating voltage in kV
        pixel_size: float,            # Pixel size in Ångstroms
        defocus: float = 0.0,         # Probe defocus in Ångstroms
        probe_positions: Optional[List[Tuple[float, float]]] = None,
        element_symbols: Optional[Dict[int, str]] = None,
        slice_thickness: float = 1.0,
        sampling: float = 0.05,
        batch_size: int = 10,
        cleanup_temp_files: bool = False,
    ) -> WFData:
        # Returns wave function data for all trajectory frames
```

### Key Features:

1. **ASE Integration**: Converts trajectory data to ASE Atoms objects for abtem compatibility
2. **Batch Processing**: Processes trajectory frames in configurable batches for memory efficiency
3. **Flexible Probe Positions**: Supports single probe, grid patterns, or custom positions
4. **Element Mapping**: Automatic conversion from atomic numbers to element symbols
5. **Caching**: Intelligent caching of wave function data to avoid recomputation
6. **Progress Tracking**: Real-time progress bars for long simulations

## Installation

### Prerequisites
- Python 3.12 or higher
- pip package manager

### Install Dependencies
```bash
# Install core requirements
pip install -r requirements.txt

# Special installation for OVITO (if needed)
pip install ovito --find-links https://www.ovito.org/pip/
```

### Key Dependencies
- **`abtem>=1.0.6`**: Multislice electron microscopy simulations
- **`ase>=3.26.0`**: Atomic simulation environment for crystal structures
- **`ovito>=3.8.0`**: LAMMPS trajectory loading and analysis
- **`numpy>=1.20.0`**: Scientific computing
- **`matplotlib>=3.5.0`**: Plotting and visualization

## Usage

### Basic Workflow

```python
from src.io.loader import TrajectoryLoader
from src.tacaw.ms_calculator_abtem import MultisliceCalculatorAbtem

# 1. Load trajectory with atomic number mapping
atomic_numbers = {1: 5, 2: 7}  # Map LAMMPS types 1→Boron, 2→Nitrogen
loader = TrajectoryLoader("trajectory.lammpstrj", timestep=0.005, atomic_numbers=atomic_numbers)
trajectory = loader.load()

# 2. Set up simulation parameters
sim_params = {
    'aperture': 0.0,         # 0.0 mrad = plane wave, >0 = convergent beam
    'voltage_kv': 100.0,     # 100 kV accelerating voltage
    'pixel_size': 0.1,       # 0.1 Å/pixel
    'slice_thickness': 0.5,  # 0.5 Å slice thickness
    'sampling': 0.1,         # Real space sampling in Å
    'batch_size': 10,        # Process 10 frames at a time
}

# 3. Run multislice simulation
calculator = MultisliceCalculatorAbtem()
wf_data = calculator.run_simulation(
    trajectory=trajectory,
    probe_positions=[(0.0, 0.0)],  # Single probe at center
    **sim_params
)

# 4. Convert to TACAW data
tacaw_data = wf_data.fft_to_tacaw_data()

# 5. Analyze results
spectrum = tacaw_data.spectrum(probe_index=0)
diffraction = tacaw_data.diffraction(probe_index=0)
```

### Advanced Usage

```python
# Multiple probe positions in a grid
from src.tacaw.ms_calculator_abtem import MultisliceCalculatorAbtem

# Grid probe positions
probe_positions = [
    (0.0, 0.0),   # Center
    (5.0, 0.0),   # Right
    (0.0, 5.0),   # Top
    (-5.0, 0.0),  # Left
    (0.0, -5.0),  # Bottom
]

# Run with multiple probes
wf_data = calculator.run_simulation(
    trajectory=trajectory,
    probe_positions=probe_positions,
    aperture=1.0,  # 1 mrad aperture for convergent beam
    **sim_params
)

# Convert and analyze
tacaw_data = wf_data.fft_to_tacaw_data()

# Extract real-space intensity map at specific frequency
frequency_thz = 50.0  # 50 THz
intensity_map = tacaw_data.spectrum_image(frequency_thz)
```

## Data Flow

```
LAMMPS dump file → TrajectoryLoader → Trajectory
                                       ↓
Trajectory → MultisliceCalculatorAbtem → WFData
                                       ↓
WFData.fft_to_tacaw_data() → TACAWData → Analysis & Visualization
```

## Features

### Trajectory Loading
- **OVITO Integration**: Robust LAMMPS dump file parsing
- **Caching**: Automatic .npy file caching for fast reloading
- **Atomic Numbers**: Optional mapping from LAMMPS types to atomic numbers
- **Validation**: Comprehensive data validation and error checking

### Multislice Simulation
- **Abtem Backend**: High-performance multislice calculations
- **ASE Compatibility**: Full integration with Atomic Simulation Environment
- **Batch Processing**: Memory-efficient processing of large trajectories
- **Progress Monitoring**: Real-time progress tracking with tqdm

### TACAW Analysis
- **FFT Processing**: Time → frequency domain conversion
- **Multiple Metrics**: Spectra, diffraction patterns, real-space maps
- **Spatial Analysis**: k-space masking and region-specific analysis
- **Visualization**: Comprehensive plotting functions

### Performance Optimizations
- **Intelligent Caching**: Avoids recomputation of expensive operations
- **Memory Management**: Configurable batch sizes for large datasets
- **Parallel Processing**: Efficient array operations with NumPy

## API Reference

### TrajectoryLoader
- `TrajectoryLoader(filename, timestep, atomic_numbers=None)`
- `load() → Trajectory`

### MultisliceCalculatorAbtem
- `run_simulation(trajectory, aperture, voltage_kv, pixel_size, ...) → WFData`

### WFData
- `fft_to_tacaw_data(layer_index=None) → TACAWData`

### TACAWData
- `spectrum(probe_index=0) → np.ndarray`
- `diffraction(probe_index=0) → np.ndarray`
- `spectral_diffraction(frequency, probe_index=0) → np.ndarray`
- `spectrum_image(frequency, probe_indices=None) → np.ndarray`
- `masked_spectrum(mask, probe_index=0) → np.ndarray`

## License

This project is developed for scientific research in electron microscopy and materials science.

## Citation
g
