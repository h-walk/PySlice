# JAX_TACAW

JAX_TACAW is a Python package for simulating and analyzing **time-resolved electron diffraction patterns** from molecular dynamics trajectories. It implements the **TACAW method** to convert time-domain electron scattering data into frequency-domain spectra, enabling the analysis of phonon dynamics and vibrational modes in materials at the atomic scale.

## Core Value Proposition

The system bridges molecular dynamics simulations with electron microscopy experiments by:
- Converting LAMMPS MD trajectories into electron diffraction patterns
- Performing multislice simulations using optimized NumPy implementations
- Applying FFT-based TACAW analysis to extract frequency-domain information
- Providing comprehensive visualization and analysis tools

## System Architecture

### Data Flow Pipeline

```
LAMMPS Trajectory → TrajectoryLoader → Trajectory Object
                                            ↓
                                   MultisliceCalculator (Optimized NumPy)
                                            ↓
                                     WaveFunction Data
                                            ↓
                                    FFT (TACAW Method)
                                            ↓
                                      TACAWData
                                            ↓
                              Analysis & Visualization
```

## Main Data Structures

### Trajectory
```python
@dataclass
class Trajectory:
    atom_types: np.ndarray      # Atomic numbers (n_atoms,)
    positions: np.ndarray       # Atomic positions (n_frames, n_atoms, 3)
    velocities: np.ndarray      # Atomic velocities (n_frames, n_atoms, 3)
    box_matrix: np.ndarray      # Simulation box matrix (3, 3)
    timestep: float            # Timestep in picoseconds
```

The `Trajectory` class stores molecular dynamics data with:
- OVITO-based loading from LAMMPS dump files
- Automatic `.npy` caching for faster subsequent loads
- Atomic number mapping from LAMMPS types to elements

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
Contains complex wavefunctions from multislice simulations with:
- Multi-probe and multi-layer support
- FFT conversion to TACAW data via `fft_to_tacaw_data()`
- Efficient complex array storage

### TACAWData (Frequency-Domain Analysis)
```python
@dataclass
class TACAWData:
    probe_positions: List[Tuple[float, float]]  # Probe positions in Ångstroms
    frequency: np.ndarray                      # Frequencies in THz
    kx: np.ndarray                            # kx sampling vectors (Å⁻¹)
    ky: np.ndarray                            # ky sampling vectors (Å⁻¹)
    intensity: np.ndarray                     # Intensity data (probe_positions, frequency, kx, ky)
```

Provides comprehensive analysis methods:
- **`spectrum(probe_index=0)`**: Extract frequency spectrum for a specific probe
- **`diffraction(probe_index=0)`**: Extract diffraction pattern (summed over frequencies)
- **`spectral_diffraction(frequency, probe_index=0)`**: Diffraction pattern at specific frequency
- **`spectrum_image(frequency, probe_indices=None)`**: Real-space intensity map
- **`masked_spectrum(mask, probe_index=0)`**: Apply k-space masks for selective analysis

## Multislice Calculators

### MultisliceCalculatorNumpy (Default)

**High-performance class-based implementation** using NumPy with Kirkland potentials:

```python
class MultisliceCalculatorNumpy:
    def run_simulation(
        self,
        trajectory: Trajectory,
        aperture: float,              # Probe aperture in milliradians
        voltage_kv: float,            # Accelerating voltage in kV
        pixel_size: float,            # Pixel size in Ångstroms
        probe_positions: Optional[List[Tuple[float, float]]] = None,
        slice_thickness: float = 0.5,
        sampling: float = 0.1,
        batch_size: int = 10,
    ) -> WFData:
```


### MultisliceCalculatorAbtem (Alternative)

Uses the abtem library for simulations:

```python
class MultisliceCalculatorAbtem:
    def run_simulation(
        self,
        trajectory: Trajectory,
        aperture: float,              # Probe aperture in milliradians
        voltage_kv: float,            # Accelerating voltage in kV
        pixel_size: float,            # Pixel size in Ångstroms
        probe_positions: Optional[List[Tuple[float, float]]] = None,
        slice_thickness: float = 1.0,
        sampling: float = 0.05,
        batch_size: int = 10,
    ) -> WFData:
```

## Installation

### Prerequisites
- Python 3.12+
- Virtual environment recommended

### Install Dependencies

**Using pip:**
```bash
# Install core requirements
pip install -r requirements.txt

# Special installation for OVITO
pip install ovito --find-links https://www.ovito.org/pip/
```

**Using uv (recommended for faster installs):**
```bash
# Install dependencies directly
uv sync
```

### Key Dependencies
- **`numpy>=1.20.0`**: Scientific computing (core)
- **`abtem>=1.0.6`**: Alternative multislice implementation
- **`ase>=3.26.0`**: Atomic simulation environment
- **`ovito>=3.8.0`**: LAMMPS trajectory loading
- **`matplotlib>=3.5.0`**: Visualization

## Usage

### Quick Start
```bash
python main.py
```

### Basic Workflow

```python
from src.io.loader import TrajectoryLoader
from src.tacaw.ms_calculator_npy import MultisliceCalculatorNumpy

# 1. Load trajectory with atomic number mapping
atomic_numbers = {1: 5, 2: 7}  # LAMMPS type 1→Boron, 2→Nitrogen
loader = TrajectoryLoader("trajectory.lammpstrj", timestep=0.005, 
                         atomic_numbers=atomic_numbers)
trajectory = loader.load()

# 2. Set simulation parameters
sim_params = {
    'aperture': 0.0,         # 0.0 mrad = plane wave
    'voltage_kv': 100.0,     # 100 kV accelerating voltage
    'pixel_size': 0.1,       # 0.1 Å/pixel
    'slice_thickness': 0.5,  # 0.5 Å per slice
    'sampling': 0.1,         # Real space sampling
    'batch_size': 10,        # Frames per batch
}

# 3. Run multislice simulation (using optimized NumPy by default)
calculator = MultisliceCalculatorNumpy()
wf_data = calculator.run_simulation(
    trajectory=trajectory,
    probe_positions=[(0.0, 0.0)],  # Single center probe
    **sim_params
)

# 4. Convert to frequency domain
tacaw_data = wf_data.fft_to_tacaw_data()

# 5. Analyze results
spectrum = tacaw_data.spectrum(probe_index=0)
diffraction = tacaw_data.diffraction(probe_index=0)
```

### Alternative: Using AbTem Calculator

```python
from src.tacaw.ms_calculator_abtem import MultisliceCalculatorAbtem

# Switch to AbTem for GPU acceleration (if available)
calculator = MultisliceCalculatorAbtem()
wf_data = calculator.run_simulation(trajectory, **sim_params)
```

### Advanced: Multiple Probe Positions

```python
# Grid-based probe setup
from main import setup_probe_positions

# Automatic grid generation
probe_positions = setup_probe_positions(trajectory, grid_dim="2x2")
# Creates 4 probes in corners with automatic padding

# Or manual positioning
probe_positions = [(0.0, 0.0), (10.0, 15.0), (25.0, 30.0)]

# Run with multiple probes
wf_data = calculator.run_simulation(
    trajectory=trajectory,
    probe_positions=probe_positions,
    aperture=30.0,  # Convergent beam
    **sim_params
)

tacaw_data = wf_data.fft_to_tacaw_data()

# Analyze each probe position
for i, pos in enumerate(tacaw_data.probe_positions):
    spectrum = tacaw_data.spectrum(probe_index=i)
    print(f"Probe {i} at {pos}: max intensity = {spectrum.max()}")
```


### Caching Strategy
1. **Trajectory Cache**: `.npy` files avoid OVITO parsing 
2. **Wave Function Cache**: `psi_data/` stores simulation results (AbTem)
3. **TACAWData Cache**: Final results saved as pickle files


## Visualization

The system generates comprehensive scientific plots:

1. **Frequency Spectra**: Intensity vs frequency showing vibrational modes
2. **Diffraction Patterns**: k-space intensity distribution
3. **Spectral Diffraction**: Frequency-resolved diffraction patterns
4. **Dispersion Plots**: Phonon band structure (ω vs k)

All plots saved as high-resolution PNG files (300 DPI) with professional styling.

## Configuration

### Key Parameters

**Trajectory Loading**:
```python
atomic_numbers = {1: 5, 2: 7}  # Map LAMMPS types to elements
timestep = 0.005  # 5 fs in picoseconds
```

**Probe Setup**:
```python
probe_positions = setup_probe_positions(
    trajectory, 
    grid_dim="1x1"  # Options: "1x1", "2x2", "3x3", etc.
)
```

**Simulation Parameters**:
```python
sim_params = {
    'aperture': 0.0,         # mrad (0 = plane wave)
    'voltage_kv': 100.0,     # Accelerating voltage
    'pixel_size': 0.1,       # Å/pixel
    'slice_thickness': 0.5,  # Å per slice
    'sampling': 0.1,         # Real space sampling
    'batch_size': 10         # Frames per batch
}
```

## API Reference

### Core Classes
- `TrajectoryLoader(filename, timestep, atomic_numbers=None)`
- `MultisliceCalculatorNumpy().run_simulation(...) → WFData` (Default, optimized)
- `MultisliceCalculatorAbtem().run_simulation(...) → WFData` (Alternative, GPU)
- `WFData.fft_to_tacaw_data(layer_index=None) → TACAWData`

### Analysis Methods
- `TACAWData.spectrum(probe_index=0) → np.ndarray`
- `TACAWData.diffraction(probe_index=0) → np.ndarray`
- `TACAWData.spectral_diffraction(frequency, probe_index=0) → np.ndarray`
- `TACAWData.spectrum_image(frequency, probe_indices=None) → np.ndarray`

## Technical Stack

- **Multislice**: Optimized NumPy implementation (primary), abtem (alternative)
- **MD Loading**: OVITO for LAMMPS trajectories
- **Numerics**: NumPy for array operations, Kirkland atomic potentials
- **Visualization**: Matplotlib with scientific styling
- **Performance**: Vectorized operations, efficient memory management

