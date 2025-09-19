# PySlice

PySlice is a Python package for simulating and analyzing multslice simulations from molecular dynamics trajectories. It implements the **TACAW method** to convert time-domain electron scattering data into frequency-domain spectra, enabling the analysis of phonon dynamics and vibrational modes in materials at the atomic scale.

## Core Value Proposition

The system bridges molecular dynamics simulations with electron microscopy experiments by:
- Converting LAMMPS MD trajectories into electron diffraction patterns
- Performing multislice simulations using optimized PyTorch/NumPy implementations
- Applying FFT-based TACAW analysis to extract frequency-domain information
- Providing comprehensive visualization and analysis tools for phonon spectroscopy

## System Architecture

### Data Flow Pipeline

```
LAMMPS Trajectory → TrajectoryLoader → Trajectory Object
                                            ↓
                             MultisliceCalculator (PyTorch/NumPy)
                                            ↓
                                     WaveFunction Data (WFData)
                                            ↓
                                      TACAWData (FFT)
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
- Element name mapping from LAMMPS types

### WFData (Wave Function Data)
```python
@dataclass
class WFData:
    probe_positions: List[Tuple[float, float]]  # Probe positions in Ångstroms
    time: np.ndarray                           # Time array in picoseconds
    kxs: np.ndarray                           # kx sampling vectors (Å⁻¹)
    kys: np.ndarray                           # ky sampling vectors (Å⁻¹)
    layer: np.ndarray                         # Layer indices
    wavefunction_data: np.ndarray             # Complex wavefunctions (probe_positions, time, kx, ky, layer)
```

Contains complex wavefunctions from multislice simulations with:
- Multi-probe and multi-layer support
- Conversion to TACAW data via `TACAWData(wfdata)` constructor
- Efficient complex array storage using PyTorch/NumPy

### TACAWData (Frequency-Domain Analysis)
```python
@dataclass
class TACAWData(WFData):
    probe_positions: List[Tuple[float, float]]  # Probe positions in Ångstroms
    frequencies: np.ndarray                     # Frequencies in THz
    kxs: np.ndarray                           # kx sampling vectors (Å⁻¹)
    kys: np.ndarray                           # ky sampling vectors (Å⁻¹)
    intensity: np.ndarray                     # Intensity data (probe_positions, frequency, kx, ky)
```

Provides comprehensive analysis methods with automatic probe averaging:
- **`spectrum(probe_index=None)`**: Extract frequency spectrum (None = average all probes)
- **`diffraction(probe_index=None)`**: Extract diffraction pattern (summed over frequencies)
- **`spectral_diffraction(frequency, probe_index=None)`**: Diffraction pattern at specific frequency
- **`spectrum_image(frequency, probe_indices=None)`**: Real-space intensity map
- **`masked_spectrum(mask, probe_index=None)`**: Apply k-space masks for selective analysis
- **`dispersion(kx_path, ky_path, probe_index=None)`**: Extract dispersion relation along k-path

## Multislice Calculators

### MultisliceCalculator (PyTorch/NumPy Implementation)

**Unified high-performance implementation** with automatic backend selection:

```python
class MultisliceCalculator:
    def setup(
        self,
        trajectory: Trajectory,
        aperture: float = 0.0,        # Probe aperture in milliradians  
        voltage_eV: float = 100e3,    # Accelerating voltage in eV
        slice_thickness: float = 0.5,  # Å per slice
        sampling: float = 0.1,         # Real space sampling in Å/pixel
        probe_positions: Optional[List[Tuple[float, float]]] = None,
        batch_size: int = 10,
    )
    
    def run(self) -> WFData:
        # Returns WFData object with wavefunction data
```

Features:
- **Automatic device selection**: CUDA → MPS → CPU
- **Kirkland atomic potentials**: Accurate quantum mechanical form factors
- **Vectorized probe processing**: Efficient multi-probe STEM support
- **Caching system**: Automatic frame-level caching in `psi_data/`
- **Memory efficient**: Smart batching for large datasets

### Core Physics Classes

The calculator uses modular physics components:

```python
# Potential generation with Kirkland form factors
Potential(xs, ys, zs, positions, atom_types, kind="kirkland")

# Probe wavefunction 
Probe(xs, ys, mrad, eV)

# Multislice propagation algorithm
Propagate(probe, potential)

# Batched probe creation for STEM
create_batched_probes(base_probe, positions)
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
- **`torch`**: GPU acceleration (optional but recommended)
- **`ovito>=3.8.0`**: LAMMPS trajectory loading
- **`matplotlib>=3.5.0`**: Visualization
- **`ipywidgets`**: Interactive visualization in Jupyter

## Usage

### Quick Start
```bash
python main.py
```

### Basic Workflow

```python
from src.io.loader import TrajectoryLoader
from src.multislice.calculators import MultisliceCalculator
from src.postprocessing.tacaw_data import TACAWData

# 1. Load trajectory with element name mapping
atom_mapping = {1: "B", 2: "N"}  # LAMMPS type 1→Boron, 2→Nitrogen
loader = TrajectoryLoader("trajectory.lammpstrj", timestep=0.005,
                         atom_mapping=atom_mapping)
trajectory = loader.load()

# 2. Setup and run multislice simulation
calculator = MultisliceCalculator()
calculator.setup(
    trajectory=trajectory,
    aperture=0.0,           # 0.0 mrad = plane wave
    voltage_eV=100e3,       # 100 keV
    sampling=0.1,           # 0.1 Å/pixel
    slice_thickness=0.5,    # 0.5 Å per slice
    probe_positions=None    # Default: center probe
)
wf_data = calculator.run()

# 3. Convert to frequency domain using TACAW
tacaw_data = TACAWData(wf_data)

# 4. Analyze results (None = average over all probes)
spectrum = tacaw_data.spectrum(probe_index=None)
diffraction = tacaw_data.diffraction(probe_index=None)
```

### STEM Imaging with Multiple Probes

```python
from src.multislice.multislice import probe_grid

# Create probe grid for STEM
probe_positions = probe_grid(
    x_range=[10, 20],  # Å
    y_range=[10, 20],  # Å  
    nx=30,  # 30x30 grid
    ny=30
)

# Setup calculator with convergent beam
calculator = MultisliceCalculator()
calculator.setup(
    trajectory=trajectory,
    aperture=30.0,  # 30 mrad convergent beam
    voltage_eV=100e3,
    probe_positions=probe_positions
)

wf_data = calculator.run()
tacaw_data = TACAWData(wf_data)

# Generate spectrum image at specific frequency
spec_img = tacaw_data.spectrum_image(frequency=35.0)  # THz
spec_img_2d = spec_img.reshape(30, 30)  # Reshape to grid
```

### Phonon Dispersion Analysis

```python
# Extract dispersion along specific k-path
kx = np.linspace(0, 10, 1000)  # Å⁻¹
ky = np.zeros_like(kx)  # Along kx axis

dispersion = tacaw_data.dispersion(kx, ky, probe_index=None)

# Plot dispersion
plt.imshow(dispersion**0.1, aspect='auto', 
           extent=[kx.min(), kx.max(), 
                   tacaw_data.frequencies.min(), 
                   tacaw_data.frequencies.max()])
plt.xlabel('kx (Å⁻¹)')
plt.ylabel('Frequency (THz)')
```

### Interactive Visualization (Jupyter)

```python
from ipywidgets import interact, IntSlider

def tacaw_viewer(tacaw):
    n_frequencies = len(tacaw.frequencies)
    
    def plot_frequency(frequency_index):
        intensity = tacaw.intensity[0, frequency_index, :, :]**0.25
        plt.imshow(intensity.T, cmap="inferno")
        plt.title(f"Frequency: {tacaw.frequencies[frequency_index]:.2f} THz")
        plt.colorbar(label="Intensity^0.25")
        plt.show()
    
    interact(plot_frequency,
             frequency_index=IntSlider(value=0, min=0, max=n_frequencies-1))

tacaw_viewer(tacaw_data)
```

## Performance & Optimization

### Device Selection
The calculator automatically selects the best available device:
1. **CUDA GPU**: Highest performance for NVIDIA GPUs
2. **MPS**: Apple Silicon acceleration
3. **CPU**: Fallback with optimized NumPy

### Caching Strategy
1. **Trajectory Cache**: `.npy` files avoid repeated OVITO parsing
2. **Frame Cache**: `psi_data/` stores computed wavefunctions
3. **Automatic cache key generation**: Based on simulation parameters

### Performance Tips
- Use PyTorch backend for GPU acceleration when available
- Process trajectories in batches for memory efficiency
- Enable frame caching for iterative workflows
- Use probe grids for efficient STEM simulations

## API Reference

### Core Classes
- `TrajectoryLoader(filename, timestep, atom_mapping=None)`
- `MultisliceCalculator()` - Unified calculator with auto backend selection
- `TACAWData(wfdata)` - Constructor takes WFData, performs FFT automatically

### Analysis Methods (all support probe_index=None for averaging)
- `TACAWData.spectrum(probe_index=None) → np.ndarray`
- `TACAWData.diffraction(probe_index=None) → np.ndarray`
- `TACAWData.spectral_diffraction(frequency, probe_index=None) → np.ndarray`
- `TACAWData.spectrum_image(frequency, probe_indices=None) → np.ndarray`
- `TACAWData.dispersion(kx_path, ky_path, probe_index=None) → np.ndarray`
- `TACAWData.masked_spectrum(mask, probe_index=None) → np.ndarray`

### Helper Functions
- `probe_grid(x_range, y_range, nx, ny)` - Generate STEM probe positions
- `gridFromTrajectory(trajectory, sampling, slice_thickness)` - Setup spatial grids

## Technical Stack

- **Multislice**: PyTorch (GPU) / NumPy (CPU) with Kirkland potentials
- **MD Loading**: OVITO for LAMMPS trajectories  
- **FFT**: PyTorch/NumPy FFT for TACAW conversion
- **Visualization**: Matplotlib with scientific styling
- **Interactive**: ipywidgets for Jupyter notebooks
- **Performance**: Vectorized operations, GPU acceleration, smart caching
