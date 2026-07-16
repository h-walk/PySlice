# PySlice

A GPU-accelerated Python package for simulating vibrational electron energy loss spectroscopy (EELS) using the **TACAW method** (Time Autocorrelation of Auxiliary Wavefunctions). PySlice integrates molecular dynamics with multislice electron scattering calculations to predict momentum- and energy-resolved phonon spectra directly from atomic trajectories.

## Features

- **TACAW Analysis**: Convert time-domain electron scattering into frequency-domain phonon spectra
- **Integrated MD**: Run molecular dynamics with universal ML potentials (ORB, MACE, CHGNet)
- **GPU Acceleration**: PyTorch backend with automatic CUDA/MPS/CPU selection
- **Flexible Input**: Load structures from CIF, LAMMPS, XYZ, ASE trajectories, or ASE Atoms objects
- **STEM Imaging**: HAADF/ADF/BF imaging and 4D-STEM diffraction

## Installation

```bash
# Clone the repository
git clone https://github.com/h-walk/PySlice.git
cd PySlice

# Install with pip. -e = editable mode. [fast] will install torch (technically optional, but provides extreme speed improvements).
pip install -e ".[fast]"

# Install OVITO for trajectory loading
pip install ovito --find-links https://www.ovito.org/pip/

# Or using uv (recommended)
uv sync

# Add the shared SEA data model, provenance, and .sea serialization
uv sync --extra sea

# Optional ORB/MD support currently requires Python 3.12 because ORB's dm-tree
# dependency does not publish Python 3.13 wheels.
uv python install 3.12
uv sync --python 3.12 --extra fast --extra md
```

### SEA data and provenance integration

Install the optional `sea` extra when PySlice outputs should participate in the
shared pySEA data model:

```bash
pip install -e ".[sea]"
```

With sea-eco available, `WFData`, `TACAWData`, and `HAADFData` are calibrated
sea-eco `Signal` objects. Their construction and scientific transformations are
recorded in `Signal.Analysis`, parent/child SEAIDs connect derived results to
their inputs, and `.to_sea()` preserves the data and processing receipt.
TACAW analysis methods return tracked `Signal` results by default; pass
`as_signal=False` when a raw NumPy array is specifically required. Without the
extra, PySlice retains its standalone array behavior.

## Quick Start

### Full TACAW Pipeline (MD → Multislice → Phonon Dispersion)

```python
from ase.build import bulk
from pyslice import ORBMDCalculator,MultisliceCalculator,TACAWData

# 1. Create structure
atoms = bulk("Si", "diamond", a=5.431, cubic=True) * (10, 10, 2)

# 2. Run molecular dynamics
md = ORBMDCalculator(model_name="orb-v3-direct-inf-omat")
md.setup(atoms, temperature=300, timestep=2.0, production_steps=500, save_interval=5)
trajectory = md.run()

# 3. Run multislice (parallel beam for TACAW)
calc = MultisliceCalculator()
calc.setup(trajectory, aperture=0, voltage_eV=100e3, sampling=0.1, slice_thickness=0.5)
wf_data = calc.run()

# 4. Compute phonon spectrum
tacaw = TACAWData(wf_data)
Z = tacaw.spectral_diffraction(15.0)  # Diffraction at 15 THz
```

### Load Existing Trajectory

```python
from pyslice.io.loader import Loader

trajectory = Loader(
    "hBN.lammpstrj",
    timestep=0.005,  # ps
    atom_mapping={1: "B", 2: "N"}
).load()

# ASE trajectory or CIF/XYZ file
trajectory = Loader("silicon.cif").load()
```

### HAADF-STEM Imaging

```python
from pyslice import Loader,MultisliceCalculator,HAADFData
import numpy as np

# Load your trajectory
trajectory = Loader(
    "hBN.lammpstrj",
    timestep=0.005,  # ps
    atom_mapping={1: "B", 2: "N"}
).load()
# Optional cropping in time and space
trajectory = trajectory.get_random_timesteps(5).slice_positions([0,20],[0,20])

# Define probe scan grid
xs = np.linspace(5,12,16) ; ys = np.linspace(5,12,16)

calc = MultisliceCalculator()
calc.setup(trajectory, aperture=30, voltage_eV=100e3, sampling=0.1, probe_xs=xs, probe_ys=ys)
wf_data = calc.run()

haadf = HAADFData(wf_data)
haadf.calculateADF(inner_mrad=60, outer_mrad=200)
haadf.plot()
```

### TEM Diffraction

```python
from pyslice import Loader,MultisliceCalculator
import numpy as np

# Load your trajectory
trajectory = Loader(
    "hBN.lammpstrj",
    timestep=0.005,  # ps
    atom_mapping={1: "B", 2: "N"}
).load()
# Optional cropping in time and space
trajectory = trajectory.get_random_timesteps(5).slice_positions([0,20],[0,20])

calc = MultisliceCalculator()
calc.setup(trajectory, aperture=0, voltage_eV=100e3, sampling=0.1)
wf_data = calc.run()

wf_data.plot(powerscaling=0.125)  # Diffraction pattern

```

### Standalone RayTEM Wave Optics

`simulate_raytem_wave` propagates one coherent, sample-free electron wave
through an existing RayTEM configuration. RayTEM supplies the column geometry
and calibrated elements; the entrance wave remains explicit because a ray
bundle does not uniquely define wave phase.

```python
from pyslice import (
    GaussianWaveSource,
    ProbeAberrationModel,
    simulate_raytem_wave,
)

# Illustrative values: use an instrument-appropriate entrance wave.
source = GaussianWaveSource(
    voltage_eV=100e3,
    rms_size_A=(500, 500),
    curvature_inv_A=(0.0, 0.0),
    center_A=(0, 0),
    tilt_mrad=(0, 0),
)

result = simulate_raytem_wave(
    "macstem.json",
    start="gun",
    stop="CL3",
    source=source,
    extent_A=4096,
    sampling_A=8,
    record=False,
)

result.output.plot_realspace()
print(result.sampling_report())
```

Ronchigram measurements normally describe the effective probe-forming system,
not individual lenses. Supply those measured coefficients as one model at their
reference plane:

```python
# Illustrative values only. Cnm values are Angstroms; orientations are radians.
ronchi = ProbeAberrationModel(
    coefficients_A={
        "C12": (200.0, 0.3),
        "C21": (500.0, -0.1),
        "C30": 1.0e6,
    },
    semiangle_mrad=10.0,
    reference_plane="sample",
    reference_side="entrance",
    metadata={"source": "Ronchigram fit"},
)

result = simulate_raytem_wave(
    "macstem.json",
    start="gun",
    stop="sample",
    source=source,
    probe_aberrations=ronchi,
    extent_A=4096,
    sampling_A=8,
    record=False,
)
```

The aberration phase and angular pupil are applied at the reference plane after
the upstream probe-forming column. The simulation warns when the unwrapped
aberration phase changes by more than pi/2 between adjacent reciprocal-grid
pixels; increase the field of view to refine angular sampling. The current
model expects already fitted coherent Cnm coefficients. Partial coherence,
energy spread, and chromatic averaging are intentionally outside this API.

RayTEM elements may also carry local Cnm coefficients. Round lenses own these
coefficients directly and apply them at an explicit `aberration_plane`:
`"entrance"`, `"principal"`, or `"exit"` (the compatibility default). The
standalone `Aberration` element remains available for deliberate phase screens
that do not belong to a lens. A measured `ProbeAberrationModel` suppresses
upstream element-local coefficients by default because a Ronchigram fit usually
already contains their net effect; downstream lens aberrations remain active.
Set `replaces_upstream_element_aberrations=False` only when the system-level
coefficients are a residual correction that should compose with those upstream
models.

`examples/raytem_column_aberrations.py` provides a complete MACSTEM example:
it decorates CL1–CL3, OL1–OL2, and PL1–PL4 with distinct synthetic Cnm models,
imports them as lens-owned coefficients, and verifies their ownership across
the gun-to-CCD column. It can optionally write the decorated RayTEM JSON with
`--write-config`. The coefficients are illustrative, not a calibration.

The Gaussian source exposes only the parameters needed to define one coherent
wave: voltage, RMS intensity width, wavefront curvature, centroid, and tilt.
Source values in examples are validation inputs, not calibrated MACSTEM
predictions.

Named planes are retained by default so the sampling audit can catch a beam
that clips the grid before later refocusing. Use `record=False` when only the
terminal wave is needed.

The adapter converts RayTEM millimeters to Angstroms and supports drifts, round
lenses and Larmor rotation, steering, thin quadrupoles, regular prisms,
apertures, and Cnm aberrations. Non-symplectic RayTEM matrices are rejected
rather than represented as lossless wave operators. Wave grids are finite and
periodic. The simulation warns when any recorded plane approaches a real- or
reciprocal-space boundary; quantitative work should adjust the field or
sampling and repeat until `result.sampling_report()` is stable.

See `examples/raytem_wave.py` for the command-line form. `OpticalColumn` remains
available for advanced manual construction, but is not required for the normal
RayTEM workflow.

## Data Flow

```
Input Sources          Processing              Analysis            Output
─────────────────────────────────────────────────────────────────────────────
CIF / XYZ / LAMMPS ─┬─→ Loader ─┬─→ ORBMDCalculator ─┐
ASE Atoms / .traj  ─┘           │   (or FAIRChem)    │
                                │                 ↓
                                └───────────→ Trajectory
                                                  │
                                                  ↓
                                          MultisliceCalculator
                                          (Probe → Potential → Propagate)
                                                  │
                                                  ↓
                                              WFData ψ(k,t)
                                                  │
                        ┌─────────────────────────┼─────────────────────────┐
                        ↓                         ↓                         ↓
                   TACAWData                 HAADFData                  WFData
                   FFT(t)→ω                  ∫|ψ|²dΩ                  (direct)
                        │                         │                         │
                        ↓                         ↓                         ↓
                Phonon Dispersion           STEM Image              Diffraction
                Spectral Diffraction        ADF/HAADF/BF            CBED/LACBED
                Spectrum Image                                      4D-STEM
```

## Main Classes

### `Loader`
Load atomic structures and trajectories from various formats.

```python
from pyslice.io.loader import Loader

# Supported: CIF, XYZ, LAMMPS dump, ASE .traj, ASE Atoms objects
traj = Loader("file.cif").load()
traj = Loader("dump.lammpstrj", timestep=0.01, atom_mapping={1: "B", 2: "N"}).load()
```

### `ORBMDCalculator` / `FAIRChemMDCalculator`
Run molecular dynamics with universal ML potentials.

```python
from pyslice.md import ORBMDCalculator

md = ORBMDCalculator(model_name="orb-v3-direct-inf-omat", device="cuda")
md.setup(
    atoms,
    temperature=300,        # K
    timestep=2.0,           # fs
    production_steps=1000,
    save_interval=5,
)
trajectory = md.run()
```

### `Trajectory`
Container for atomic dynamics data.

```python
trajectory.positions   # (n_frames, n_atoms, 3)
trajectory.velocities  # (n_frames, n_atoms, 3)
trajectory.atom_types  # Atomic numbers
trajectory.box_matrix  # (3, 3) simulation cell
trajectory.timestep    # Frame spacing in ps
```

### `MultisliceCalculator`
Compute exit wavefunctions via multislice algorithm.

```python
from pyslice.multislice.calculators import MultisliceCalculator

calc = MultisliceCalculator()
calc.setup(
    trajectory,
    aperture=0,           # mrad (0 = parallel beam)
    voltage_eV=100e3,     # Accelerating voltage
    sampling=0.1,         # Å/pixel
    slice_thickness=0.5,  # Å
    probe_positions=None, # Optional (N,2) array for STEM
)
wf_data = calc.run()
```

### `TACAWData`
Frequency-domain phonon analysis.

```python
from pyslice.postprocessing.tacaw_data import TACAWData

tacaw = TACAWData(wf_data)

# Analysis methods
tacaw.frequencies                        # Available frequencies (THz)
tacaw.spectral_diffraction(freq_THz)     # k-space intensity at frequency
tacaw.dispersion(kx_path, ky_path)       # Phonon dispersion along k-path
tacaw.spectrum_image(freq_THz)           # Real-space map at frequency (STEM)
```

### `HAADFData`
STEM imaging analysis.

```python
from pyslice.postprocessing.haadf_data import HAADFData

haadf = HAADFData(wf_data)
haadf.calculateADF(inner_mrad=60, outer_mrad=200)
haadf.plot()
```

## Examples

See the `tests/` directory for detailed examples:
- `00_probe.py` - Probe wavefunction visualization
- `01_potentials.py` - Atomic potential calculations
- `04_haadf.py` - HAADF-STEM imaging
- `05_tacaw.py` - TACAW phonon spectroscopy
- `06_loaders.py` - Loading various file formats
- `15_molecular_dynamics.py` - MD with ORB potentials

## Requirements

**Core:**
- Python 3.10+
- NumPy, SciPy, Matplotlib
- ASE (Atomic Simulation Environment)
- OVITO

**Recommended:**
- PyTorch (GPU acceleration)

## License

MIT License - see LICENSE file for details.
