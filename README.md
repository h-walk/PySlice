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
# Clone the repository ("git clone https://github.com/h-walk/PySlice" will work, but here we exclude the tests folder so as to greatly increase download speed and reduce disk usage)
git clone --filter=blob:none --no-checkout https://github.com/h-walk/PySlice
cd PySlice/
git sparse-checkout set --no-cone '/*' '!/tests/*'
git checkout main

# Install with pip. -e = editable mode. [fast] will install torch (technically optional, but provides extreme speed improvements).
pip install -e ".[fast]"

# Or using uv (recommended)
uv sync

# Optional ORB/MD support currently requires Python 3.12 because ORB's dm-tree
# dependency does not publish Python 3.13 wheels.
uv python install 3.12
uv sync --python 3.12 --extra fast --extra md
```

## Quick Start

### Basic Multislice: HAADF-STEM Imaging

```python
from pyslice import Loader,MultisliceCalculator,HAADFData
import numpy as np

# Load your cif file
trajectory = Loader("SiO2.cif").load()

# single unit-cell imported, need to tile to create a slab
trajectory = trajectory.tile_positions([10,10,3])

# "frozen phonon" technique: create many snapshots (or "frames") of atomic configurations, with random (gaussian) atomic displacements to emulate atomic motion
trajectory = trajectory.generate_random_displacements(10, 0.3, seed=0) # 10 snapshots, 0.3 sigma value for gaussian, random seed for reproducible results

# Define probe scan grid over a 2x2 unit-cell grid
a = trajectory.box_matrix[0,0] ; b = trajectory.box_matrix[1,1]
xs = np.linspace(a,3*a,18,endpoint=False) ; ys = np.linspace(b,3*b,16,endpoint=False)

# MultisliceCalculator object is responsible for slicing / wave propagation
calc = MultisliceCalculator()
calc.setup(trajectory, aperture=30, voltage_eV=100e3, sampling=0.1, probe_xs=xs, probe_ys=ys) # aperture is in mrad, voltage is in eV
wf_data = calc.run()

# Exit waves are stored as a WFData object, which has an "array" attribute, with probe,snapshot,kx,ky,layer indices. But we don't want the user to need to sift through these on their own. instead, we provide standardized functions/objects for things like ADF (coherent summation of the exit wave around an annular detector). 
haadf = HAADFData(wf_data)
haadf.calculateADF(inner_mrad=60, outer_mrad=200)
haadf.plot(filename="quartz_ADF.png")
```

### Basic Multislice: TEM Diffraction

```python
from pyslice import Loader,MultisliceCalculator
import numpy as np

# Load your trajectory
trajectory = Loader( "hBN.lammpstrj",atom_mapping={1: "B", 2: "N"}).load() # You can also load atomic positions from LAMMPS dump files, but you must specify the mapping of atom types
    
# Optional cropping in time and space
trajectory = trajectory.get_random_timesteps(5).slice_positions([0,20],[0,20])

calc = MultisliceCalculator()
# for TEM diffraction, we'll use a parallel beam (aperture = 0 mrad). If you opt for an ultra-small convergence angle, check that the probe is contained within the simulation bounds. e.g: calc.setup(trajectory, aperture=5, voltage_eV=100e3, sampling=0.1) ; calc.base_probe.plot(filename="5mrad.png")
calc.setup(trajectory, aperture=0, voltage_eV=100e3, sampling=0.1)
wf_data = calc.run()

wf_data.plot(powerscaling=0.125)  # Diffraction pattern

```

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
    timestep=0.005,  # If using your lammps trajectory for TACAW, it is important to specify the timestep size, or your frequencies may be incorrect!
    atom_mapping={1: "B", 2: "N"}
).load()

# ASE trajectory or CIF/XYZ file
trajectory = Loader("silicon.cif").load()
```

### Tips and Tricks
_**Are you running our of memory?**_ We offer a few flags to mitigate this, but first it is important to understand _why_ we are running out of RAM. Fundamentally, there is an exit wave which describes a wave-function, sampled across a grid of points (for a system of length L, discretized to $\Delta l$, yielding $L/ \Delta l$ points, for both $x$ and $y$). We also attempt to run the calculation across $P$ probes simultaneously through each of $T$ timesteps (or snapshots), meaning we must store $P \times T \times nx \times ny$ values in memory, just for the exit wave. Holding this all in memory makes post-processing faster (e.g., summation across snapshots for calculation of an ADF image), so we prefer to write to disk if and only if we are required to. 

Below are a list of flags that can be passed to MultisliceCalculator:
- 'max_kx' and 'max_ky': this will crop your exit wave. $k = 1/ \Delta l$, and while fine real-space resolution may be required (0.1 $\mathring{A}$ by default), sampling out to high $k$ likely is not. 
- 'kth': since $\Delta k = 1/L$, large systems may have unnecessarily fine k-space resolution. this will add binning to your exit wave
- 'min_dk': since $\Delta k = 1/L$, large systems may have unnecessarily fine k-space resolution. instead of binning your exit-wave in-post (via 'kth' above), we can also spatially-crop your probe (be cautious of boundary effects!), which will naturally produce a lower-res exit wave. This is effectively propagating a probe through a cropped sub-region of your system.
- 'loop_probes': while we would prefer to process all probe positions simultaneously, high-res ADF images will easily blow your RAM. this specifies the number of probes to process simultaneously. 
- 'use_memmap': while we would prefer to hold everything in memory, we offer the ability to use memmapping to store the large exit wave array on disk instead. this may come with a severe performance reduction however. 
- 'ADF': in the HAADF example above, we calculated all exit waves (full datacube $P \times T\times kx \times ky$), then calculated the ADF signal afterwards (coherent summation around the annular detector: a ring in k-space). We can instead calculate ADF on the fly, meaning we don't need to store the full exit wave (it is effectively "compressed" across $kx$ and $ky$). We may still have intermediate datacubes ($P \times kx \times ky$), but this can be combined with 'loop_probes' if system sizes are particularly large. 

_**Noticing excessive disk usage?**_ We save caches of the exit waves, which means your script can be re-run and the potentially-expensive multislice steps can be skipped (provided that your atomic configurations, probe positions and run parameters (accelerating voltage, convergence angle, etc) are the same). This is particularly useful if you have long runs (large systems, high-res ADF with many probes, and/or TACAW with many timesteps): if your run is interrupted, it can be resumed from where it left off. 

To turn off or adjust the level of caching, use the following:
- 'cache_wavefunctions': defaults to True, but can be set to False to avoid disk consumption.
- 'cache_potentials': defaults to False, but can be set to True to further expedite resumption

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
