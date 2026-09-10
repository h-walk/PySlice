# Getting started

PySlice simulates electron microscopy: conventional TEM/STEM/4D-STEM imaging
and diffraction through the multislice algorithm, and vibrational EELS
(phonon spectroscopy) through the TACAW method applied to molecular-dynamics
trajectories.

## Install

```bash
pip install -e .                 # core: ase, numpy, scipy, matplotlib, tqdm, h5py, ovito
pip install -e ".[fast]"         # + torch, for GPU acceleration
pip install -e ".[sea]"          # + sea-eco, so results are calibrated SEA objects
pip install -e ".[md]"           # + ORB machine-learned potentials (pins Python 3.12)
pip install -e ".[mcp]"          # + the MCP server, to drive PySlice from an LLM client
```

Verify the install:

```bash
python -c "import pyslice; print(pyslice.__version__)"
python -c "from pyslice.data.seashell import sea_available; print('SEA objects:', sea_available)"
```

Two install notes that cause most first-run problems:

- **OVITO is not on ordinary PyPI.** Install it with
  `--find-links https://www.ovito.org/pip/`. It is needed *only* for the
  XYZ / LAMMPS-dump / ASE-trajectory loading path; CIF loading and everything
  downstream work without it.
- **`requires-python >= 3.12`**, driven by the MD stack.

Install `[sea]` unless you have a reason not to: without it PySlice still
simulates, but results are plain PySlice objects rather than calibrated SEA
containers, and `.sea` export is unavailable. See {doc}`sea_results`.

## Notes on the `pySEA` namespace

PySlice is distributed as **`PySlice`** and imports as **`pyslice`**. It is the
one pySEA-ecosystem simulation package that does *not* live under the shared
`pySEA.*` namespace — sibling packages import as `pySEA.sea_eco`,
`pySEA.sea_sand`, `pySEA.polly`, and so on, while PySlice keeps its own
top-level `pyslice` package.

The namespace is deliberately modular: each package installs into the shared
`pySEA` directory and is usable on its own, so you take only the pieces you
need. PySlice's one ecosystem coupling is optional — sea-eco, via the `[sea]`
extra — and it is confined to a single module (`pyslice.data.seashell`).

```python
import pyslice                                   # this package
from pyslice.io.loader import Loader
from pySEA.sea_eco.io import load as load_sea    # sibling package, optional here
```

For the current list of ecosystem packages and what each owns, see the
generated catalog in the
[ecosystem AI wiki](https://github.com/sea-ecosystem/sea-ecosystem) rather than
any list copied into these docs.

## The smallest useful workflow

Load a structure, propagate it, look at the result:

```python
from ase.build import bulk
from pyslice.io.loader import Loader
from pyslice.multislice.calculators import MultisliceCalculator

trajectory = Loader(atoms=bulk("C", "diamond", a=3.567, cubic=True)).load()

calculator = MultisliceCalculator()          # auto-selects CUDA / MPS / CPU
calculator.setup(
    trajectory,
    aperture=0.0,          # 0 mrad = parallel beam (TEM / diffraction)
    voltage_eV=100e3,
    sampling=0.05,         # Å per pixel
    slice_thickness=0.5,   # Å
)
wave = calculator.run()                       # blocking, with a progress bar

print(wave.name, wave.data.shape)             # 'Wavefunction' (probe, time, kx, ky, layer)
wave.show(filename="diffraction.png")         # calibrated axes, via sea-eco
```

`wave` is a sea-eco `Signal`. Nothing was converted: see {doc}`sea_results`.

## Units, and the two that bite

| Quantity | Unit |
|---|---|
| Lengths — sampling, defocus, slice thickness, probe positions | Å |
| Angles — aperture, detector inner/outer | mrad |
| Accelerating voltage | eV (`100e3` = 100 kV) |
| **Trajectory frame spacing** (`Trajectory.timestep`) | **ps** |
| **MD integration timestep** (`MDCalculator.setup(timestep=…)`) | **fs** |
| TACAW frequencies | THz (1 THz ≈ 4.136 meV) |
| k-axes | 1/Å |

The two that bite:

- **ps versus fs.** Frame spacing is picoseconds; the MD integrator step is
  femtoseconds. Mixing them silently changes your TACAW frequency window by
  three orders of magnitude.
- **`aperture=0` means parallel beam** (TEM, diffraction, momentum-resolved
  work). Any value above zero makes a convergent STEM probe.

## Where to go next

| You want to | Read |
|---|---|
| Understand why results are already SEA objects | {doc}`sea_results` |
| Get a structure from a database, or build an oriented slab | {doc}`structures` |
| Describe a simulation in words and have it planned | {doc}`prompted_simulations` |
| Drive PySlice from an LLM client | {doc}`../ai_tools/mcp` |
| Change PySlice itself | {doc}`../sea-weeds/index` |
