# PySlice

PySlice simulates electron scattering from atomic structures and trajectories.
It supports TEM diffraction, STEM/ADF/4D-STEM, and TACAW vibrational EELS, with
optional molecular dynamics through ORB or FAIRChem models.

## Installation

PySlice requires Python 3.12 or newer. Install the accelerated multislice path
with:

```bash
git clone https://github.com/h-walk/PySlice.git
cd PySlice
python -m pip install -e ".[fast]"
```

Add ORB molecular dynamics with `python -m pip install -e ".[fast,md]"` under
Python 3.12. A NumPy-only installation is available with
`python -m pip install -e .`. See [INSTALL.md](INSTALL.md) for environment,
backend, and verification details.

For a small checkout without the historical simulation datasets, see the
[lightweight checkout instructions](INSTALL.md#lightweight-checkout) below.

## Quick start

This small calculation runs on CPU, downloads no model, and produces one static
TEM diffraction pattern:

```python
from ase.build import bulk
from pyslice import Loader, MultisliceCalculator

trajectory = Loader(
    atoms=bulk("Si", "diamond", a=5.431, cubic=True) * (2, 2, 1)
).load()

calc = MultisliceCalculator()
calc.setup(
    trajectory,
    aperture=0,
    voltage_eV=100e3,
    sampling=0.2,
    slice_thickness=0.5,
    cache_wavefunctions=False,
)
wf = calc.run()
wf.plot_reciprocal(powerscaling=0.25, nuke_zerobeam=True)
```

This is a smoke calculation, not a convergence study. The annotated version is
[examples/tem_diffraction.py](examples/tem_diffraction.py).

## Choose a workflow

| Goal | Start with | Cost |
|---|---|---|
| Load a structure or trajectory | [loading guide](docs/user-guide/loading-trajectories.md) | light |
| Run TEM diffraction | [tem_diffraction.py](examples/tem_diffraction.py) | light |
| Run ADF/HAADF without storing 4D-STEM | [haadf_stem.py](examples/haadf_stem.py) | intermediate |
| Analyze an existing trajectory with TACAW | [tacaw_from_trajectory.py](examples/tacaw_from_trajectory.py) | substantial |
| Generate MD, then run TACAW | [MD-to-TACAW guide](docs/user-guide/md-to-tacaw.md) | substantial; optional model download |
| Reproduce publication-scale workflows | [examples index](examples/README.md#reproduction-scale) | advanced |

The [examples index](examples/README.md) identifies canonical, advanced, and
reproduction-scale scripts. [`example.ipynb`](example.ipynb) is the longer
notebook tutorial. Numbered files in `tests/` are regressions, not tutorials.

## Contracts worth knowing

- `Trajectory.timestep` is the spacing between saved frames in ps, not the MD
  integrator step. A wrong value rescales the entire TACAW frequency axis.
- `Trajectory` stores one fixed cell. NPT is allowed for ASE equilibration, then
  PySlice freezes the final cell for NVT or NVE production. NPT production and
  variable-cell trajectory loading are rejected.
- Reciprocal coordinates are spatial frequencies in cycles/Å. A period `a`
  corresponds to `1/a`; add `2π` only when converting to angular wavevector.
- `sampling` controls the propagation grid, not STEM scan spacing.
  `max_kx`/`max_ky` reduce stored output but not propagation cost.
- `WFData.array` is complex wave amplitude; intensity is `abs(psi)**2`.
- TACAW currently returns FFT-derived arbitrary units and selects the nearest
  represented frequency bin. Compare runs only after checking time-window and
  reciprocal-grid convergence.
- For ADF-only work, use `ADF=(inner, outer)`, `return_layers=None`, and
  `cache_wavefunctions=False` to avoid retaining a 4D-STEM cube.

## Focused guides

- [Loading trajectories](docs/user-guide/loading-trajectories.md)
- [Multislice grids](docs/user-guide/multislice-grids.md)
- [STEM and 4D-STEM](docs/user-guide/stem-and-4dstem.md)
- [TACAW analysis](docs/user-guide/tacaw-analysis.md)
- [Molecular dynamics to TACAW](docs/user-guide/md-to-tacaw.md)
- [Results and storage](docs/user-guide/results-and-storage.md)
- [Scaling, memory, and disk](docs/user-guide/scaling.md)
- [Troubleshooting](docs/user-guide/troubleshooting.md)

## License

MIT License. See [LICENSE](LICENSE).
