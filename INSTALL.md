# Installing PySlice

The [README](README.md#installation) is the canonical installation guide.
PySlice requires Python 3.12 or newer; ORB molecular dynamics currently
requires Python 3.12 specifically.

## Recommended editable installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[fast]"
```

Add ORB molecular dynamics support with:

```bash
python -m pip install -e ".[fast,md]"
```

For a NumPy-only installation, use `python -m pip install -e .`. To use
`FAIRChemMDCalculator`, install `fairchem-core` separately.

## Verify the installation

```bash
PYSLICE_BACKEND=numpy python - <<'PY'
from ase.build import bulk
from pyslice import Loader, __version__

trajectory = Loader(
    atoms=bulk("Si", "diamond", a=5.431, cubic=True)
).load()

print(f"PySlice {__version__}")
print(f"{trajectory.n_frames} frame, {trajectory.n_atoms} atoms")
PY
```

Expected output includes one frame and eight atoms. This verifies the package,
ASE conversion, and NumPy path without network access or external data.

Inspect the automatically selected backend with:

```bash
python - <<'PY'
from pyslice import make_backend
b = make_backend()
print(type(b).__name__, b.device, b.float_dtype, b.complex_dtype)
PY
```

`fast` may select Torch CPU when no accelerator is available. Override with
`PYSLICE_BACKEND=numpy` or `PYSLICE_DEVICE=cpu|cuda|mps`.

## Troubleshooting

- If `import pyslice` fails while working from a checkout, activate the same
  environment in which the editable package was installed.
- If ORB fails to install under Python 3.13, recreate the environment with
  Python 3.12 and install the `md` extra.
- For ADF-only runs, use `ADF=(inner, outer)`, `return_layers=None`, and
  `cache_wavefunctions=False` together. `return_layers=None` alone does not
  prevent frame caches.
- `loop_probes` reduces peak device memory, not returned cube size.
  `max_kx`/`max_ky` reduce stored output, not propagation memory.
- Use the [scaling guide](docs/user-guide/scaling.md) before a dense scan.
- Set `PYSLICE_BACKEND=numpy` to force the NumPy backend when diagnosing a
  PyTorch device problem.
- See [troubleshooting](docs/user-guide/troubleshooting.md) for warning
  severity, cache invalidation, and the information to include in a bug report.
