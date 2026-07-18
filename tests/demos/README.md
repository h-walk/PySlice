# tests/demos — manual example scripts (NOT the CI suite)

These numbered `NN_*.py` scripts are the historical, human-run examples: they
`plt.show()`, load large `inputs/*.lammpstrj` trajectories, benchmark, and
otherwise need eyes and data that are unavailable on a clean CI runner. They are
**not** collected by pytest (the config only collects `tests/test_*.py`), so they
never run in CI.

Run one by hand from the repository root, e.g.:

```bash
python tests/demos/05_tacaw.py
```

The automated, self-contained suite lives one level up in `tests/test_*.py` and
is what `pytest` runs. When a demo encodes a behaviour worth guarding, port it
into a `test_*.py` case (using the fixtures in `tests/conftest.py`) rather than
relying on the demo.
