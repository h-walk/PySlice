# Results are SEA objects

**With sea-eco installed, every PySlice result already *is* a sea-eco
container.** There is no conversion step, no `to_sea()` call needed to obtain
one, and no separate "SEA mode". You run a simulation and what comes back is a
calibrated `Signal`; you hold a `Trajectory` and its structure is available as
a calibrated `SignalCollection`.

This page is the user-facing view. For why it works this way and how to extend
it, see {doc}`../sea-weeds/resolution_layer`.

## What you get

```python
from ase.build import bulk
from pyslice.io.loader import Loader
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.postprocessing.tacaw_data import TACAWData

trajectory = Loader(atoms=bulk("C", "diamond", a=3.567, cubic=True)).load()
trajectory = trajectory.generate_random_displacements(8, sigma=0.06, seed=0)

calculator = MultisliceCalculator()
calculator.setup(trajectory, aperture=20.0, voltage_eV=100e3, sampling=0.1,
                 slice_thickness=0.5, probe_xs=[1.0, 2.0], probe_ys=[1.0, 2.0],
                 ADF=(60, 200))
wave, haadf = calculator.run()
haadf.calculateADF(60, 200)
tacaw = TACAWData(wave)
```

Every one of `wave`, `haadf`, and `tacaw` is a sea-eco `Signal`:

```python
from pySEA.sea_eco.architecture.base_structure import Signal

isinstance(wave, Signal)        # True
wave.name                       # 'Wavefunction'
wave.dimension_signature        # ['probe', 'time', 'kx', 'ky', 'layer']
wave.Provenance                 # a SEAID, minted at construction
wave.Analysis                   # an AnalysisCollection, ready for lineage
```

So ordinary sea-eco behaviour works on them directly — no bridging:

```python
haadf.show(filename="haadf.png")                 # calibrated axes
tacaw.show(dims="det", backend="plotly")         # interactive
wave.metadata.Simulation.voltage_eV              # 100000.0
tacaw(frequency=12.0)                            # name-keyed calibrated slicing
```

And a trajectory resolves to a structure collection:

```python
structure = trajectory.sea            # implicit; cached
structure["atoms"]["element"].data    # array(['C', 'C', ...])
structure["atoms"]["position"]        # (time, atom, coordinate), Å
structure["cell"]["cell"]             # (cell_vector, coordinate), a/b/c × x/y/z
structure.schema_profile              # 'atomic-structure'
```

That collection conforms to sea-eco's `signal-containers` schema
(`atomic-structure` profile v1), so anything in the ecosystem that understands
atomic structures understands it. See {doc}`../sea-weeds/signal_containers`.

## Saving

Because results *are* SEA objects, saving is just sea-eco:

```python
haadf.to_sea("haadf.sea")                       # sea-eco's own method
```

Reload without PySlice installed anywhere in sight:

```python
from pySEA.sea_eco.io import load
signal = load("haadf.sea")
```

To package results together with the material and sample provenance in one
file, use the MCP tool `pyslice_export_sea_file` (see
{doc}`../ai_tools/mcp`), which writes simulations plus `Materials` entries —
a Material (the unit cell, with its database origin) and a Sample (the built
structure, with its build record).

## Anything can be resolved

If you have a PySlice object and want its SEA form explicitly — for instance
in code that accepts several types — ask the resolution layer:

```python
from pyslice.data.seashell import resolve

resolve(trajectory)          # -> SignalCollection (atomic-structure)
resolve(wave)                # -> the same object; already a Signal
resolve(numpy_array)         # -> a minimal Signal wrapping the array
```

`resolve` is idempotent, so it is safe to call on anything.

## Without sea-eco

PySlice does not require sea-eco. Without it:

- PySlice **imports and simulates unchanged**. Numerics, caching, plotting via
  each class's own `plot()` methods, and every example script still work.
- `sea_available` is `False`:

  ```python
  from pyslice.data.seashell import sea_available
  ```

- SEA-specific operations fail loudly but harmlessly: `.to_sea()` warns, and
  `trajectory.sea` / `resolve(...)` raise `ImportError` naming the fix —
  `pip install 'pyslice[sea]'`.

Nothing silently produces a degraded result. If you get a SEA object, it is a
real one.

## Common mistakes

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: … requires sea-eco` | `[sea]` extra not installed | `pip install -e ".[sea]"` |
| `.to_sea()` only warns, writes nothing | same | same |
| `trajectory.sea` looks stale after a transform | it is not — transforms return **new** trajectories, each resolving on its own | use the returned object, not the original |
| A result has no `Analysis` history | `Analysis` starts empty; lineage accrues as sea-eco pipelines run on it | expected |
