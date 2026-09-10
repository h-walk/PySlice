# The sea-eco resolution layer

**What it is:** one module, `pyslice.data.seashell`, that owns every point of
contact between PySlice and sea-eco, and makes PySlice results *be* sea-eco
containers rather than convert into them.

**Why it exists:** the coupling used to be opt-in and scattered. Results
carried a `.to_sea()` method you had to call to get a SEA form, trajectories
had no SEA identity at all, and four modules imported `pySEA` directly. That
had three consequences worth naming, because they are what the design fixes:

1. The SEA form was *something you asked for*, so most code paths never got
   one, and the ones that did each did it slightly differently.
2. The result classes subclassed `Signal` but **bypassed `Signal.__init__`
   entirely** — no `name`, no `Provenance`, no `Analysis`, no dimension
   signature. They were Signals by declaration and not by behaviour: ordinary
   sea-eco methods raised `AttributeError` on them.
3. A sea-eco rename could break PySlice in four places at once, and did — see
   *Failure modes* below.

**The pattern:** this is rayTEM's `seashells.py` approach
(`sea-ecosystem/rayTEM_original: src/pySEA/rayTEM/seashells.py`), and the
plugin guide in sea-eco's `examples/example_3rd_party/` describes the same
shape. PySlice's version adds a resolver registry, because PySlice has several
result types rather than one.

## Architecture

```
                    pyslice.data.seashell
                    ─────────────────────
   sea-eco present ──→ re-export real Signal / SignalSet /
                       SignalCollection / Dimensions / Metadata /
                       SignalQuantities / SEAID / SEAFile /
                       mark_atomic_structure / validate_atomic_structure
                       resolve()   register_resolver()   adopt_signal_state()
                          │
   sea-eco absent ───→ dummy SEASerializable + dummy Signal *class*
                       resolve() raises with the install command
                          │
                          ▼
   everything else in PySlice imports these names from here
```

Two mechanisms do the work.

### `adopt_signal_state` — results *are* Signals

PySlice's result classes build their own arrays, `Dimensions`, and `Metadata`
in their constructors, and cannot simply delegate to `Signal.__init__` (their
`data` is a lazily-converted backend tensor, and their arrays are assembled
before calibration is known). So each constructor ends with:

```python
adopt_signal_state(self, "Wavefunction")
```

which supplies exactly what `Signal.__init__` would have established and the
class skipped: a display `name`, a minted `Provenance` SEAID, an
`AnalysisCollection` at `.Analysis`, an empty scalar `SignalQuantities`, the
dimension signature, and the plain attribute defaults (`signal_type`,
`is_lazy`, `dimensions_domain`, and friends).

The result is that `WFData`, `HAADFData`, and `TACAWData` behave as Signals
everywhere — `show()`, name-keyed slicing, provenance, and lineage — with no
bridging layer and no conversion.

### `resolve` — everything else becomes a container

```python
resolve(obj, **kwargs)
```

- Already a `Signal`/`SignalSet`/`SignalCollection` → returned unchanged.
  Results hit this branch, which makes `resolve` **idempotent**.
- A registered type → its resolver. `Trajectory` is registered to
  {doc}`the atomic-structure builder <signal_containers>`.
- Array-like → a minimal `Signal` wrapping it.
- Anything else → `TypeError` naming `register_resolver`.

`Trajectory.sea` is a thin cached property over `resolve(self)`. Caching is
safe *because* trajectory transforms return new objects — a tiled or displaced
trajectory is a different object and resolves on its own, so a cache can never
go stale. That property is the whole of the implicitness: no conversion call
appears in user code.

## Design decisions, and what they cost

**Trajectory resolves; it does not inherit.** `Trajectory` could have *been* a
`SignalCollection`, storing atoms as members. It does not, because 128 sites
in `src/pyslice` read `trajectory.positions` / `box_matrix` / `atom_types` /
`velocities`, including the per-frame propagation loop in
`multislice/calculators.py`. Routing that hot path through container member
lookup would risk the numerics for no scientific gain. *Cost:* a trajectory is
not itself a container, so code that wants one must touch `.sea` (or
`resolve`). That is one attribute access, and it is what "implicit" buys.

**Registration is keyed by class name, and explicit.** `register_resolver`
takes `"Trajectory"`, not the class, so `seashell` never imports PySlice
internals — which is what keeps it free of import cycles while sitting at the
bottom of the dependency order. `resolve` calls `_ensure_builtin_resolvers()`
on first use, which imports the built-in resolver module and registers it *by
calling* `register_resolver`. An earlier version relied on import side effects
and could not re-establish itself once the module was already in
`sys.modules`; a test caught that.

**The dummy is a class, not `None`.** PySlice declares
`class WFData(PySliceSerial, Signal)`. If the no-sea-eco branch binds
`Signal = None`, that declaration is a `TypeError: metaclass conflict` at
import time and the entire package becomes unimportable without an optional
dependency. The fallback therefore defines a real dummy `Signal` class.
Containers PySlice never subclasses (`SignalSet`, `SignalCollection`,
`SEAFile`) stay `None`, and `resolve` filters `None` out of its `isinstance`
check.

**Results are converted to plain `Signal`s on SEAFile export, not before.**
sea-eco's generic child loader instantiates nested objects with no arguments,
which a `WFData` cannot satisfy. So `pyslice_export_sea_file` writes plain
`Signal` copies into `Simulations`. A deliberate side benefit: those `.sea`
files open with no PySlice installed at all.

## Invariants

1. **`data/seashell.py` is the only module importing `pySEA`.** Enforced by a
   source scan over `src/pyslice` in `tests/32_seashell_resolution.py`, which
   fails with the offending file and line. `data/atomic_structure.py` and
   `mcp/service.py` take their sea-eco names *from* the layer.
2. **PySlice imports and simulates without sea-eco.** Anything else is a
   regression.
3. **`resolve` is idempotent**, so callers may apply it defensively.
4. **Resolution never silently degrades.** Either you get a real SEA container
   or an explicit `ImportError`/`TypeError` naming the fix.
5. **`adopt_signal_state` is tolerant by design.** It runs inside every result
   constructor, so an individual step that a future sea-eco makes unnecessary
   is skipped rather than raised.

## Extension points

**A new result type.** If it subclasses `Signal`, call `adopt_signal_state` at
the end of its constructor and it is done — `resolve` already returns it
unchanged.

**A new non-Signal type** (a detector model, an instrument description):

```python
from pyslice.data.seashell import register_resolver

def my_thing_to_sea(obj, **kwargs):
    ...  # build and return a Signal / SignalSet / SignalCollection

register_resolver("MyThing", my_thing_to_sea)
```

Register from the module that owns the type, or add it to
`_ensure_builtin_resolvers` if it should always be available.

**A new sea-eco name.** Add it to the import block in `seashell.py` and give
it a `None` (or a dummy class, if PySlice subclasses it) in the fallback.
Never import it elsewhere.

## Failure modes

| Failure | Symptom | Guard |
|---|---|---|
| sea-eco absent | `ImportError` from `resolve`/`.sea`, warning from `to_sea` | `require_sea`, dummy classes |
| sea-eco renames or moves a name | One `ImportError` at the layer, not four scattered ones | Single import site + `tests/32` scan |
| A sea-eco name is bound to `None` that PySlice subclasses | `TypeError: metaclass conflict` at import | Dummy `Signal` class; no-sea path is tested |
| Resolver registered only as an import side effect | Silent `TypeError` if the module was already imported | Explicit registration in `_ensure_builtin_resolvers` |
| Result constructor skips `adopt_signal_state` | `AttributeError` from ordinary Signal methods | `tests/32` asserts name, Provenance, Analysis, signature per class |

Historical note, because it explains the shape of the code: the pre-existing
bridge caught its sea-eco import in a bare `except Exception`, so when sea-eco
removed `base_structure_numpy` the failure was **invisible** — `Signal` became
a stub and `Dimensions` became `None`, which made every
`if Dimensions is not None:` block skip. Every `.to_sea()` call in PySlice was
silently a no-op, and a test was passing *because* of it. Localizing the import
is what makes that class of failure loud.

## Limitations

- Resolution is **write-only**: PySlice produces SEA containers but has no
  reader that turns an `atomic-structure` collection back into a
  `Trajectory`. Ingest is `Loader` (CIF/XYZ/LAMMPS/ASE).
- `Trajectory` is not itself a container (see *Design decisions*).
- `adopt_signal_state` tracks what `Signal.__init__` currently sets. A future
  sea-eco that adds required state needs a matching addition here; the tests
  assert the fields that matter today.
- Only `Trajectory` has a built-in resolver. Probes, potentials, and
  calculators are not resolvable — nobody has needed them to be.

## Provenance and verification

| Aspect | Where |
|---|---|
| Implementation | `src/pyslice/data/seashell.py`; `src/pyslice/data/atomic_structure.py`; `Trajectory.sea` in `src/pyslice/multislice/trajectory.py`; `adopt_signal_state` calls in `src/pyslice/postprocessing/{wf_data,haadf_data,tacaw_data}.py` |
| Serialization bridge | `src/pyslice/data/pyslice_serial.py` (imports names from the layer) |
| Pattern precedent | `sea-ecosystem/rayTEM_original: src/pySEA/rayTEM/seashells.py` and its wiki page; `sea-eco/examples/example_3rd_party/basic_example.md` |
| Contract implemented | sea-eco `signal-containers` schema — see {doc}`signal_containers` and `docs/conformance/signal-containers.md` |
| Focused tests | `tests/32_seashell_resolution.py` (both paths, idempotence, caching, self-registration, single-coupling-point scan); `tests/31_sea_file_export.py` (SEAFile packaging, schema validation on reload) |
| User guide | {doc}`../guides/sea_results` |
| AI-tool artifacts | `pyslice_export_sea_file` and `pyslice_render_signal` in `src/pyslice/mcp/server.py`; `skills/structure-retrieval/SKILL.md` |
| API reference | {doc}`../api_reference` — `pyslice.data.seashell`, `pyslice.data.atomic_structure` |
