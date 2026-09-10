# Getting and building structures

A simulation needs a structure. PySlice takes one from a file, from a crystal
structure database, or from an ASE object — and then builds it into the
oriented, thick, periodic sample you actually want to propagate.

## From a file

```python
from pyslice.io.loader import Loader

trajectory = Loader(filename="structure.cif").load()                 # via ASE
trajectory = Loader(filename="dump.lammpstrj",                       # via OVITO
                    atom_mapping={1: "B", 2: "N"},
                    timestep=0.01).load()                            # ps between frames
```

**The `atom_types` gotcha:** the ASE/CIF path gives element-symbol strings, the
OVITO path gives integer type ids. Always pass `atom_mapping` for LAMMPS files
— otherwise downstream code sees bare integers where it expects symbols.
Loaded arrays are cached as `.npy` siblings of the source file.

## From a database

Search the Materials Project (computed, DFT-relaxed) or the Crystallography
Open Database (experimental, published), then fetch a CIF:

```python
from pyslice.io.databases import search_structures, load_structure_from_database

entries = search_structures("cod", elements=["Ti", "O"], limit=10)   # keyless
trajectory = load_structure_from_database("cod", entries[0]["id"], output_dir="structures")

entries = search_structures("mp", formula="SiC")                     # needs an API key
```

Materials Project needs a free key from
<https://next-gen.materialsproject.org/api>, read from `PYSLICE_MP_API_KEY` or
`MP_API_KEY`. COD needs none — prefer it when you have no key. MP results are
sorted most-stable-first by `energy_above_hull_eV`; a hull value near zero is
the ground-state phase.

Watch for **partial occupancies**, common in COD refinements: they load through
ASE but are not physical for a multislice run. Pick an ordered entry, or build
an ordered approximation, before simulating.

## Building an oriented slab

To simulate "40 nm of [110]-oriented diamond", do not rotate and carve — build
an exactly periodic slab:

```python
from ase.build import bulk
from pyslice.io.build import build_slab

slab, record = build_slab(
    bulk("C", "diamond", a=3.567),
    indices=(1, 1, 0),      # the plane stacked along the beam
    thickness_A=400.0,      # 40 nm, rounded up to whole layers
    min_lateral_A=60.0,     # widened to whole cells
    vacuum_A=0.0,
)
record["layers"], record["box_A"]      # what you actually got
```

`build_slab` stacks the requested plane along the beam with ASE, orthogonalizes
the in-plane cell when it is oblique (hexagonal cells become orthorhombic
automatically), and guarantees a diagonal box — which the multislice grid
requires, since it reads only `np.diag(box_matrix)`.

**Why not rotate and carve?** `Trajectory.rotate_to((h, k, l))` followed by
`slice_positions` gives you a rotated block whose edges are *not*
crystallographically periodic. Under multislice's periodic boundaries those cut
edges wrap around and scatter, which is an artifact source — tolerable for a
probe that stays well away from the edges, damaging for parallel-beam
diffraction. Use it only when no small orthogonal periodic cell exists for your
orientation, and know what you are accepting.

**One convention to be aware of:** `indices` is the ASE *surface plane* (hkl),
and its normal becomes the beam axis. For cubic crystals the (hkl) normal and
the [hkl] direction coincide, so "view down [110]" works as written. For
non-cubic cells they differ — "view down [uvw]" and "expose the (hkl) surface"
are different vectors.

## Then shape it

Everything else is a `Trajectory` transform, and every transform returns a
**new** trajectory:

```python
trajectory = trajectory.tile_positions((4, 4, 1))                    # supercell
trajectory = trajectory.tilt_positions(alpha=0.01, beta=0.0)         # radians
trajectory = trajectory.slice_positions(z_range=(0.0, 50.0))         # crop, Å
trajectory = trajectory.generate_random_displacements(12, 0.06, 0)   # frozen phonon
```

Structure edits are never `setup()` arguments — tiling, orientation, tilt,
cropping, and thermal displacement all happen here, before the calculator sees
the trajectory.

## Verify before you spend compute

A wrong structure wastes the whole run. Check the cheap things first:

```python
trajectory.n_atoms, trajectory.extent, set(trajectory.atom_types)
trajectory.plot(view="xz")            # look at it
trajectory.sea["cell"]["cell"].data   # the cell, calibrated
```

How many frames you need is a physics question, not a preference — see
{doc}`prompted_simulations`, which derives it from your target.
