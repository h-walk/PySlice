---
name: prepare-pyslice-trajectories
description: Prepare atomic structures and trajectories for PySlice by loading CIF, XYZ, LAMMPS, ASE, or in-memory structures; checking geometry, elements, cells, and timing; applying trajectory transforms; or planning optional molecular dynamics. Use when a user wants to import, inspect, crop, tile, subsample, generate, or troubleshoot simulation inputs before multislice, STEM, frozen-phonon, or TACAW work.
---

# Prepare PySlice Trajectories

Build a simulation-ready `Trajectory` without mistaking successful parsing for physical validity.

## Establish the intended use

Determine whether the trajectory will support:

- a static diffraction or STEM calculation;
- an unordered frozen-phonon/configuration ensemble;
- a uniformly sampled time series for TACAW;
- a new molecular-dynamics run.

Preserve chronology and a nonzero saved-frame timestep for TACAW. Treat randomly selected frames and generated frozen-phonon configurations as ensembles, not time series.

## Use the live contract

Read the relevant repository guidance before constructing the input:

- `docs/user-guide/loading-trajectories.md` for formats, units, caches, and transforms;
- `docs/user-guide/md-to-tacaw.md` before generating molecular dynamics;
- `examples/loading_trajectories.py` for canonical loading patterns.

Prefer the top-level public API: `from pyslice import Loader`.

## Load by source type

- For an ASE `Atoms` object or fixed-cell sequence, pass `atoms=...`.
- For a LAMMPS dump, provide a complete `atom_mapping` and the spacing between saved frames in ps.
- For an ASE trajectory, load all frames explicitly and provide the saved-frame spacing in ps.
- For CIF or XYZ, inspect the inferred cell and coordinate convention rather than trusting the extension.

Do not infer timing from a filename. Distinguish the MD integrator step in fs from `Trajectory.timestep`, which is the interval between saved frames in ps.

## Preflight the result

Inspect and report:

1. `positions.shape == (frame, atom, 3)` and finite coordinates in Angstrom;
2. stable atom count, identity, and ordering;
3. unique element identities and any applied mapping;
4. the full `box_matrix`, positive volume, and coordinate bounds;
5. frame count and saved-frame timestep;
6. beam direction and representative first, middle, and last frames.

Require an axis-aligned orthogonal cell for ordinary multislice. Use `fold_positions_to_orthogonal_box()` only as an explicit representation choice, then verify periodic neighbors and the projected structure.

## Transform deliberately

- Use `tile_positions(...)` to build a supercell and scale its cell.
- Use `slice_positions(...)` only after verifying its orthogonal-box and coordinate-shift semantics.
- Use chronological timestep slicing for TACAW and update the saved-frame spacing consistently.
- Use random-frame selection or random displacements only for incoherent configuration ensembles.
- Unwrap periodic trajectories before interpreting Cartesian displacements.

After each transform, repeat the preflight rather than assuming invariants survived.

## Generate MD only when requested

Treat ORB and FAIRChem models as optional external models, not validated phonon references. Before a substantial or downloading run, establish the model, material, ensemble, device, runtime scale, output directory, and seed.

Use NVT for equilibration and test NVE for spectral production. Do not use NPT for TACAW because PySlice trajectories require one fixed cell. Record the actual saved-frame spacing:

```text
frame_dt_ps = timestep_fs * save_interval / 1000
```

Use a new output directory for a new realization unless overwrite is explicitly intended.

## Hand off a trustworthy input

Provide the loading code or command, the preflight findings, all transforms, and any residual concerns. State explicitly whether the result is a static structure, configuration ensemble, or uniform time series, and whether it is ready for TACAW.
