---
name: run-pyslice-simulations
description: Design and run user-facing PySlice multislice calculations for TEM diffraction, CBED, STEM, ADF/HAADF/BF imaging, 4D-STEM, or wavefunction generation while controlling grids, detector bandwidth, backend, memory, disk, caching, and convergence. Use when a user asks to simulate electron scattering, choose microscope parameters, adapt an example, estimate a run, resolve an out-of-memory problem, or produce `WFData` or `HAADFData` from a prepared trajectory.
---

# Run PySlice Simulations

Translate a microscopy goal into the smallest credible PySlice calculation, then scale only after inspecting its numerical and resource behavior.

## Choose the retained result first

Classify the goal before configuring `MultisliceCalculator`:

- TEM diffraction or one CBED pattern: retain the exit-wave diffraction data.
- One annular STEM image: integrate with `ADF=(inner_mrad, outer_mrad)`, set `return_layers=None`, and disable wavefunction caching.
- Multiple virtual detectors or COM/DPC: retain a 4D-STEM diffraction cube.
- TACAW: retain uniformly sampled wavefunctions over all trajectory frames.

Do not store a diffraction cube merely because it is available; it can dominate memory and disk.

## Ground the setup in live guidance

Read the relevant files before choosing parameters:

- `docs/user-guide/multislice-grids.md` for sampling, reciprocal bandwidth, layers, and convergence;
- `docs/user-guide/stem-and-4dstem.md` for scans and detector planning;
- `docs/user-guide/scaling.md` before a dense scan or long trajectory;
- `docs/user-guide/api.md` for the current `setup()` and return contracts;
- `examples/tem_diffraction.py` or `examples/haadf_stem.py` for canonical patterns.

Use the stable top-level API: `Loader`, `Trajectory`, and `MultisliceCalculator`.

## Keep the three grids distinct

- `sampling` is the real-space propagation pixel size and controls reciprocal bandwidth.
- `probe_xs` and `probe_ys` are scan coordinates and control image-pixel spacing.
- specimen lateral size controls reciprocal-grid spacing, approximately `1/L`.

Decreasing `sampling` does not refine the scan. Increasing the lateral cell does not extend the Nyquist range. Use `endpoint=False` for periodic scan axes unless both edges are physically distinct.

## Bound cost before execution

Estimate returned wavefunction storage:

```text
bytes ~= probes * frames * retained_kx * retained_ky * returned_layers
        * bytes_per_complex
```

Account for caches and temporary copies. Use `loop_probes` to reduce peak device memory; use reciprocal crops to reduce returned storage; use on-the-fly ADF to avoid retaining diffraction patterns. Do not claim that `max_kx` or `max_ky` solves propagation-memory pressure because cropping occurs after propagation.

## Execute in stages

1. Verify the installed backend and input trajectory.
2. Run one frame and one probe without unnecessary caches.
3. Inspect shapes, realized reciprocal axes, detector bandwidth, runtime, RAM/device memory, and disk.
4. Add configurations, then a small scan.
5. Extrapolate to the proposed production run.
6. Increase scale only when the user has requested or accepted the cost.

Treat PRISM, `min_dk`, `kth`, non-z slice axes, and nonzero PRISM defocus as advanced. Verify advanced paths against ordinary multislice or stop when the live API marks the combination unsupported.

## Respect detector and layer contracts

Use `HAADFData.max_detector_mrad` for the realized result. Treat warnings about cropped or tapered annuli as scientific limitations, not cosmetic log noise.

Remember:

- `return_layers=-1` returns the exit wave;
- `return_layers="all"` may be enormous;
- `return_layers=[...]` stores selected source slices;
- `return_layers=None` or `[]` returns an empty layer axis.

## Handle caches honestly

Treat a cache as disposable acceleration, not a checkpoint or archival result. Record `save_path`, cache settings, and reuse messages. Use `run(force_rerun=True)` when testing whether reuse explains a result.

## Report the outcome

Return runnable code, the actual backend/device and dtypes, input and output shapes, realized axes, selected layers, detector limits, cache behavior, and the estimated or observed cost. Distinguish a successful smoke test from a converged production calculation.
