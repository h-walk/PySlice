---
name: analyze-pyslice-results
description: Inspect, plot, transform, compare, export, and scientifically interpret PySlice `WFData`, `TACAWData`, and `HAADFData`, including diffraction, 4D-STEM, detector integration, spectra, spectral diffraction, dispersion, spectrum images, Bose correction, axes, normalization, and convergence. Use when a user has PySlice outputs or caches and asks what they contain, how to visualize or export them, whether a frequency or detector is represented, why results differ, or what conclusions the calculation supports.
---

# Analyze PySlice Results

Identify the data contract first, then separate array manipulation, numerical validation, and scientific interpretation.

## Inspect before transforming

Read:

- `docs/user-guide/results-and-storage.md` for shapes, backend conversion, mutation, caches, and export;
- `docs/user-guide/tacaw-analysis.md` for frequency, normalization, paths, Bose correction, and convergence;
- `docs/user-guide/stem-and-4dstem.md` for detector and 4D-STEM conventions;
- `docs/user-guide/api.md` for current method contracts.

Report the object type, array shape and dtype, backend/device, coordinate ranges and spacings, selected layers, probe positions, frame times or frequencies, and available provenance before plotting.

## Interpret each result type correctly

- Treat `WFData.array` as complex amplitude with shape `(probe, frame, kx, ky, layer)`; compute intensity as `abs(wf.array)**2`.
- Treat `TACAWData.intensity` as `(probe, frequency, kx, ky)` raw arbitrary-unit spectral intensity unless the object was deliberately kept complex.
- Treat `HAADFData.adf` as a two-dimensional scan result produced by a specified disk or annulus.

Convert accelerator data to NumPy deliberately because conversion may copy and transfer it to CPU.

## Analyze TACAW with its frequency budget

Require uniformly spaced, chronological wavefunctions with a nonzero time interval. Compute and report:

```text
delta_f_THz = 1 / (n_frames * dt_ps)
nyquist_THz = 1 / (2 * dt_ps)
```

Treat positive frequencies as loss and negative frequencies as gain. For requested fixed frequencies, call `nearest_frequency()` and report both the request and selected bin.

Remember that reciprocal paths and mask radii use Cartesian cycles per Angstrom, not radians per Angstrom or fractional reciprocal coordinates. `dispersion()` snaps to stored pixels without interpolation.

Do not compare absolute TACAW magnitudes across different frame counts, windows, chunking, reciprocal grids, crops, or Bose states without an explicit normalization. Treat current values as uncalibrated spectral intensity, not an absolute cross section.

Apply Bose correction only to intensity data, at positive temperature, and only once. Record the temperature and mutation state.

## Analyze STEM and diffraction data

Verify that detector boundaries lie within `HAADFData.max_detector_mrad`. Surface incomplete-annulus warnings in the conclusion.

For 4D-STEM, reconstruct scan axes only after confirming a complete Cartesian probe grid and its flattening order. Preserve the distinction between reciprocal data cropping and scan reshaping.

Use plotting `extent` only for display/data selection; do not present it as additional simulated bandwidth.

## Protect mutable results

Copy data before destructive post-processing when the original is still needed. Methods such as wavefunction recentering, cropping, masking, aberration, propagation, decoherence, TACAW FFT replacement, Bose correction, and detector recalculation can mutate the object.

Treat caches as recomputable intermediates. For portable export, store arrays together with axes and separately record commit, environment, input hashes, backend/device, parameters, cache identity, normalization, and selected frequency bins.

## Calibrate the strength of the conclusion

Use this evidence ladder:

1. the object loads and methods execute;
2. shapes, units, axes, and invariants are internally consistent;
3. the observable is stable under relevant numerical refinements;
4. independent trajectories, configurations, or references support the physical claim.

Vary one relevant control at a time: sampling, slice thickness, lateral cell, frame count, saved-frame interval, total duration, probe spacing, detector geometry, reciprocal crop, and normalization. State the highest rung actually established and identify the next discriminating check rather than calling a plausible image or spectrum "validated."

## Deliver an auditable analysis

Provide the exact extraction or plotting code, represented axes/bins, mutation and normalization state, detector/bandwidth checks, comparisons performed, and a calibrated conclusion with residual uncertainties.
