# TACAW analysis

TACAW subtracts the temporal mean and applies an FFT over uniformly spaced
saved wavefunctions. Default output is raw `|FFT|²` in arbitrary units.

## Frequency budget

```python
import numpy as np

dt_ps = float(np.mean(np.diff(wf.time)))
n = len(wf.time)
print("Δf:", 1 / (n * dt_ps), "THz")
print("Nyquist:", 1 / (2 * dt_ps), "THz")
```

Frequencies are `fftshift(fftfreq(n, d=dt_ps))`. They are signed; PySlice calls
positive frequency loss and negative frequency gain. Fixed-frequency methods
select the nearest bin:

```python
requested = 15.0
actual = tacaw.nearest_frequency(requested)
pattern = tacaw.spectral_diffraction(actual)
print(f"requested {requested:g}; using {actual:g} THz")
```

## Normalization

Current TACAW values are not calibrated cross sections. Spectra sum reciprocal
pixels without `Δkx Δky` or solid-angle weighting. Magnitudes depend on frame
count, windowing/chunking, reciprocal sampling/crop, and Bose state. Compare
absolute values only under identical grids and an explicitly chosen downstream
normalization.

## Chunking

| Setting | Effect | Scientific consequence |
|---|---|---|
| `chunkFFT=True` | batches kx (GPU: 16 MiB target; CPU: one column) | same full-time FFT |
| `fft_batch_max_bytes=N` | enables spatial batching with a byte target | does not change frequency resolution; see [memory limits](scaling.md) |
| `chunk_size_time=None` | one full window | best available `Δf` |
| `chunk_size_time=M` | sums block intensities | coarser `Δf`; block phases discarded |

## k paths and masks

Paths use Cartesian simulation-frame cycles/Å, not radians/Å or fractional
reciprocal coordinates. PySlice does not infer a Brillouin zone. `dispersion()`
snaps each requested point to the nearest stored pixel without interpolation.
Use a path no denser than the effective reciprocal grid and check endpoints.

Round-mask radii are also cycles/Å. Convert a detector semi-angle by
`radius = angle_rad / tacaw.probe.wavelength`.

## Bose correction and optional gain/loss folding

Folding defaults to `fold=False`, including when Bose correction is enabled.
`apply_bose=True` multiplies the existing signed-frequency intensity by
`βhν / (1 - exp(-βhν))` without imposing inversion symmetry.

Set `fold=True` to first average the inversion-related classical estimates

```text
I(q, -frequency) = I(-q, frequency)
```

Pair averaging preserves spectral weight and enforces this symmetry. Use it
when that symmetry is appropriate for the observable being analyzed. Folding
can be enabled independently of Bose correction:

```python
tacaw = TACAWData(wf, apply_bose=True, temperature_K=300.0)  # fold=False
folded = TACAWData(wf, apply_bose=True, temperature_K=300.0, fold=True)
classical_folded = TACAWData(wf, fold=True)  # no Bose weighting
```

For an existing, uncorrected object, call `apply_bose_correction(300.0, fold=True)`
to fold before weighting, or `fold_gain_loss()` to fold alone. The method
also defaults to `fold=False`; it does not undo an earlier fold. When folding
is enabled before weighting, the retained gain and loss partners obey the
imposed detailed-balance relation. FFT caches retain the uncorrected,
unfolded data so later calls can select either option.

For an even number of frames, the single Nyquist sample has no distinct
positive/negative partner. It is treated as the periodic self-bin; do not use
that edge bin for a quantitative gain/loss comparison.

Both operations mutate the object, reject complex-amplitude data, and record
their state. Bose correction also rejects nonpositive temperature and cannot be
applied twice. Record temperature, folding, and correction state when comparing
runs.

`keep_complex=True` and `space="real"` are advanced representations whose
physical analysis contract is narrower than the default reciprocal intensity
workflow. Do not treat an inverse FFT of `|FFT|²` as a real-space spectral
intensity; use scanned-probe `spectrum_image()` for spatial maps.

## Convergence

Check trajectory stationarity, saved-frame Nyquist, total-time resolution,
independent trajectories/blocks, cell size, multislice sampling and slices,
reciprocal crop, detector/probe geometry, selected frequency bin, and identical
normalization/Bose state.
