# Multislice grids

Three quantities are easily conflated:

```text
cell length Lx  -> reciprocal spacing Δkx ≈ 1/Lx
real pixel dx   -> Nyquist range 1/(2 dx)
anti-aliasing   -> complete untapered range below roughly (2/3 - 0.02)/(2 dx)
```

`sampling` requests the real-space propagation spacing. PySlice divides the
periodic cell into an integer number of pixels and reports reciprocal axes from
the realized spacing. Decreasing sampling extends reciprocal bandwidth; it
does not refine a STEM scan. Increasing lateral cell length refines reciprocal
spacing.

The pixel counts are nx = int(Lx / sampling) + 1 and likewise ny, which can land
on sizes with large prime factors (e.g. 1279, a prime, or 1138 = 2·569). FFT
libraries, GPU ones in particular, fall back to slower algorithms for such
sizes. `fft_friendly=True` rounds nx and ny up to the next size whose prime
factors are all 2, 3, 5 or 7 (`next_fast_len`). The requested `sampling` then
becomes an upper bound on dx; the reciprocal spacing 1/Lx is unchanged, so the
retained k pixels are the same ones, and the Nyquist and anti-aliasing limits
move out by the rounding (at most 6.5 % for 256 to 4096 pixels). With `min_dk`,
the cropped window is rounded up likewise and keeps Δk at or below `min_dk`.
A parallel beam (`aperture=0`) has unit amplitude per pixel, so its k-space
intensities scale with (nx·ny)²; a convergent probe's do not. The option is off
by default and not supported with PRISM.

`max_kx` and `max_ky` crop returned data after propagation. They reduce result
storage, not propagation memory. `extent` on plotting methods is display/data
cropping, not a substitute for simulation bandwidth.

## Layer return contract

| `return_layers` | Returned wavefunction |
|---|---|
| `-1` | exit wave only |
| `"all"` | every post-transmission slice; potentially enormous |
| `[i, j]` | selected source slice indices |
| `None` or `[]` | empty layer axis; mainly for on-the-fly ADF |

`wf.array[..., j]` is output slot `j`; `wf.layer[j]` is its source slice index.

## Convergence order

1. Decrease `sampling` until the observable stabilizes.
2. Decrease `slice_thickness` independently.
3. Increase lateral cell size when reciprocal sampling is too coarse.
4. Establish the physical bandwidth before applying stored-output crops.
5. Converge configuration count, probe spacing, and detector geometry for the
   actual observable.

Current ordinary multislice is intended for an orthogonal cell propagated
along z. Treat other slice axes, PRISM, `min_dk`, and `kth` as advanced and
verify them against an ordinary-multislice reference.
