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
