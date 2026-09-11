# Scaling, memory, and disk

Estimate returned wavefunction storage before a run:

```text
bytes ≈ probes × frames × retained_kx × retained_ky × returned_layers
        × bytes_per_complex
```

PySlice currently uses complex128 (16 bytes) on NumPy, Torch CPU, and CUDA;
MPS uses complex64 (8 bytes). Frame caches can add another copy. Benchmark one
frame and a small probe batch, then extrapolate runtime approximately with
`frames × slices × probes × nx × ny × log(nx×ny)`.

| Option | Reduces device peak? | Reduces result/disk? | Caveat |
|---|---:|---:|---|
| `loop_probes=N` | yes | no | tune from 1–8 upward |
| `potential_cache_max_bytes` | caps retained potential volume | no | default 16 MiB, only for multiple probe batches; 0 disables |
| `use_memmap=True` | RAM pressure | moves output to disk | prefer fast local storage |
| `max_kx/max_ky` | no | yes | post-propagation crop; also limits detectors |
| `kth` | no | yes | changes retained reciprocal sampling |
| `min_dk` | yes | yes | changes propagation window/physics; advanced |
| `return_layers=None` | sometimes | yes | pair with on-the-fly ADF and cache off |
| `cache_wavefunctions=False` | no | disk | disables frame reuse |
| on-the-fly `ADF` | yes | dramatically | discards later virtual-detector analysis |

If `max_kx` does not fix GPU OOM, that is expected: it crops returned data
after propagation. If `loop_probes` does not shrink disk use, that is also
expected: it changes batch size, not result shape.

## Reusing work without large caches

Atom membership in each slice/element group is indexed once per frame. The
Kirkland parameter text is parsed once per process; only its small immutable CPU
table is shared, with independent arrays supplied to each backend. Uncropped
transmission and Fresnel operators broadcast across scan positions within each
energy copy. Cropped-probe index arrays are also computed once per propagation.
None of these changes reduces precision or changes the physical sampling.

Within each calculator run, reciprocal grids and element form factors are shared
across frames; Fresnel/anti-alias planes are shared across probe batches too.
These are run-local, not global caches: each `run()` rebuilds them, including
after changes to energy copies, geometry, or backend. CPU probe-selection
metadata is prepared once. Probe positioning transforms the template once per
batch and applies all phase shifts together, bounded by `loop_probes`.

For `loop_probes`, a potential volume is reused across batches only if the whole
frame fits `potential_cache_max_bytes` (default `16 * 1024**2` bytes). This is a
cap on retained **potential** data, not total simulation memory; waves and FFT
temporaries still require space. It never caches multiple frames. Disable it
with `potential_cache_max_bytes=0` to keep strictly slice-at-a-time generation.

Larger volumes still stream. Setting `cache_potentials=True` can avoid repeated
potential calculations in that case by writing slice files and reading them for
subsequent probe batches. This trades disk space/I/O for computation; it remains
off by default. No extra files are created by the bounded in-memory reuse.

## Output and spectral batching

The real-space multislice path writes completed probe batches directly into the
final wavefunction array, avoiding a duplicate full-frame output buffer. With
`use_memmap=True`, this final array is disk-backed. Frame cache files are only
written when `cache_wavefunctions=True`; cache reads use memory maps and upload
only the selected batch. PRISM retains its separate scattering-matrix storage.
Without memmapping, the final all-frame output still occupies device memory.

`TACAWData(..., fft_batch_max_bytes=16 * 1024**2)` enables spatially batched FFTs.
`chunkFFT=True` uses the same 16 MiB target on GPUs when no explicit budget is
supplied; CPU execution retains one-column batching unless a budget is supplied.
The default unchunked path remains unchanged. The target estimates four complex
work arrays and chooses how many kx columns to transform together. At least one
column must fit in working memory, even when it exceeds the target; input and
output arrays and FFT-library workspace are additional. This is not a hard
process/device memory limit. Disk-backed TACAW output stays disk-backed on cache
reload. Spatial batching does **not** shorten the time window, alter frequency
bins, change gain/loss folding, or combine incoherent copies as amplitudes.

For network filesystems, test memmap/cache performance on a small run and prefer
node-local scratch when possible.
