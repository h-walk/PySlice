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

For network filesystems, test memmap/cache performance on a small run and prefer
node-local scratch when possible.
