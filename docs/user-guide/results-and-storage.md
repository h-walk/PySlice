# Results and storage

## Result contracts

| Object | Backend-native property | NumPy property | Shape |
|---|---|---|---|
| `WFData` | `.array` | `.data` | `(probe, frame, kx, ky, layer)` |
| `TACAWData` | `.intensity` | `.array` / `.data` | `(probe, frequency, kx, ky)` |
| `HAADFData` | `.adf` | `.array` / `.data` | `(probe_x, probe_y)` |

Coordinates are NumPy arrays. Converting accelerator data to NumPy may copy and
transfer it to CPU. `WFData.array` is complex amplitude; intensity is
`abs(wf.array)**2`.

## Cache, checkpoint, and export

- A cache accelerates an identical computation and may be deleted.
- A checkpoint resumes complete computational state. PySlice caches are not
  exact MD or workflow checkpoints.
- An export is a portable scientific result with axes and provenance.

Multislice caches live below `psi_data/<backend>_<fingerprint>` by default.
`save_path` selects another cache root. The fingerprint includes full atomic
positions, cell, elements, probe, layer, grid, crop, and algorithm settings.
`run(force_rerun=True)` bypasses frame reuse.

TACAW writes `tacaw.npy`, frequency coordinates, and a manifest recording the
source fingerprint, selected layer, representation, axes, and time-window
configuration. Incompatible settings recompute rather than reuse by shape.

Example portable export:

```python
import numpy as np
from pyslice import to_numpy

np.savez(
    "wf_result.npz",
    wavefunction=to_numpy(wf.array),
    time=wf.time,
    kxs=wf.kxs,
    kys=wf.kys,
    layers=wf.layer,
    probe_positions=np.asarray(wf.probe_positions),
)
```

Add commit, environment, input hashes, backend/device, parameters, and
normalization separately for an archival result.

## Mutation

Plotting and TACAW extraction methods are read-only. These methods mutate data:

- `WFData.counts`, `recenter`, `pad_real_space`, `propagate_*`,
  `addSpatialDecoherence`, `applyMask`, `crop`, and `aberrate`.
- `TACAWData.fold_gain_loss`, `apply_bose_correction`, and `fft_from_wf_data`.
- `HAADFData.calculateADF` replaces the current detector result.

Copy an object or array before applying destructive post-processing.
