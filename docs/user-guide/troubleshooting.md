# Troubleshooting

| Condition | Meaning | Action |
|---|---|---|
| detector outer-angle warning | result uses an incomplete/tapered annulus | reduce angle or decrease sampling/crop less |
| detector inner-angle `ValueError` | no complete detector annulus exists | change simulation bandwidth |
| cache reuse log | informational reuse | use `run(force_rerun=True)` to recompute |
| TACAW recomputes | cache manifest differs | expected after source/layer/window changes |
| PRISM plus setup defocus | unsupported | use ordinary multislice or zero defocus |
| NPT production request | PySlice stores one fixed cell | use NPT only for ASE equilibration, then NVT/NVE production |
| MD equilibration `RuntimeError` | production was not started | inspect log and revise protocol |
| GPU/device OOM | batch or propagation grid too large | reduce `loop_probes`; then grid/problem size |
| disk growth | result/caches are large | estimate size; disable unnecessary caches/layers |

To diagnose backend selection:

```python
from pyslice import make_backend
b = make_backend()
print(type(b).__name__, b.device, b.float_dtype, b.complex_dtype)
```

## Reporting a bug

Open an issue at <https://github.com/h-walk/PySlice/issues> with:

```text
PySlice version or Git commit:
Python, OS, and install command/extras:
Backend, device, and dtypes:
Minimal script and full traceback:
Input shape and saved-frame timestep:
Voltage, aperture, sampling, slices, detector/crop:
Cache reuse? Does force_rerun=True change it?
Expected versus observed result:
```

Do not attach proprietary or enormous trajectories without first reducing the
failure to the smallest shareable input.
