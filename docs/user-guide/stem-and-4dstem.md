# STEM and 4D-STEM

## Choose the retained result

| Goal | Probe grid | Retain diffraction? | Path |
|---|---|---:|---|
| One CBED pattern | one position | yes | ordinary `run()` |
| One BF/ADF/HAADF image | scan | no | `ADF=...`, `return_layers=None` |
| Several virtual detectors | scan | yes | store one 4D-STEM cube |
| COM/DPC | scan | yes | store 4D-STEM |

Safe annular-image default:

```python
calc.setup(
    trajectory,
    aperture=30,
    voltage_eV=100e3,
    sampling=0.05,
    probe_xs=probe_xs,
    probe_ys=probe_ys,
    ADF=(60, 200),
    return_layers=None,
    cache_wavefunctions=False,
    loop_probes=16,
)
_, image = calc.run()
```

On-the-fly integration discards the diffraction patterns. Store a cube only
when later detector changes or diffraction-based analysis are required.

## The three grids

- `sampling`: propagation pixels; controls reciprocal bandwidth.
- `probe_xs`, `probe_ys`: scan coordinates; their differences are image-pixel spacing.
- specimen lateral dimensions: reciprocal-grid spacing, approximately `1/L`.

For a periodic scan use `endpoint=False` unless both physical edges are distinct.

## Detector planning

For small angles, `q_outer ≈ theta_outer_rad / wavelength_A`. A planning
estimate for a complete untapered circle is:

```text
q_full ≈ min((2/3 - 0.02)/(2*sampling_A), max_kx, max_ky)
theta_full_mrad ≈ 1000 * wavelength_A * q_full
```

`HAADFData.max_detector_mrad` is authoritative for the realized result. An
asymmetric reciprocal crop is limited by its smaller axis.

`HAADFData` is a historical name for disk/annular integration. BF, ABF, ADF,
and HAADF boundaries are experiment-specific; match the detector geometry and
verify its bandwidth rather than relying on a universal label.

## 4D-STEM axes

`WFData` is `(probe, frame/configuration, kx, ky, layer)`. For a complete scan,
x varies fastest in the flattened probe list. A single-frame exit-wave cube can
be rebuilt as:

```python
intensity = abs(wf.data[:, 0, :, :, -1]) ** 2
cube = intensity.reshape(
    len(probe_ys), len(probe_xs), len(wf.kxs), len(wf.kys)
).swapaxes(0, 1)
# (scan_x, scan_y, kx, ky)
```
