# Example notebooks

PySlice ships runnable technique scripts rather than notebooks. Each is a
complete, executable end-to-end workflow in `examples/`, and they are the
closest thing to a notebook the repository currently has.

| Script | Workflow |
|---|---|
| `examples/tem_diffraction.py` | Parallel-beam diffraction from a loaded structure |
| `examples/haadf_stem.py` | Convergent probe, probe grid, annular detector → STEM image |
| `examples/lacbed.py` | Large-angle convergent-beam electron diffraction |
| `examples/aberrations.py` | Applying abTEM-convention Cnm aberrations to the probe |
| `examples/loading_trajectories.py` | The loader paths: CIF, XYZ, LAMMPS dump, ASE trajectory |
| `examples/molecular_dynamics.py` | ML-potential MD producing a thermal trajectory |
| `examples/tacaw_from_trajectory.py` | Trajectory → multislice → TACAW spectra |
| `examples/tacaw_spectrum_image.py` | Real-space phonon mapping over a probe grid |
| `examples/tacaw_pipeline.py` | The full MD → multislice → TACAW chain |
| `examples/k_space_tmdc_showcase_pub.py` | Publication-style k-space TMDC figure |
| `examples/real_space_phonon_showcase_pub.py` | Publication-style real-space phonon figure |

Run one directly:

```bash
python examples/tem_diffraction.py
```

**Environment:** all of these need the core install. `molecular_dynamics.py`,
`tacaw_*.py`, and the showcase scripts need `[md]` (and realistically a GPU
plus downloaded ML-potential weights); `loading_trajectories.py` needs OVITO
for its non-CIF paths.

**Justified omission:** there is no executable notebook section yet, so this
page indexes the scripts instead of duplicating them. Converting the TACAW
chain into a notebook with recorded outputs is the obvious first candidate.
