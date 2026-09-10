---
name: tacaw-phonons
description: Use when simulating vibrational EELS, phonon spectroscopy, or phonon dispersions with PySlice's TACAW method — planning MD length and frame spacing for a frequency window, computing TACAWData from wavefunctions, extracting spectra, spectrum images, spectral diffraction, and dispersion curves, or applying Bose corrections.
---

# TACAW Phonons

TACAW (Time Autocorrelation of Auxiliary Wavefunctions) turns a multislice
run over an MD trajectory into vibrational-EELS observables: FFT the exit
wavefunctions along time to get |Ψ(ω,q)|², then slice by frequency, k-path,
or probe position.

## Read First

- `skills/pyslice/SKILL.md` — units and workflow
- `skills/md-setup/SKILL.md` — producing the MD trajectory TACAW consumes
- `skills/simulation-parameter-selection/SKILL.md` rule 8 — the frequency
  window ↔ frame spacing/count math (single source of truth)
- `examples/tacaw_from_trajectory.py`, `examples/tacaw_spectrum_image.py`

## Core Rules

- **The time axis is the spectrometer.** Frame spacing Δt sets
  f_max = 1/(2Δt); total time N·Δt sets Δf = 1/(N·Δt). Plan these BEFORE
  running MD — they cannot be fixed afterwards.
- **Trajectory timestep is picoseconds** (`Trajectory.timestep`) and TACAW
  frequencies come out in **THz** (1 THz ≈ 4.136 meV).
- **Production MD should be NVE** (no thermostat noise in the dynamics);
  equilibrate in NVT first (`production_ensemble='nve'` in `md-setup`).
- **TACAW needs a multi-frame WFData** — a single frame has no time axis.
  The FFT subtracts the time-mean (elastic line) automatically.
- **Bose correction** balances energy gain/loss:
  `TACAWData(wf, temperature_K=300, apply_bose=True)` or
  `tacaw.apply_bose_correction(T)` afterwards (intensity data only).
- **Signal choices:** `spectrum()` integrates all k; `masked_spectrum()`
  mimics a collection aperture; `spectral_diffraction(f)` is the k-map at one
  frequency; `spectrum_image(f)` maps intensity over probe positions;
  `dispersion(kx_path, ky_path)` gives frequency-vs-k along a path.
- **k-resolution comes from the supercell** (Δk = 1/L): dispersion work needs
  enough unit cells along the path direction. Aperture choice: parallel beam
  (0 mrad) for clean q-resolved diffraction; convergent probes for
  atomic-resolution spectrum images.
- **Memory:** TACAW keeps (probe, freq, kx, ky); use `chunkFFT=True` /
  `chunk_size_time` for large runs, `max_kx`/`max_ky` crops at setup, and
  `keep_complex=False` (default) unless phases are needed.

## Natural Language Task Routing

- "phonon spectrum / vibrational EELS of X" → MD (md-setup) → multislice over
  the trajectory → `TACAWData(wf)` → `spectrum()`.
- "resolve up to N THz with resolution R" → rule 8: Δt ≤ 1/(2N) ps between
  saved frames, frames ≥ 1/(R·Δt); state both numbers.
- "phonon dispersion / band structure along Γ–K/M" → build the k-path in 1/Å
  from the reciprocal lattice → `dispersion(kx_path, ky_path)`.
- "map a phonon mode in real space" → `spectrum_image(frequency)` over a
  probe grid.
- "diffuse scattering at frequency f" → `spectral_diffraction(f)`.
- "temperature effects / detailed balance" → Bose correction with the MD
  temperature.
- "energy in meV" → multiply THz by 4.136.

## Python API Examples For Agents

```python
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.postprocessing.tacaw_data import TACAWData
import numpy as np

# traj: MD trajectory, e.g. 200 frames saved every 16.7 fs (f_max 30 THz, df 0.3 THz)
calc = MultisliceCalculator()
calc.setup(traj, aperture=0.0, voltage_eV=60e3, sampling=0.1, slice_thickness=0.5)
wf = calc.run()

tacaw = TACAWData(wf, temperature_K=300, apply_bose=True)
freqs = tacaw.frequencies                    # THz, fftshifted (negative = gain side)
spectrum = tacaw.spectrum()                  # k-integrated

# dispersion along kx from Gamma to 1.5 1/A
kxs = np.linspace(0, 1.5, 60); kys = np.zeros_like(kxs)
disp = tacaw.dispersion(kxs, kys)            # (n_freq, n_k)

tacaw.to_sea("tacaw.sea")
```

## MCP Tool Use

`pyslice_suggest_parameters` (goal `tacaw`) plans the MD frame spacing/count;
`pyslice_run_md` produces the trajectory; `pyslice_setup_multislice` +
`pyslice_run_multislice` propagate it; then `pyslice_compute_tacaw`,
`pyslice_tacaw_spectrum` (dominant peaks in THz), `pyslice_spectrum_image`,
`pyslice_dispersion`, and `pyslice_export_sea`.
