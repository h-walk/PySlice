"""
multislice.py — Probe generation and multislice wave propagation.

All array operations go through an injected Backend instance.
No module-level backend state is used anywhere in this file.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from tqdm import tqdm

from pyslice.backend import Backend, make_backend, to_numpy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
m_electron = 9.109383e-31    # kg
q_electron = 1.602177e-19    # J/eV  (= kg m² s⁻² eV⁻¹)
c_light    = 299792458.0     # m/s
h_planck   = 6.62607015e-34  # m² kg s⁻¹


# ---------------------------------------------------------------------------
# Physical helper functions
# ---------------------------------------------------------------------------

def m_effective(eV: float) -> float:
    """
    Relativistic effective electron mass at kinetic energy eV.

    From E = mc², the extra inertia from the kinetic energy is eV/c²:
        m_eff = m_e + eV·q / c²
    Units: kg.
    """
    return m_electron + eV * q_electron / c_light ** 2


def wavelength(eV, backend: Optional[Backend] = None):
    """
    Relativistic de Broglie wavelength in Angstroms.

    The relativistic momentum of an electron accelerated through voltage V is:
        p = sqrt(E_k² + 2 E_k m_e c²) / c
    where E_k = eV·q is the kinetic energy in Joules.

    From λ = h/p:
        λ = h·c / sqrt(E_k² + 2 E_k m_e c²)

    Multiplied by 1e10 to convert metres → Angstroms.

    The computation is done in backend arrays so that scalar eV, a 1-D array
    of eVs (temporal decoherence), or a tensor all work identically and stay
    on the correct device throughout.
    """
    return_numpy = backend is None
    backend = make_backend() if backend is None else backend
    eV_J  = backend.asarray(eV,                    dtype=backend.float_dtype) * q_electron
    m_c2  = backend.asarray(m_electron * c_light**2, dtype=backend.float_dtype)
    h_c   = backend.asarray(h_planck   * c_light,    dtype=backend.float_dtype)
    momentum = backend.sqrt(eV_J**2 + 2 * eV_J * m_c2)
    result = h_c / momentum * 1e10   # Å
    if return_numpy:
        result = to_numpy(result)
        return float(result) if np.ndim(result) == 0 else result
    return result


# ---------------------------------------------------------------------------
# Anti-aliasing aperture
# ---------------------------------------------------------------------------

def antialias_aperture(kxs, kys, backend: Backend,
                       cutoff_fraction: float = 2/3,
                       taper_width: float = 0.02):
    """
    2/3-Nyquist anti-aliasing aperture with a smooth cosine taper.

    The standard multislice anti-aliasing criterion (Kirkland 2010 §6.8)
    sets the bandwidth limit at 2/3 of the Nyquist frequency so that
    aliasing artefacts from the transmission function multiplication
    (which doubles the bandwidth) stay below the Nyquist limit.

    A hard cutoff produces Gibbs ringing; the cosine taper suppresses it:
        aperture(k) = 1                                     if k < k_cut - taper
        aperture(k) = 0.5·(1 + cos(π·(k - k_cut + t)/t))  if k_cut-t < k < k_cut
        aperture(k) = 0                                     if k ≥ k_cut

    Returns a 2-D real array with the same device/dtype as kxs.
    """
    kx_max   = float(backend.amax(backend.absolute(kxs)))
    ky_max   = float(backend.amax(backend.absolute(kys)))
    k_max    = min(kx_max, ky_max)           # Nyquist limit
    k_cutoff = cutoff_fraction * k_max
    taper    = taper_width * k_max

    kx_grid, ky_grid = backend.meshgrid(kxs, kys, indexing='ij')
    k_r = backend.sqrt(kx_grid**2 + ky_grid**2)

    aperture = backend.ones_like(k_r)
    mask_taper = (k_r > k_cutoff - taper) & (k_r < k_cutoff)
    mask_outer = k_r >= k_cutoff
    aperture[mask_taper] = 0.5 * (
        1 + backend.cos(backend.pi * (k_r[mask_taper] - k_cutoff + taper) / taper)
    )
    aperture[mask_outer] = 0.0
    return aperture


# ---------------------------------------------------------------------------
# Aberration function
# ---------------------------------------------------------------------------

def aberrationFunction(kxs, kys, wavelength_A, aberrations: dict,
                       backend: Backend):
    """
    Phase aberration function χ(kx, ky) in the Cnm convention.

    Following Kirkland (2010) Eq. 2.10 and the abTEM notation
    (https://abtem.readthedocs.io/en/latest):

        χ(k, φ) = (2π/λ) · Σ_{n,m}  1/(n+1) · C_{nm} · (k·λ)^(n+1) · cos(m·(φ - φ₀))

    where
        k   = |k|  is the radial spatial frequency (Å⁻¹)
        φ   = arctan(ky/kx)  is the azimuthal angle
        n,m = integer indices parsed from the key string (e.g. 'C30' → n=3, m=0)
        C   = aberration coefficient in Å  (or (C, φ₀) for off-axis terms)

    The returned array is exp(-i·χ), i.e. the complex transfer function
    to be multiplied onto the wavefunction in reciprocal space.

    Note: kxs, kys are unshifted (reciprocal origin at corner, as produced
    by fftfreq), which is consistent with the probe array convention used
    throughout this module.
    """
    dPhi  = backend.zeros_like(kxs[:, None] * kys[None, :])
    ks    = backend.sqrt(kxs[:, None]**2 + kys[None, :]**2)
    theta = backend.arctan2(kys[None, :], kxs[:, None])

    for key, val in aberrations.items():
        n, m  = int(key[1]), int(key[2])   # e.g. 'C30' → n=3, m=0
        C, phi0 = (val, 0.0) if isinstance(val, (int, float)) else val
        dPhi += (
            2 * np.pi / wavelength_A
            * (1 / (n + 1))
            * C
            * (ks * wavelength_A) ** (n + 1)
            * backend.cos(m * (theta - phi0))
        )
    return backend.exp(-1j * dPhi)


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

class Probe:
    """
    Electron probe wavefunction for multislice calculations.

    Generates the probe in reciprocal space as a top-hat (or Gaussian-VOA)
    aperture function, then transforms to real space.  Multiple probe
    positions are handled by applying a phase ramp in reciprocal space
    (equivalent to a real-space translation).

    The probe array has shape (nc, npt, nx, ny) where
        nc  = number of coherent copies (e.g. temporal decoherence channels)
        npt = number of probe positions
        nx, ny = real-space grid size
    """

    def __init__(
            self,
            xs, ys,
            mrad: float,
            eV: float,
            backend: Optional[Backend] = None,
            array=None,
            device: Optional[str] = None,
            gaussianVOA: float = 0,
            preview: bool = False,
            probe_xs=None,
            probe_ys=None,
            probe_positions=None,
            cropping: int = False,
            defer_shifts: bool = False,
            stay_reciprocal: bool = False,
            crop_reciprocal=False,
    ):
        self._backend = make_backend(device) if backend is None else backend
        backend = self._backend

        # ------------------------------------------------------------------
        # Spatial grid
        # ------------------------------------------------------------------
        self.xs = backend.asarray(xs, dtype=backend.float_dtype)
        self.ys = backend.asarray(ys, dtype=backend.float_dtype)

        nx = len(xs); ny = len(ys)
        dx = float(xs[1] - xs[0]); dy = float(ys[1] - ys[0])
        lx = nx * dx;  ly = ny * dy
        self.nx = nx; self.ny = ny
        self.dx = dx; self.dy = dy
        self.lx = lx; self.ly = ly

        # ------------------------------------------------------------------
        # Probe positions
        # ------------------------------------------------------------------
        self.probe_xs = probe_xs
        self.probe_ys = probe_ys
        self.probe_positions = probe_positions

        if probe_xs is not None and probe_ys is not None:
            x, y = np.meshgrid(probe_xs, probe_ys)
            self.probe_positions = np.reshape([x, y], (2, len(x.flat))).T

        if self.probe_positions is None:
            self.probe_positions = [(lx / 2, ly / 2)]
            self.probe_xs = [lx / 2]
            self.probe_ys = [ly / 2]

        # ------------------------------------------------------------------
        # Beam parameters
        # ------------------------------------------------------------------
        self.mrad       = mrad
        self.eV         = eV
        self.wavelength = wavelength(eV, backend)   # scalar, Å
        self.eVs        = backend.asarray([eV], dtype=backend.float_dtype)
        self.wavelengths = wavelength(self.eVs, backend)
        self.temporal_decoherence = None
        self.spatial_decoherence  = None
        self.gaussianVOA          = gaussianVOA

        self.stay_reciprocal = stay_reciprocal
        self.crop_reciprocal = crop_reciprocal

        # ------------------------------------------------------------------
        # Reciprocal-space frequency arrays (unshifted: 0,1,…,-2,-1)
        # ------------------------------------------------------------------
        self.kxs = backend.fftfreq(nx, d=dx)
        self.kys = backend.fftfreq(ny, d=dy)

        # ------------------------------------------------------------------
        # Build or adopt the probe array
        # ------------------------------------------------------------------
        if array is not None:
            # Caller supplies a pre-built array (e.g. create_batched_probes).
            # Ensure it lives on the correct device and has complex dtype.
            self._array = backend.asarray(array, dtype=backend.complex_dtype)
        else:
            # Generate the single template probe, then broadcast to shape
            # (1, 1, nx, ny) so decoherence expansion and applyShifts work
            # uniformly on (nc, npt, nx, ny) shaped arrays throughout.
            single = self.generate_single_probe(
                mrad, self.wavelength, gaussianVOA, preview=preview)
            self._array = (
                single[None, None, :, :]
                * backend.ones((1, 1), dtype=backend.complex_dtype)[:, :, None, None]
            )

        self.cropping = cropping
        # Pixel offsets used when probe is cropped to a sub-region.
        self.offsets = np.zeros((len(self.probe_positions), 2), dtype=int)

        if not defer_shifts:
            self.applyShifts()

    # ------------------------------------------------------------------
    # Probe generation
    # ------------------------------------------------------------------

    def generate_single_probe(self, mrad: float, wavelength_val,
                              gaussianVOA: float, preview: bool = False):
        """
        Build one probe wavefunction centred at the grid origin.

        In reciprocal space the probe is a circle of radius r = mrad·1e-3 / λ
        (converting convergence semi-angle from mrad to Å⁻¹).  The circle is
        filled with amplitude 1; everything outside is 0.  Transforming to
        real space via IFFT gives the diffraction-limited probe shape.

        The kx/ky arrays are unshifted (origin at corner), so the resulting
        real-space probe is centred at the corner after IFFT.  ifftshift then
        moves the centre to (lx/2, ly/2), i.e. the middle of the simulation
        cell — where applyShifts expects it.

        If stay_reciprocal=True the probe is kept in reciprocal space and a
        phase ramp is applied to centre it (equivalent but saves two FFTs
        when shifting will be done in reciprocal space later).
        """
        b = self._backend
        kxs, kys = self.kxs, self.kys
        if self.crop_reciprocal:
            # Trim high-k frequencies from the unshifted arrays.
            # midcrop(a, n) removes the n outermost frequencies from each end
            # of an unshifted array: 0,1,2,…,-2,-1 → 0,1,…,-(n-1) (cropped).
            kxs = b.midcrop(self.kxs, self.crop_reciprocal[0])
            kys = b.midcrop(self.kys, self.crop_reciprocal[1])

        nx, ny = len(kxs), len(kys)
        if mrad == 0:
            # Plane wave: uniform amplitude, no aperture.
            return b.zeros((nx, ny), dtype=b.complex_dtype) + 1

        reciprocal = b.zeros((nx, ny), dtype=b.complex_dtype)
        # Convergence semi-angle in Å⁻¹: α_rad = mrad·1e-3, r = α_rad / λ
        radius = (mrad * 1e-3) / wavelength_val

        kx_grid, ky_grid = b.meshgrid(kxs, kys, indexing='ij')
        radii = b.sqrt(kx_grid**2 + ky_grid**2)

        if gaussianVOA == 0:
            # Hard aperture: mask all k-vectors inside the convergence radius.
            reciprocal[radii < radius] = 1.0
        else:
            # Gaussian virtual objective aperture — smooth roll-off via erf.
            from scipy.special import erf
            reciprocal = 1 - erf((radii - radius) / (gaussianVOA * radius))

        if preview:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            extent = (float(b.amin(self.kxs)), float(b.amax(self.kxs)),
                      float(b.amin(self.kys)), float(b.amax(self.kys)))
            ax.imshow(to_numpy(b.fftshift(reciprocal)).T,
                      cmap="inferno", extent=extent)
            ax.set_xlabel("kx (Å⁻¹)"); ax.set_ylabel("ky (Å⁻¹)")
            plt.show()
            plt.close(fig)

        if self.stay_reciprocal:
            # Centre in reciprocal space by multiplying with a phase ramp
            # equivalent to a real-space shift of lx/2, ly/2.
            # exp(-2πi kx · lx/2) shifts the probe to the cell centre.
            return (
                reciprocal
                * b.exp(-2j * b.pi * kxs[:, None] * self.lx / 2)
                * b.exp(-2j * b.pi * kys[None, :] * self.ly / 2)
            )

        # IFFT → real space; ifftshift moves centre from corner to middle.
        return b.ifftshift(b.ifft2(reciprocal))

    # ------------------------------------------------------------------
    # Position shifting
    # ------------------------------------------------------------------

    def applyShifts(self):
        """
        Expand the (nc,1,nx,ny) template array to (nc,npt,nx,ny) by shifting
        the probe to each requested position.

        The shift is applied in reciprocal space as a phase ramp:
            ψ_shifted(k) = ψ(k) · exp(-2πi (kx·Δx + ky·Δy))
        which is exact (no interpolation) and GPU-friendly.
        """
        b = self._backend
        nc, npt, nx, ny = self._array.shape
        if npt > 1:
            # Shifts have already been applied — nothing to do.
            return

        # Broadcast the single template probe to all npt positions.
        if self.cropping:
            # Crop the probe to a sub-window centred at lx/2, ly/2.
            i1 = nx // 2 - self.cropping // 2; i2 = i1 + self.cropping
            j1 = ny // 2 - self.cropping // 2; j2 = j1 + self.cropping
            self._array = (
                self._array[:, 0, None, i1:i2, j1:j2]
                * b.ones(len(self.probe_positions))[:, None, None][None, :]
            )
        else:
            self._array = (
                self._array[:, 0, None, :, :]
                * b.ones(len(self.probe_positions))[None, :, None, None]
            )

        for i, (px, py) in enumerate(self.probe_positions):
            if px - self.lx / 2 == 0 and py - self.ly / 2 == 0:
                continue   # already centred
            self._array[:, i, :, :], (dpx, dpy) = self.placeProbe(
                self._array[:, i, :, :], px, py)
            self.offsets[i, 0] = int(dpx)
            self.offsets[i, 1] = int(dpy)

    def placeProbe(self, array, x: float, y: float):
        """
        Shift the probe to real-space position (x, y) via a k-space phase ramp.

        The probe template starts centred at (lx/2, ly/2).  A displacement
        (Δx, Δy) = (x - lx/2, y - ly/2) is applied by multiplying the
        Fourier-space probe by exp(-2πi (kx·Δx + ky·Δy)).

        For cropped probes the large pixel-aligned part of the shift is handled
        by tracking an integer offset (dpx, dpy) and only the sub-pixel
        remainder is applied as a phase ramp.  This avoids wrapping artefacts.

        Returns (shifted_probe_in_original_domain, (offset_x, offset_y)).
        """
        b = self._backend
        dx = x - self.lx / 2
        dy = y - self.ly / 2

        if self.cropping:
            i1 = self.nx // 2 - self.cropping // 2
            j1 = self.ny // 2 - self.cropping // 2
            kxs = b.fftfreq(self.cropping, d=self.dx)
            kys = b.fftfreq(self.cropping, d=self.dy)
            # Split into pixel-aligned and sub-pixel shifts.
            dpx = dx // self.dx; dpy = dy // self.dy
            offset_x = i1 + dpx; offset_y = j1 + dpy
            dx -= dpx * self.dx; dy -= dpy * self.dy
        elif self.crop_reciprocal:
            kxs = b.midcrop(self.kxs, self.crop_reciprocal[0])
            kys = b.midcrop(self.kys, self.crop_reciprocal[1])
            offset_x = offset_y = 0
        else:
            kxs, kys = self.kxs, self.kys
            offset_x = offset_y = 0

        if not self.stay_reciprocal:
            probe_k = b.fft2(array)
        else:
            probe_k = array

        # Phase ramp:  multiply by exp(-2πi kx Δx) · exp(-2πi ky Δy).
        # Broadcasting: kxs has shape (nx,), so kxs[None,:,None] matches
        # the (nc, nx, ny) array layout.
        kx_shift = b.exp(-2j * b.pi * kxs[None, :, None] * dx)
        ky_shift = b.exp(-2j * b.pi * kys[None, None, :] * dy)
        probe_k_shifted = probe_k * kx_shift * ky_shift

        if self.stay_reciprocal:
            return probe_k_shifted, (offset_x, offset_y)
        return b.ifft2(probe_k_shifted), (offset_x, offset_y)

    # ------------------------------------------------------------------
    # Decoherence
    # ------------------------------------------------------------------

    def defocus(self, dz):
        """
        Apply a defocus of dz Angstroms to the probe.

        In reciprocal space, a free-space propagation of distance dz is
        multiplication by the Fresnel propagator:
            P(k, dz) = exp(-iπ λ dz |k|²)
        (Kirkland 2010, Eq. 6.65 with a positive dz shifting the waist
        below the entrance surface, i.e. positive defocus focuses deeper).

        If dz is scalar, the same defocus is applied to all nc copies.
        If dz is a 1-D array of length N, the probe array is expanded to
        (N·nc, npt, nx, ny) — one set of copies per defocus value.
        """
        b = self._backend
        if isinstance(dz, (int, float)):
            dz = b.zeros(len(self._array)) + dz
        kx_grid, ky_grid = b.meshgrid(self.kxs, self.kys, indexing='ij')
        k_sq = kx_grid**2 + ky_grid**2
        # P shape: (nz, nx, ny);  _array shape after FFT: (nc, npt, nx, ny)
        P = b.exp(-1j * b.pi * self.wavelength * dz[:, None, None] * k_sq[None, :, :])
        nz = len(dz); nc, npt, nx, ny = self._array.shape
        # Apply P to each of the nc coherent copies independently.
        self._array = b.ifft2(
            P[:, None, None, :, :] * b.fft2(self._array)[None, :, :, :, :]
        )
        self._array = b.reshape(self._array, (nz * nc, npt, nx, ny))

    def addTemporalDecoherence(self, sigma_eV: float, N: int):
        """
        Simulate temporal (chromatic) decoherence by incoherently averaging
        over a Gaussian energy spread of width sigma_eV.

        N probe copies are generated, each at a slightly different accelerating
        voltage sampled from the range [eV - 2σ, eV + 2σ].  Each copy is
        weighted by a Gaussian amplitude:
            A_n = exp(-(eV - eV_n)² / σ²)

        The resulting _array has shape (N, 1, nx, ny).  If spatial decoherence
        was already configured, it is re-applied so the order of operations
        is always: temporal → spatial → shift.
        """
        b = self._backend
        nc, npt, nx, ny = self._array.shape
        if self.temporal_decoherence is not None:
            logger.warning("addTemporalDecoherence called twice — overwriting previous.")
        self.temporal_decoherence = (sigma_eV, N)

        self.eVs        = b.asarray(b.linspace(self.eV - 2*sigma_eV,
                                                self.eV + 2*sigma_eV, N),
                                    dtype=b.float_dtype)
        self.wavelengths = wavelength(self.eVs, b)
        amplitudes       = b.exp(-(self.eV - self.eVs)**2 / sigma_eV**2)

        self._array = b.zeros((N, 1, nx, ny), dtype=b.complex_dtype)
        for n, eV_n in enumerate(self.eVs):
            lam_n = wavelength(eV_n, b)
            self._array[n, 0, :, :] = (
                amplitudes[n] * self.generate_single_probe(self.mrad, lam_n, self.gaussianVOA)
            )

        if self.spatial_decoherence is not None:
            self.addSpatialDecoherence(*self.spatial_decoherence)
        if self._array.shape[1] == 1:
            self.applyShifts()

    def addSpatialDecoherence(self, sigma_dz: float, N: int):
        """
        Simulate spatial (focal) decoherence by incoherently averaging over
        a Gaussian distribution of defocus values of width sigma_dz (Å).

        N defocus values are sampled from [-2σ, +2σ] with Gaussian weights.
        defocus() is called to expand the probe array from (nc, 1, nx, ny)
        to (N·nc, 1, nx, ny), and then each slice is amplitude-weighted.

        The eVs and wavelengths arrays are also tiled so every element of
        the expanded nc dimension has the correct associated value.
        """
        b = self._backend
        if self.spatial_decoherence is not None:
            logger.warning("addSpatialDecoherence called twice — overwriting previous.")
        self.spatial_decoherence = (sigma_dz, N)

        dzs        = b.asarray(b.linspace(-2*sigma_dz, 2*sigma_dz, N),
                               dtype=b.float_dtype)
        amplitudes = b.exp(-dzs**2 / sigma_dz**2)

        nc, npt, nx, ny = self._array.shape
        self.defocus(dzs)   # expands: (nc,1,nx,ny) → (N·nc,1,nx,ny)

        # Apply Gaussian weights — the defocus loop is the outer (slower)
        # index: result[i·nc + j] = defocus[i] × probe[j]
        for i in range(N):
            for j in range(nc):
                self._array[i * nc + j] *= amplitudes[i]

        nc = self._array.shape[0]
        # Tile eVs/wavelengths to match the expanded nc dimension.
        self.eVs         = b.reshape(
            b.ones(N)[:, None] * self.eVs[None, :], (nc,))
        self.wavelengths = b.reshape(
            b.ones(N)[:, None] * self.wavelengths[None, :], (nc,))

        if self._array.shape[1] == 1:
            self.applyShifts()

    # ------------------------------------------------------------------
    # Aberrations
    # ------------------------------------------------------------------

    def aberrate(self, aberrations: dict):
        """
        Apply wavefront aberrations defined in the Cnm convention.

        Aberrations are phase errors at the aperture plane, so they must be
        applied in reciprocal space.  The real-space probe was built via:
            probe_real = ifftshift(ifft2(aperture))
        Reversing this: fft2(fftshift(probe_real)) gives back the aperture,
        to which we multiply the aberration phase factor, then transform back.
        """
        b = self._backend
        dP = aberrationFunction(self.kxs, self.kys, self.wavelength, aberrations, b)
        reciprocal = b.fft2(b.fftshift(self._array))
        reciprocal *= dP
        self._array = b.ifftshift(b.ifft2(reciprocal))

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self, selected_probes=None) -> Probe:
        """
        Deep copy, optionally selecting a subset of probe positions.

        selected_probes is an index array into the npt dimension.  It is
        kept as a numpy array (CPU) since it is used only for indexing, not
        computation.
        """
        b = self._backend
        new_probe = Probe.__new__(Probe)
        new_probe._backend = b
        # Copy all non-array scalar attributes.
        for attr, val in self.__dict__.items():
            if attr.startswith('_') or 'array' in attr:
                continue
            setattr(new_probe, attr, b.clone(val))

        if selected_probes is not None:
            sel = to_numpy(selected_probes)
            nc, npt, nx, ny = self._array.shape
            new_probe._array = b.clone(
                self._array[:, :, :, :] if npt == 1 else self._array[:, sel, :, :]
            )
            new_probe.offsets        = self.offsets[sel, :]
            new_probe.probe_positions = np.asarray(self.probe_positions)[sel, :]
        else:
            new_probe._array = b.clone(self._array)

        return new_probe

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @property
    def array(self) -> np.ndarray:
        """CPU NumPy copy of the probe array."""
        return to_numpy(self._array)

    def plot(self, filename=None, title=None):
        import matplotlib.pyplot as plt
        b = self._backend
        # Mean over nc copies, select position 0, flip y for imshow convention.
        arr = np.flip(to_numpy(b.mean(b.absolute(self._array), axis=0)), axis=1)[0].T
        plot_arr = np.abs(arr) ** 0.25

        xs_np = to_numpy(self.xs)
        ys_np = to_numpy(self.ys)
        extent = (xs_np.min(), xs_np.max(), ys_np.min(), ys_np.max())

        fig, ax = plt.subplots()
        ax.imshow(plot_arr, cmap="inferno", extent=extent)
        ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
        if title:
            ax.set_title(title)
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# PrismProbe
# ---------------------------------------------------------------------------

class PrismProbe:
    """
    PRISM algorithm probe representation.

    Instead of propagating one real-space probe per position, PRISM propagates
    a sparse set of plane-wave Fourier components (sinusoids) that are shared
    by all probe positions.  The exit wave for any real-space probe is then
    reconstructed as a linear combination of these component exit waves
    (calculateProbesFromS).

    probe_positions here stores (kx, ky) reciprocal-space coordinates of the
    sinusoids, NOT real-space (x, y) positions as in Probe.

    Reference: Ophus, C. (2017). A fast image simulation algorithm for
    scanning transmission electron microscopy. Advanced Structural and
    Chemical Imaging, 3(1), 13.
    """

    def __init__(
            self,
            xs, ys,
            mrad: float,
            eV: float,
            backend: Optional[Backend] = None,
            device: Optional[str] = None,
            gaussianVOA: float = 0,
            preview: bool = False,
            nkx: int = 25,
            nky: Optional[int] = None,
            kth: int = 1,
    ):
        self._backend = make_backend(device) if backend is None else backend
        b = self._backend

        self.dx = float(xs[1] - xs[0]); self.dy = float(ys[1] - ys[0])
        self.nx = len(xs);               self.ny = len(ys)
        self.xs = b.asarray(xs, dtype=b.float_dtype)
        self.ys = b.asarray(ys, dtype=b.float_dtype)

        # Shifted frequency arrays (origin at centre) for k-space indexing.
        self.kxs = b.fftshift(b.fftfreq(self.nx, d=self.dx))
        self.kys = b.fftshift(b.fftfreq(self.ny, d=self.dy))

        # Template array — will be expanded by applyShifts.
        self._array = b.zeros((1, 1, self.nx, self.ny), dtype=b.complex_dtype)

        # ------------------------------------------------------------------
        # Sparse k-grid for PRISM
        # ------------------------------------------------------------------
        if nky is None:
            nky = nkx
        self.nx_cropped = nkx; self.ny_cropped = nky

        # i1, j1: indices into the full shifted kxs/kys that mark the start
        # of the central (nkx × nky) sub-grid of Fourier components.
        self.i1 = self.nx // 2 - self.nx_cropped // 2
        self.j1 = self.ny // 2 - self.ny_cropped // 2
        # Re-compute the actual cropped sizes (may differ by 1 due to integer division).
        self.nx_cropped = self.nx - 2 * self.i1
        self.ny_cropped = self.ny - 2 * self.j1

        # Store the (kx, ky) pair for every Fourier component in the sparse grid.
        pos = b.zeros((self.nx_cropped, self.ny_cropped, 2))
        for i, kx in enumerate(self.kxs[self.i1:self.nx - self.i1]):
            for j, ky in enumerate(self.kys[self.j1:self.ny - self.j1]):
                pos[i, j, 0] = kx
                pos[i, j, 1] = ky
        self.probe_positions = b.reshape(pos, (self.nx_cropped * self.ny_cropped, 2))

        # ------------------------------------------------------------------
        # Beam parameters (mirrored from Probe for API compatibility)
        # ------------------------------------------------------------------
        self.mrad        = mrad
        self.eV          = eV
        self.wavelength  = wavelength(eV, b)
        self.eVs         = b.asarray([eV], dtype=b.float_dtype)
        self.wavelengths = wavelength(self.eVs, b)
        self.temporal_decoherence = None
        self.spatial_decoherence  = None
        self.gaussianVOA = gaussianVOA
        self.cropping    = False
        self.kth         = kth

    def applyShifts(self):
        """
        Populate _array with the complex sinusoids for each Fourier component.

        The nth sinusoid for (kx_n, ky_n) is:
            ψ_n(x, y) = exp(2πi (kx_n·x + ky_n·y))

        This is the PRISM "interpolation factor" basis: propagating each of
        these through the sample yields the S-matrix column for that k-vector.
        """
        b = self._backend
        self._array = (
            self._array[:, 0, None, :, :]
            * b.ones(len(self.probe_positions))[None, :, None, None]
        )
        for n, (kx, ky) in enumerate(self.probe_positions):
            self._array[:, n, :, :] = (
                b.exp(2j * b.pi * self.xs[:, None] * kx)
                * b.exp(2j * b.pi * self.ys[None, :] * ky)
            )

    def calculateProbesFromS(self, array, positions, chunksize: int = 100,
                             load_into=None, ADF=False):
        """
        Reconstruct real-space probe exit waves from the S-matrix columns.

        After propagating the sinusoidal entrance waves through the sample,
        each exit wave ψ_n(k) is a column of the scattering matrix S.  The
        exit wave for a probe at real-space position (x, y) is:

            Ψ_exit(k; x,y) = Σ_n  P_n(k; x,y) · S_n(k)

        where P_n(k; x,y) is the Fourier component of the probe function
        at k-vector n, evaluated for a probe centred at (x, y).  This is
        equivalent to selecting and phase-ramping the relevant rows of the
        FFT'd probe aperture — exactly what the "factors" array contains.

        The reconstruction is done in chunks of size `chunksize` to avoid
        blowing up RAM with a full (n_positions × nkx × nky) intermediate.

        array shape on input: (n_sinusoids, nkx, nky, n_layers, 1)
            reshaped to:      (nx_cropped, ny_cropped, nkx, nky)
        """
        b = self._backend

        if load_into is None and not ADF:
            result = b.zeros(
                (len(positions), b.ceil(self.nx / self.kth), b.ceil(self.ny / self.kth)),
                dtype=b.complex_dtype,
            )
        elif not ADF:
            result = load_into
        else:
            ADF, ADFmask, ADFindex = ADF
            result = None

        npt, nkx, nky, _, _ = array.shape
        # Reshape from (sinusoid_index, kx, ky, layer, 1)
        # to (nx_cropped, ny_cropped, kx, ky) for einsum below.
        array = b.reshape(array, (self.nx_cropped, self.ny_cropped, nkx, nky))

        chunksize = max(1, chunksize)
        for n, (x, y) in enumerate(tqdm(positions)):
            if n % chunksize != 0:
                continue

            # Build a mini-Probe for this chunk of positions so we can reuse
            # the phase-ramp logic in placeProbe / generate_single_probe.
            # stay_reciprocal=True keeps the result in reciprocal space,
            # avoiding two unnecessary FFTs.
            probes = Probe(
                to_numpy(self.xs), to_numpy(self.ys),
                self.mrad, self.eV, backend=b,
                probe_positions=positions[n:n + chunksize],
                stay_reciprocal=True,
            )
            # fftshift moves the reciprocal-space probe so k=0 is at centre,
            # matching the shifted kxs/kys convention used for the sparse grid.
            probe_ks = b.fftshift(probes._array[0, :, :, :], axes=(-2, -1))

            # Crop to the sparse k-grid (i1:-i1, j1:-j1 in shifted indexing).
            # These factors give the weighting of each sinusoid for this probe.
            factors = probe_ks[:, self.i1:self.nx - self.i1,
                                   self.j1:self.ny - self.j1]

            # Reconstruct the exit wave:  Σ_{kx_n, ky_n} factors · S_n(kx, ky)
            # Indices: p=probe chunk, k=sparse kx, q=sparse ky, x=full kx, y=full ky
            chunked = b.einsum('pkq,kqxy->pxy', factors, array)

            if isinstance(result, np.memmap):
                chunked = to_numpy(chunked)

            if ADF:
                intensities = b.einsum(
                    'pxy,xy->p', b.absolute(chunked)**2, ADFmask)
                for intensity, pp in zip(intensities, range(n, n + chunksize)):
                    ADF._array[ADFindex == pp] += intensity
            else:
                result[n:n + chunksize, :, :] = chunked

        return result

    def copy(self, selected_probes=None) -> PrismProbe:
        b = self._backend
        new_probe = PrismProbe.__new__(PrismProbe)
        new_probe._backend = b
        for attr, val in self.__dict__.items():
            if attr.startswith('_') or 'array' in attr:
                continue
            setattr(new_probe, attr, b.clone(val))

        if selected_probes is not None:
            nc, npt, nx, ny = self._array.shape
            new_probe._array = b.clone(
                self._array[:, :, :, :] if npt == 1 else self._array[:, selected_probes, :, :]
            )
            new_probe.probe_positions = self.probe_positions[selected_probes, :]
        else:
            new_probe._array = b.clone(self._array)

        return new_probe

    @property
    def array(self) -> np.ndarray:
        return to_numpy(self._array)


# ---------------------------------------------------------------------------
# Batched probe creation helper
# ---------------------------------------------------------------------------

def create_batched_probes(base_probe: Probe, probe_positions,
                          backend: Optional[Backend] = None,
                          device: Optional[str] = None) -> Probe:
    """
    Build a Probe with one shifted copy per position in probe_positions.

    This is a convenience wrapper for cases where the caller wants a fully
    materialised (n_probes, nx, ny) probe cube without going through
    applyShifts.  Each copy is shifted via a k-space phase ramp as in
    placeProbe, avoiding any interpolation.
    """
    b = backend or (make_backend(device) if device is not None else base_probe._backend)
    nx, ny = len(base_probe.xs), len(base_probe.ys)
    dx = float(to_numpy(base_probe.xs[1] - base_probe.xs[0]))
    dy = float(to_numpy(base_probe.ys[1] - base_probe.ys[0]))
    lx = nx * dx; ly = ny * dy

    probe_arrays = []
    template = base_probe._array[:, 0, :, :]
    for px, py in probe_positions:
        probe_k = b.fft2(template, axes=(-2, -1))
        kx_shift = b.exp(-2j * b.pi * base_probe.kxs[:, None]  * (px - lx / 2))
        ky_shift = b.exp(-2j * b.pi * base_probe.kys[None, :]  * (py - ly / 2))
        probe_arrays.append(b.ifft2(probe_k * kx_shift * ky_shift))

    return Probe(
        to_numpy(base_probe.xs), to_numpy(base_probe.ys),
        base_probe.mrad, base_probe.eV, backend=b,
        array=b.stack(probe_arrays, axis=1),
        probe_positions=probe_positions,
    )


# ---------------------------------------------------------------------------
# Multislice propagator
# ---------------------------------------------------------------------------

def Propagate(
        probe,
        potential,
        backend: Optional[Backend] = None,
        device: Optional[str] = None,
        progress: bool = False,
        onthefly: bool = True,
        store_all_slices: bool = False,
        stored_slice_indices=None,
):
    """
    Multislice wave propagation (Kirkland 2010, §6.5).

    The algorithm alternates between two operations per slice z:

    1. Transmission:
           ψ'(r, z) = t(r, z) · ψ(r, z)
       where the transmission function t = exp(iσ V(r, z)) encodes the
       projected potential V of slice z, and σ is the interaction parameter:
           σ = (2π / λ E₀) · (E₀ + eV) / (2E₀ + eV)
       (Kirkland Eq. 5.6, with E₀ = m_e c² the electron rest energy).

    2. Fresnel propagation to the next slice:
           ψ(r, z+dz) = ℱ⁻¹[ P(k, dz) · ℱ[ψ'(r, z)] ]
       where the propagator in k-space is:
           P(k, dz) = exp(-iπ λ dz |k|²)
       (Kirkland Eq. 6.65).  The 2/3-Nyquist anti-aliasing aperture is
       folded into P to bandwidth-limit the wavefunction at every slice
       and suppress aliasing from the transmission step.

    The probe array shape is (nc, npt, nx, ny); it is flattened to
    (nc·npt, nx, ny) for vectorised propagation then returned as-is.

    Args:
        probe:            Probe or PrismProbe object.
        potential:        Potential object (must expose .zs, .kxs, .kys,
                          ._array or ._calculate_slice).
        backend:          Active Backend instance.
        device:           Optional device used to construct a backend when one
                          cannot be inferred from ``probe`` or ``potential``.
        progress:         Show a tqdm progress bar over slices.
        onthefly:         If True, compute potential slices one at a time
                          (lower peak RAM).  If False, build the full 3-D
                          potential first.
        store_all_slices: If True, return wavefunction at every slice
                          (shape: n_slices, nc·npt, nx, ny) instead of
                          only the final exit wave.
        stored_slice_indices: Optional 0-based slice indices to return when
                          ``store_all_slices`` is True. ``None`` returns all
                          slices.

    Returns:
        Array of shape (nc·npt, nx, ny) or (n_slices, nc·npt, nx, ny).
    """
    b = backend or getattr(probe, "_backend", None) or getattr(potential, "_backend", None)
    if b is None:
        b = make_backend(device)

    nc, npt, nx, ny = probe._array.shape
    # Flatten coherent copies and probe positions into a single batch index.
    array = b.reshape(probe._array, (nc * npt, nx, ny))

    # Expand wavelength and eV arrays to match the flattened batch dimension.
    # probe.wavelengths has shape (nc,); each wavelength applies to all npt positions.
    probe_wavelengths = b.reshape(
        probe.wavelengths[:, None] * b.ones(npt)[None, :], (nc * npt,))
    probe_eVs = b.reshape(
        probe.eVs[:, None] * b.ones(npt)[None, :], (nc * npt,))

    # ------------------------------------------------------------------
    # Interaction parameter σ  (Kirkland Eq. 5.6)
    # ------------------------------------------------------------------
    # σ = (2π / λ·eV) · (E₀ + eV) / (2E₀ + eV)
    # where E₀ = m_e·c² / q  is the rest energy in eV.
    # σ has units of 1/(V·Å) so that σ·V(Å) is dimensionless.
    E0_eV = m_electron * c_light**2 / q_electron   # rest energy, eV
    sigma = (
        (2 * b.pi) / (probe_wavelengths * probe_eVs)
        * (E0_eV + probe_eVs) / (2 * E0_eV + probe_eVs)
    )

    # ------------------------------------------------------------------
    # Slice thickness
    # ------------------------------------------------------------------
    dz = float(to_numpy(potential.zs[1] - potential.zs[0])) if len(potential.zs) > 1 else 0.5

    # ------------------------------------------------------------------
    # k-space grids and Fresnel propagator
    # ------------------------------------------------------------------
    kx, ky = potential.kxs, potential.kys
    if probe.cropping:
        # Use a smaller k-grid matching the cropped probe size.
        kx = b.fftfreq(probe.cropping, d=probe.dx)
        ky = b.fftfreq(probe.cropping, d=probe.dy)

    kx_grid, ky_grid = b.meshgrid(kx, ky, indexing='ij')
    k_sq = kx_grid**2 + ky_grid**2

    # Anti-aliasing aperture: zeros out the outer 1/3 of k-space.
    aa = antialias_aperture(kx, ky, b)

    # Fresnel propagator folded with anti-aliasing aperture.
    # Shape: (nc·npt, nx, ny) — one P per wavelength in the batch.
    # The aa aperture is broadcast over the batch dimension.
    P = b.exp(-1j * b.pi * probe_wavelengths[:, None, None] * dz * k_sq[None, :, :]) * aa[None, :, :]

    if not onthefly:
        potential.build()

    if store_all_slices and stored_slice_indices is not None:
        stored_slice_indices = set(int(i) for i in stored_slice_indices)
        invalid_slices = [
            i for i in stored_slice_indices
            if i < 0 or i >= len(potential.zs)
        ]
        if invalid_slices:
            raise ValueError(
                f"stored_slice_indices contains out-of-range slices "
                f"{sorted(invalid_slices)}; valid range is "
                f"[0, {len(potential.zs) - 1}]"
            )

    slice_wavefunctions = [] if store_all_slices else None
    iterator = range(len(potential.zs))
    if progress:
        print("Propagating through slices")
        iterator = tqdm(iterator)

    for z in iterator:
        # ------------------------------------------------------------------
        # Transmission function:  t(r) = exp(i σ V(r, z))
        # ------------------------------------------------------------------
        if onthefly:
            potential_slice = potential._calculate_slice(z)
        else:
            potential_slice = potential.array[:, :, z]

        if probe.cropping:
            # Build index arrays to extract the cropped sub-window for each
            # probe position without allocating a full (npt, nx, ny) array.
            nx_full, ny_full = potential_slice.shape
            xr = b.arange(nx_full); yr = b.arange(ny_full)
            xi = b.zeros((len(sigma), probe.cropping), dtype=int)
            yi = b.zeros((len(sigma), probe.cropping), dtype=int)
            for p, (ox, oy) in enumerate(probe.offsets):
                xi[p, :] = b.roll(xr, -ox)[:probe.cropping]
                yi[p, :] = b.roll(yr, -oy)[:probe.cropping]
            # Advanced indexing: pot_stack[p, i, j] = potential_slice[xi[p,i], yi[p,j]]
            pot_stack = potential_slice[xi[:, :, None], yi[:, None, :]]
            t = b.exp(1j * sigma[:, None, None] * pot_stack)
        else:
            # Broadcast: sigma shape (nc·npt,), potential_slice shape (nx, ny).
            t = b.exp(1j * sigma[:, None, None] * potential_slice[None, :, :])

        array = t * array

        if store_all_slices and (
            stored_slice_indices is None or z in stored_slice_indices
        ):
            slice_wavefunctions.append(b.clone(array))

        # ------------------------------------------------------------------
        # Fresnel propagation:  ψ(z+dz) = ℱ⁻¹[ P · ℱ[ψ'(z)] ]
        # (skipped for the last slice — we want the exit wave, not another step)
        # ------------------------------------------------------------------
        if z < len(potential.zs) - 1:
            array = b.ifft2(P * b.fft2(array, axes=(-2, -1)), axes=(-2, -1))

    if store_all_slices:
        # Shape: (n_slices, nc·npt, nx, ny)
        return b.stack(slice_wavefunctions, axis=0)

    return array


# ---------------------------------------------------------------------------
# Inverse multislice: recover object from entrance and exit waves
# ---------------------------------------------------------------------------

def calculateObject(probe, exitwave, guessedObject, backend: Optional[Backend] = None,
                    weighting: float = 0.5, dz: float = 0.5,
                    damping: float = 0.01):
    """
    Recover the projected potential slice from a known entrance and exit wave.

    From the multislice equations (Kirkland 2010):
        ψ_exit = ℱ⁻¹[ P · ℱ[ t · ψ_entrance ] ]

    Inverting step by step:
        ℱ[ψ_exit] / P = ℱ[ t · ψ_entrance ]
        t · ψ_entrance = ℱ⁻¹[ ℱ[ψ_exit] / P ]
        t = ℱ⁻¹[ ℱ[ψ_exit] / P ] / ψ_entrance

    Since t = exp(iσ O) the object O is:
        O = angle(t) / σ    (taking the phase of t, the most stable route)

    A probe-amplitude damping mask is applied so that pixels where the probe
    has no intensity (|ψ_entrance| ≈ 0) do not contribute unreliable estimates:
        weight(r) = |ψ_entrance(r)| / (|ψ_entrance(r)| + δ · max|ψ_entrance|)

    The returned delta is a weighted update to be added to the current estimate
    of the object (iterative refinement loop in the caller).
    """
    b = backend or getattr(probe, "_backend", None) or make_backend()
    psi1  = probe._array[0, 0, :, :]
    lamda = probe.wavelengths[0]
    eV    = probe.eVs[0]

    E0_eV = m_electron * c_light**2 / q_electron
    sigma = (
        (2 * np.pi) / (lamda * eV)
        * (E0_eV + eV) / (2 * E0_eV + eV)
    )

    kx_grid, ky_grid = b.meshgrid(probe.kxs, probe.kys, indexing='ij')
    k_sq = kx_grid**2 + ky_grid**2
    # Propagator P — dividing by P in Fourier space back-propagates one slice.
    P = b.exp(-1j * b.pi * lamda * dz * k_sq)

    # Back-propagate exit wave through one slice and deconvolve entrance wave.
    t = b.ifft2(b.fft2(exitwave) / P) / psi1

    # Extract object via phase: O = angle(t) / σ
    O = b.angle(t) / sigma

    # Probe-amplitude weighting suppresses contributions from low-intensity regions.
    psi1_abs = b.absolute(psi1)
    weight = psi1_abs / (psi1_abs + damping * b.amax(psi1_abs))

    delta = (O - b.asarray(guessedObject)) * weight
    return delta * weighting
