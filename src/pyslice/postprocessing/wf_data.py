"""
Wave function data structure.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from ..multislice.multislice import Probe, aberrationFunction
from ..data.pyslice_serial import (
    PySliceSerial,
    Signal,
    Dimensions,
    Dimension,
    Metadata,
    record_pyslice_operation,
    track_pyslice_action,
)
from pyslice.backend import Backend, to_numpy


class WFData(PySliceSerial, Signal):
    """
    Wavefunction data with format: (probe_positions, frame, kx, ky, layer).

    All GPU/CPU array operations are performed via the injected Backend instance.
    Coordinate arrays are stored in their native backend type internally; the
    public properties (kxs, kys, xs, ys, time, layer) always return NumPy arrays
    for downstream compatibility.
    """

    _sea_config = {
        'tensor_attrs': ['_kxs', '_kys', '_xs', '_ys', '_time', '_layer', '_array'],
        'path_attrs': ['cache_dir'],
        'tuple_list_attrs': ['probe_positions'],
        'exclude_attrs': ['probe', '_backend'],
        'force_datasets': ['_array', 'probe_positions', '_kxs', '_kys',
                           '_xs', '_ys', '_time', '_layer'],
    }

    def __init__(
        self,
        probe_positions: List[Tuple[float, float]],
        probe_xs: List[float],
        probe_ys: List[float],
        time: np.ndarray,
        kxs,
        kys,
        xs,
        ys,
        layer,
        array,
        probe: Probe,
        backend: Backend,
        cache_dir: Optional[Path] = None,
    ):
        self._backend = backend

        self.probe_positions = probe_positions
        self.probe_xs = probe_xs
        self.probe_ys = probe_ys
        self._time  = time
        self._kxs   = kxs
        self._kys   = kys
        self._xs    = xs
        self._ys    = ys
        self._layer = layer
        self.probe  = probe
        self.cache_dir = cache_dir
        self.probability = None
        self._array = array

        # Build Signal dimensions
        if Dimensions is not None:
            layer_arr = to_numpy(layer) if layer is not None else np.array([0])
            layer_dimension = (
                Dimension(
                    name='layer',
                    space='position',
                    size=0,
                    scale=1,
                    offset=0,
                )
                if layer_arr.size == 0
                else Dimension(
                    name='layer',
                    space='position',
                    values=layer_arr,
                )
            )
            dimensions = Dimensions([
                Dimension(name='probe',  space='position',
                          values=np.arange(len(probe_positions))),
                Dimension(name='time',   space='temporal',   units='ps',
                          values=to_numpy(time)),
                Dimension(name='kx',     space='scattering', units='Å⁻¹',
                          values=to_numpy(kxs)),
                Dimension(name='ky',     space='scattering', units='Å⁻¹',
                          values=to_numpy(kys)),
                layer_dimension,
            ], nav_dimensions=[0, 1], det_dimensions=[2, 3, 4])

            pp_array = np.array(probe_positions).flatten().tolist()
            metadata = Metadata({
                'General': {
                    'title': 'Multislice Wavefunction',
                    'signal_type': 'Wavefunction',
                },
                'Simulation': {
                    'voltage_eV':    float(probe.eV),
                    'wavelength_A':  float(probe.wavelength),
                    'aperture_mrad': float(probe.mrad),
                    'probe_positions': pp_array,
                    'n_probes': len(probe_positions),
                },
            })
            Signal.__init__(
                self,
                data=array,
                name='Multislice Wavefunction',
                dimensions=dimensions,
                signal_type='Diffraction',
                metadata=metadata,
            )
            record_pyslice_operation(
                self,
                "WFData.__init__",
                parameters={
                    "array_shape": tuple(getattr(array, "shape", ())),
                    "n_probe_positions": len(probe_positions),
                    "n_time": len(time) if time is not None else None,
                    "n_layers": len(layer_arr),
                    "cache_dir": None if cache_dir is None else str(cache_dir),
                },
                callable_obj=type(self).__init__,
            )

    @classmethod
    def from_probe(
        cls,
        *,
        extent_A: float | Tuple[float, float],
        sampling: float | Tuple[float, float],
        voltage_eV: float,
        aperture: float = 0.0,
        defocus: float = 0.0,
        probe_positions: Optional[List[Tuple[float, float]]] = None,
        backend: Optional[Backend] = None,
        device: Optional[str] = None,
    ) -> "WFData":
        """Create a standalone electron probe without a specimen.

        The returned wave is centered on an optical-axis coordinate system and
        is immediately compatible with :class:`pyslice.optics.OpticalColumn`.
        ``sampling``, ``voltage_eV``, ``aperture``, ``defocus``, and
        ``probe_positions`` have the same meaning as in
        :meth:`MultisliceCalculator.setup`; only ``extent_A`` is additional,
        because there is no specimen cell from which to derive the field of
        view. Probe positions are relative to the optical axis rather than
        absolute specimen-cell coordinates. A zero aperture creates a plane
        wave. Requested extents are rounded to an even number of samples so
        the FFT grid has a unique central pixel.
        """
        extent_x, extent_y = (
            (float(extent_A), float(extent_A))
            if isinstance(extent_A, (int, float))
            else (float(extent_A[0]), float(extent_A[1]))
        )
        dx, dy = (
            (float(sampling), float(sampling))
            if isinstance(sampling, (int, float))
            else (float(sampling[0]), float(sampling[1]))
        )
        if extent_x <= 0 or extent_y <= 0:
            raise ValueError("extent_A values must be positive.")
        if dx <= 0 or dy <= 0:
            raise ValueError("sampling values must be positive.")
        if voltage_eV <= 0:
            raise ValueError("voltage_eV must be positive.")
        if probe_positions is None:
            probe_positions = [(0.0, 0.0)]
        relative_positions = [
            (float(position[0]), float(position[1]))
            for position in probe_positions
        ]
        if not relative_positions:
            raise ValueError("probe_positions must contain at least one position.")

        nx = max(2, int(round(extent_x / dx)))
        ny = max(2, int(round(extent_y / dy)))
        # Probe generation and shifted FFT coordinates share an unambiguous
        # central pixel on even grids. Round odd requests up by one sample
        # rather than introducing a half-pixel origin mismatch.
        nx += nx % 2
        ny += ny % 2
        xs = np.arange(nx, dtype=float) * dx
        ys = np.arange(ny, dtype=float) * dy
        optical_xs = (np.arange(nx, dtype=float) - nx // 2) * dx
        optical_ys = (np.arange(ny, dtype=float) - ny // 2) * dy
        # Match the discrete origin used by ``optical_xs``/``optical_ys``.
        # For an odd-sized grid, ``n * d / 2`` lies between pixels and shifts
        # the generated probe by one sample after the FFT shift.
        center_x = (nx // 2) * dx
        center_y = (ny // 2) * dy
        absolute_positions = [
            (center_x + x, center_y + y) for x, y in relative_positions
        ]
        probe = Probe(
            xs,
            ys,
            mrad=float(aperture),
            eV=float(voltage_eV),
            backend=backend,
            device=device,
            probe_positions=absolute_positions,
        )
        defocus_value = float(defocus)
        if defocus_value:
            probe.defocus(defocus_value)
        b = probe._backend
        reciprocal = b.fftshift(
            b.fft2(probe._array[0], axes=(-2, -1)), axes=(-2, -1)
        )
        wf = cls(
            probe_positions=relative_positions,
            probe_xs=sorted({position[0] for position in relative_positions}),
            probe_ys=sorted({position[1] for position in relative_positions}),
            time=np.array([0.0]),
            kxs=b.fftshift(probe.kxs),
            kys=b.fftshift(probe.kys),
            xs=b.asarray(optical_xs),
            ys=b.asarray(optical_ys),
            layer=np.array([0]),
            array=reciprocal[:, None, :, :, None],
            probe=probe,
            backend=b,
        )
        return wf

    # ------------------------------------------------------------------
    # Properties — public interface always returns numpy
    # ------------------------------------------------------------------

    @property
    def kxs(self)   -> np.ndarray: return to_numpy(self._kxs)
    @property
    def kys(self)   -> np.ndarray: return to_numpy(self._kys)
    @property
    def xs(self)    -> np.ndarray: return to_numpy(self._xs)
    @property
    def ys(self)    -> np.ndarray: return to_numpy(self._ys)
    @property
    def time(self)  -> np.ndarray: return to_numpy(self._time) if self._time is not None else None
    @property
    def layer(self) -> np.ndarray: return to_numpy(self._layer) if self._layer is not None else None

    @property
    def data(self):
        """Lazy conversion to NumPy for Signal compatibility."""
        return to_numpy(self._array) if self._array is not None else None

    @data.setter
    def data(self, value):
        self._array = value

    @property
    def array(self):
        """Raw array (may be a backend tensor)."""
        return self._array

    @array.setter
    def array(self, value):
        self._array = value

    # ------------------------------------------------------------------
    # Reshape helper
    # ------------------------------------------------------------------

    def reshaped(self):
        """
        Reshape _array from (nc*npt, nt, kx, ky, nl)
        to (nc, nx_probe, ny_probe, nt, kx, ky, nl).
        """
        b = self._backend
        nc, nptp, _, _ = self.probe._array.shape
        nptp = len(self.probe_positions)
        _, nt, nkx, nky, nl = self._array.shape
        intermediate = b.reshape(self._array, (nc, nptp, nt, nkx, nky, nl))
        nx, ny = len(self.probe_xs), len(self.probe_ys)
        reshaped = b.reshape(intermediate, (nc, ny, nx, nt, nkx, nky, nl))
        # swap probe_x / probe_y axes to get (nc, nx, ny, nt, kx, ky, nl)
        return reshaped.swapaxes(1, 2)

    # ------------------------------------------------------------------
    # Photon-counting simulation
    # ------------------------------------------------------------------

    @track_pyslice_action
    def counts(self, N: int):
        b = self._backend
        npt, nt, nx, ny, nl = self._array.shape
        if self.probability is None:
            self.probability = self._array
            ary = self._array / b.sum(b.absolute(self._array))
            ary = b.absolute(b.reshape(ary, (npt * nt * nx * ny * nl,)))
            self.buckets = b.zeros(len(ary) + 1, type_match=ary)
            self.buckets[1:] = b.cumsum(ary)
        detector_hits = b.asarray(b.randfloats(N))
        hist, _ = b.histogram(detector_hits, bins=self.buckets)
        self._array = b.asarray(hist.reshape((npt, nt, nx, ny, nl)))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_reciprocal(self,
                        filename=None,
                        whichProbe="mean",
                        whichTimestep="mean",
                        powerscaling=0.25,
                        extent=None,
                        nuke_zerobeam=False,
                        title=None):
        import matplotlib.pyplot as plt

        b = self._backend
        raw = self._array[:, :, :, :, -1]   # p,t,kx,ky
        npt, nt, nkx, nky = raw.shape
        accum = b.zeros((nkx, nky))

        probe_indices = np.arange(npt) if whichProbe == "mean" else (
            [whichProbe] if isinstance(whichProbe, int) else whichProbe)
        time_indices  = np.arange(nt)  if whichTimestep == "mean" else (
            [whichTimestep] if isinstance(whichTimestep, int) else whichTimestep)

        for p in probe_indices:
            for t in time_indices:
                layer = b.absolute(raw[p, t, :, :])
                if isinstance(raw, np.memmap):
                    layer = b.asarray(layer)
                accum += layer
        accum /= (len(time_indices) * len(probe_indices))

        kxs_np = to_numpy(self._kxs)
        kys_np = to_numpy(self._kys)

        if extent is not None:
            kx_min, kx_max, ky_min, ky_max = extent
            kx_mask = (kxs_np >= kx_min) & (kxs_np <= kx_max)
            ky_mask = (kys_np >= ky_min) & (kys_np <= ky_max)
            accum = accum[kx_mask, :][:, ky_mask]
            kxs_np = kxs_np[kx_mask]
            kys_np = kys_np[ky_mask]
            actual_extent = (kxs_np[0], kxs_np[-1], kys_np[0], kys_np[-1])
        else:
            actual_extent = (kxs_np.min(), kxs_np.max(), kys_np.min(), kys_np.max())

        accum_np = to_numpy(accum).T  # imshow: y,x
        if nuke_zerobeam:
            accum_np[np.argmin(np.abs(kys_np)), np.argmin(np.abs(kxs_np))] = 0

        img = (np.abs(accum_np) ** 2) ** powerscaling
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="inferno", extent=actual_extent, origin='lower', aspect=1)
        ax.set_xlabel("kx (Å⁻¹)")
        ax.set_ylabel("ky (Å⁻¹)")
        if title:
            ax.set_title(title)
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close(fig)

    plot = plot_reciprocal

    def plot_phase(self, filename=None, whichProbe=0, whichTimestep=0,
                   extent=None, avg=False):
        import matplotlib.pyplot as plt

        b = self._backend
        if avg:
            raw = b.mean(self._array[whichProbe, :, :, :, -1], axis=0)
        else:
            raw = self._array[whichProbe, whichTimestep, :, :, -1]

        real_space = b.ifft2(raw)
        xs_np = to_numpy(self._xs)
        ys_np = to_numpy(self._ys)

        if extent is not None:
            x_min, x_max, y_min, y_max = extent
            xm = (xs_np >= x_min) & (xs_np <= x_max)
            ym = (ys_np >= y_min) & (ys_np <= y_max)
            real_space = real_space[xm, :][:, ym]
            actual_extent = (xs_np[xm][0], xs_np[xm][-1], ys_np[ym][0], ys_np[ym][-1])
        else:
            actual_extent = (xs_np.min(), xs_np.max(), ys_np.min(), ys_np.max())

        phase = to_numpy(b.angle(real_space)).T
        fig, ax = plt.subplots()
        im = ax.imshow(phase, cmap='hsv', extent=actual_extent, origin='lower',
                       vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im, ax=ax, label='Phase (radians)')
        ax.set_title('Phase in real space')
        ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close(fig)

    def plot_realspace(self, whichProbe="mean", whichTimestep="mean",
                       extent=None, filename=None,powerscaling=0.25):
        import matplotlib.pyplot as plt

        b = self._backend
        array = b.absolute(b.ifft2(self._array[:, :, :, :, -1]))

        if whichProbe == "mean":
            array = b.mean(array, axis=0)
        else:
            array = array[whichProbe]
        if whichTimestep == "mean":
            array = b.mean(array, axis=0)
        else:
            array = array[whichTimestep]

        xs_np = to_numpy(self._xs)
        ys_np = to_numpy(self._ys)
        if extent is None:
            extent = (xs_np.min(), xs_np.max(), ys_np.min(), ys_np.max())

        img = to_numpy(b.absolute(array) ** powerscaling).T
        fig, ax = plt.subplots()
        ax.imshow(img, cmap="inferno", extent=extent)
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @track_pyslice_action
    def recenter(self):
        b = self._backend
        self._xs -= b.mean(self._xs)
        self._ys -= b.mean(self._ys)

    def sampling_report(
        self,
        *,
        real_edge_fraction: float = 0.1,
        reciprocal_edge_fraction: float = 0.2,
    ) -> dict:
        """Return real- and reciprocal-space sampling diagnostics.

        Edge fractions describe the outer portion of each axis included in the
        audit. For example, ``real_edge_fraction=0.1`` measures power in the
        outer 10 percent at either side of the real-space field. The report is
        backend-independent and contains ordinary Python/NumPy values.
        """
        if not 0.0 < real_edge_fraction < 0.5:
            raise ValueError("real_edge_fraction must be between 0 and 0.5.")
        if not 0.0 < reciprocal_edge_fraction < 0.5:
            raise ValueError(
                "reciprocal_edge_fraction must be between 0 and 0.5."
            )

        b = self._backend
        nx, ny = len(self._xs), len(self._ys)
        dx = float(to_numpy(self._xs[1] - self._xs[0]))
        dy = float(to_numpy(self._ys[1] - self._ys[0]))
        center_x = self._xs[nx // 2]
        center_y = self._ys[ny // 2]
        half_extent_x = nx * dx / 2.0
        half_extent_y = ny * dy / 2.0
        real_mask = (
            (
                b.absolute(self._xs - center_x)
                >= (1.0 - real_edge_fraction) * half_extent_x
            )[:, None]
            | (
                b.absolute(self._ys - center_y)
                >= (1.0 - real_edge_fraction) * half_extent_y
            )[None, :]
        )

        real = b.ifft2(
            b.ifftshift(self._array, axes=(2, 3)), axes=(2, 3)
        )
        real_power = b.absolute(real) ** 2
        real_totals = np.asarray(
            to_numpy(b.sum(real_power, axis=(2, 3))), dtype=float
        )
        real_edge_powers = np.asarray(
            to_numpy(
                b.sum(
                    real_power * real_mask[None, None, :, :, None],
                    axis=(2, 3),
                )
            ),
            dtype=float,
        )
        max_real_edge_fraction = float(
            np.max(real_edge_powers / np.maximum(real_totals, 1e-300))
        )

        kx_limit = float(np.max(np.abs(to_numpy(self._kxs))))
        ky_limit = float(np.max(np.abs(to_numpy(self._kys))))
        reciprocal_mask = (
            (
                b.absolute(self._kxs)
                >= (1.0 - reciprocal_edge_fraction) * kx_limit
            )[:, None]
            | (
                b.absolute(self._kys)
                >= (1.0 - reciprocal_edge_fraction) * ky_limit
            )[None, :]
        )
        reciprocal_power = b.absolute(self._array) ** 2
        reciprocal_totals = np.asarray(
            to_numpy(b.sum(reciprocal_power, axis=(2, 3))), dtype=float
        )
        reciprocal_edge_powers = np.asarray(
            to_numpy(
                b.sum(
                    reciprocal_power
                    * reciprocal_mask[None, None, :, :, None],
                    axis=(2, 3),
                )
            ),
            dtype=float,
        )
        max_reciprocal_edge_fraction = float(
            np.max(
                reciprocal_edge_powers
                / np.maximum(reciprocal_totals, 1e-300)
            )
        )

        wavelength_A = float(to_numpy(self.probe.wavelength))
        return {
            "shape": (nx, ny),
            "sampling_A": (dx, dy),
            "extent_A": (nx * dx, ny * dy),
            "real_edge_power_fraction": max_real_edge_fraction,
            "reciprocal_edge_power_fraction": max_reciprocal_edge_fraction,
            "theta_nyquist_mrad": (
                wavelength_A * kx_limit * 1e3,
                wavelength_A * ky_limit * 1e3,
            ),
            "physical_norm": float(np.sum(real_totals)) * dx * dy,
        }

    @track_pyslice_action
    def pad_real_space(self, add_x=0, add_y=0) -> dict:
        """Add zero-valued real-space margins while preserving grid spacing.

        ``add_x`` and ``add_y`` are the requested margins on *each* side of
        the existing field in Angstroms. They are rounded to whole pixels. The
        shifted reciprocal-space convention, optical-axis coordinate, and
        physical wave norm are preserved.
        """
        add_x = float(add_x)
        add_y = float(add_y)
        if add_x < 0 or add_y < 0:
            raise ValueError("Real-space padding must be non-negative.")

        b = self._backend
        dx = float(to_numpy(self._xs[1] - self._xs[0]))
        dy = float(to_numpy(self._ys[1] - self._ys[0]))
        pix_x = int(round(add_x / dx))
        pix_y = int(round(add_y / dy))
        npt, nt, nx, ny, nl = self._array.shape
        new_nx = nx + 2 * pix_x
        new_ny = ny + 2 * pix_y
        before = self.sampling_report()
        if new_nx == nx and new_ny == ny:
            return {
                "requested_padding_A": (add_x, add_y),
                "actual_padding_A": (0.0, 0.0),
                "old_shape": (nx, ny),
                "new_shape": (nx, ny),
                "before": before,
                "after": before,
            }

        real = b.ifft2(
            b.ifftshift(self._array, axes=(2, 3)), axes=(2, 3)
        )
        padded = b.zeros(
            (npt, nt, new_nx, new_ny, nl), type_match=self._array
        )
        padded[:, :, pix_x : pix_x + nx, pix_y : pix_y + ny, :] = real
        self._array = b.fftshift(
            b.fft2(padded, axes=(2, 3)), axes=(2, 3)
        )

        center_x = float(to_numpy(self._xs[nx // 2]))
        center_y = float(to_numpy(self._ys[ny // 2]))
        self._xs = (b.arange(new_nx) - new_nx // 2) * dx + center_x
        self._ys = (b.arange(new_ny) - new_ny // 2) * dy + center_y
        self._kxs = b.fftshift(b.fftfreq(new_nx, d=dx))
        self._kys = b.fftshift(b.fftfreq(new_ny, d=dy))

        if getattr(self, "dimensions", None) is not None:
            try:
                self.dimensions["kx"].values = to_numpy(self._kxs)
                self.dimensions["ky"].values = to_numpy(self._kys)
            except Exception:
                pass

        after = self.sampling_report()
        return {
            "requested_padding_A": (add_x, add_y),
            "actual_padding_A": (pix_x * dx, pix_y * dy),
            "old_shape": (nx, ny),
            "new_shape": (new_nx, new_ny),
            "before": before,
            "after": after,
        }


    @track_pyslice_action
    def propagate_through_lens(self,f):
        self.propagate_through_astigmatic_lens(f, f)

    @track_pyslice_action
    def propagate_through_astigmatic_lens(self, f_x, f_y):
        b = self._backend
        array = b.ifft2(self._array, axes=(2, 3))
        xs = b.asarray(self._xs)#-self.probe_positions[-1][0]
        ys = b.asarray(self._ys)#-self.probe_positions[-1][1]
        x_grid, y_grid = b.meshgrid(xs,ys, indexing='ij')
        k = 2*b.pi / self.probe.wavelength
        phase = b.zeros(x_grid.shape, type_match=x_grid)
        if f_x is not None and np.isfinite(float(f_x)):
            phase += x_grid ** 2 / float(f_x)
        if f_y is not None and np.isfinite(float(f_y)):
            phase += y_grid ** 2 / float(f_y)
        L = b.exp(-1j * k / 2 * phase)
        array = L[None, None, :, :, None] * array
        self._array = b.fft2(array, axes=(2, 3))

    @track_pyslice_action
    def propagate_free_space(self, dz: float):
        self.propagate_anisotropic_free_space(dz, dz)

    @track_pyslice_action
    def propagate_anisotropic_free_space(self, dz_x: float, dz_y: float):
        b = self._backend
        kx_grid, ky_grid = b.meshgrid(self._kxs, self._kys, indexing='ij')
        P = b.exp(
            -1j * b.pi * self.probe.wavelength
            * (float(dz_x) * kx_grid ** 2 + float(dz_y) * ky_grid ** 2)
        )
        self._array = P[None, None, :, :, None] * self._array

    @track_pyslice_action
    def apply_beam_tilt(self, theta_x: float = 0.0, theta_y: float = 0.0):
        """Apply a transverse angular kick to the wavefunction."""
        b = self._backend
        array = b.ifft2(self._array, axes=(2, 3))
        x_grid, y_grid = b.meshgrid(
            b.asarray(self._xs), b.asarray(self._ys), indexing='ij'
        )
        phase = 2j * b.pi / self.probe.wavelength * (
            float(theta_x) * x_grid + float(theta_y) * y_grid
        )
        self._array = b.fft2(
            b.exp(phase)[None, None, :, :, None] * array,
            axes=(2, 3),
        )

    @track_pyslice_action
    def rotate_real_space(self, angle_rad: float):
        """Rotate the transverse wave without interpolation using FFT shears."""
        angle = float(angle_rad)
        if angle == 0.0:
            return

        # Three-shear rotation becomes ill-conditioned near pi, so split large
        # rotations into pieces no larger than 45 degrees.
        n_steps = max(1, int(np.ceil(abs(angle) / (np.pi / 4.0))))
        step = angle / n_steps
        for _ in range(n_steps):
            self._rotate_real_space_step(step)

    def _rotate_real_space_step(self, angle_rad: float):
        """Apply one three-shear real-space rotation step."""
        b = self._backend
        array = b.ifft2(
            b.ifftshift(self._array, axes=(2, 3)),
            axes=(2, 3),
        )
        dx = float(to_numpy(self._xs[1] - self._xs[0]))
        dy = float(to_numpy(self._ys[1] - self._ys[0]))
        kx = b.fftfreq(len(self._xs), d=dx)
        ky = b.fftfreq(len(self._ys), d=dy)
        xs = b.asarray(self._xs)
        ys = b.asarray(self._ys)

        shear_x = np.tan(float(angle_rad) / 2.0)
        shear_y = -np.sin(float(angle_rad))

        phase_x = b.exp(2j * b.pi * shear_x * kx[:, None] * ys[None, :])
        array = b.ifft(
            b.fft(array, axes=2) * phase_x[None, None, :, :, None],
            axes=2,
        )

        phase_y = b.exp(2j * b.pi * shear_y * xs[:, None] * ky[None, :])
        array = b.ifft(
            b.fft(array, axes=3) * phase_y[None, None, :, :, None],
            axes=3,
        )

        array = b.ifft(
            b.fft(array, axes=2) * phase_x[None, None, :, :, None],
            axes=2,
        )
        self._array = b.fftshift(
            b.fft2(array, axes=(2, 3)), axes=(2, 3)
        )

    @track_pyslice_action
    def addSpatialDecoherence(self, sigma_dz: float, N: int):
        b = self._backend
        dzs = b.linspace(-2 * sigma_dz, 2 * sigma_dz, N)
        amplitudes = b.exp(-dzs ** 2 / sigma_dz ** 2)
        self._array = self._array[:, None, :, :, :, :] * b.ones(N)[None, :, None, None, None, None]
        nc, npt, nt, nx, ny, nl = self._array.shape
        kx_grid, ky_grid = b.meshgrid(self._kxs, self._kys, indexing='ij')
        k_sq = kx_grid ** 2 + ky_grid ** 2
        for i in range(N):
            P = b.exp(-1j * b.pi * self.probe.wavelength * dzs[i] * k_sq)
            self._array[:, i, :, :, :, :] *= amplitudes[i] * P[None, None, :, :, None]
        self._array = b.reshape(self._array, (nc * npt, nt, nx, ny, nl))

    @track_pyslice_action
    def applyMask(self, radius: float, realOrReciprocal: str = "reciprocal"):
        b = self._backend
        if realOrReciprocal == "reciprocal":
            radii = b.sqrt(self._kxs[:, None] ** 2 + self._kys[None, :] ** 2)
            mask = b.zeros(radii.shape)
            mask[radii < radius] = 1
            self._array *= mask[None, None, :, :, None]
        else:
            center_x = self._xs[len(self._xs) // 2]
            center_y = self._ys[len(self._ys) // 2]
            radii = b.sqrt(
                (self._xs[:, None] - center_x) ** 2 +
                (self._ys[None, :] - center_y) ** 2
            )
            mask = b.zeros(radii.shape)
            mask[radii < radius] = 1
            real = b.ifft2(b.ifftshift(self._array, axes=(2, 3)), axes=(2, 3))
            real *= mask[None, None, :, :, None]
            self._array = b.fftshift(
                b.fft2(real, axes=(2, 3)), axes=(2, 3)
            )

    @track_pyslice_action
    def crop(self, kx_range=None, ky_range=None):
        kxs_np = to_numpy(self._kxs)
        kys_np = to_numpy(self._kys)
        _, _, nx, ny, _ = self._array.shape
        i1, i2, j1, j2 = 0, nx, 0, ny
        if kx_range is not None:
            i1 = int(np.argwhere(kxs_np >= kx_range[0])[0])
            i2 = int(np.argwhere(kxs_np <= kx_range[1])[-1]) + 1
        if ky_range is not None:
            j1 = int(np.argwhere(kys_np >= ky_range[0])[0])
            j2 = int(np.argwhere(kys_np <= ky_range[1])[-1]) + 1
        self._array = self._array[:, :, i1:i2, j1:j2, :]
        self._kxs = self._kxs[i1:i2]
        self._kys = self._kys[j1:j2]

    @track_pyslice_action
    def aberrate(self, aberrations: dict):
        dP = aberrationFunction(
            self._kxs, self._kys, self.probe.wavelength, aberrations, self._backend
        )
        self._array[:, :, :, :, :] *= dP[None, None, :, :, None]
