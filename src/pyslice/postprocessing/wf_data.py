"""
Wave function data structure.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from ..multislice.multislice import Probe, aberrationFunction
from ..data.pyslice_serial import PySliceSerial, Signal, Dimensions, Dimension, Metadata
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
            self.dimensions = Dimensions([
                Dimension(name='probe',  space='position',
                          values=np.arange(len(probe_positions))),
                Dimension(name='time',   space='temporal',   units='ps',
                          values=to_numpy(time)),
                Dimension(name='kx',     space='scattering', units='Å⁻¹',
                          values=to_numpy(kxs)),
                Dimension(name='ky',     space='scattering', units='Å⁻¹',
                          values=to_numpy(kys)),
                Dimension(name='layer',  space='position',
                          values=layer_arr),
            ], nav_dimensions=[0, 1], sig_dimensions=[2, 3, 4])

            pp_array = np.array(probe_positions).flatten().tolist()
            self.metadata = Metadata({
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

    def recenter(self):
        b = self._backend
        self._xs -= b.mean(self._xs)
        self._ys -= b.mean(self._ys)
    def pad_real_space(self,add_x=0,add_y=0):
        b = self._backend
        dx = self._xs[1]-self._xs[0] ; dy = self._ys[1]-self._ys[0]
        pix_x = int(round(add_x/dx)) ; pix_y = int(round(add_y/dy))
        npt, nt, nx, ny, nl = self._array.shape
        array = b.ifft2(self._array,axes=(-3,-2)) # npt, nt, nx, ny, nl indices. iFFT x,y
        new = b.zeros((npt, nt, nx+pix_x*2, ny+pix_y*2, nl), type_match = self._array)
        i1=pix_x ; i2=pix_x+nx ; j1=pix_y ; j2=pix_y+ny
        new[:,:,i1:i2,j1:j2,:] += array
        self._array = b.fft2(new,axes=(-3,-2))
        # xs=[0,1,2,3,4], dx=1, add 3. new should be -3,-2,-1,0,1,2,3,4,5,6,7. from x[0]-N*dx, to x[-1]+N*dx, and length len(xs)+2*N
        self._xs = b.linspace( self._xs[0]-dx*pix_x , self._xs[-1]+dx*pix_x , nx+pix_x*2 )
        self._ys = b.linspace( self._ys[0]-dy*pix_y , self._ys[-1]+dy*pix_y , ny+pix_y*2 )
        self._kxs = b.fftshift(b.fftfreq(nx+pix_x*2, dx))  # k-space in 1/Å
        self._kys = b.fftshift(b.fftfreq(ny+pix_y*2, dy))  # k-space in 1/Å


    def propagate_through_lens(self,f):
        b = self._backend
        array = b.ifft2(self._array[:, :, :, :, -1])
        xs = b.asarray(self._xs)#-self.probe_positions[-1][0]
        ys = b.asarray(self._ys)#-self.probe_positions[-1][1]
        x_grid, y_grid = b.meshgrid(xs,ys, indexing='ij')
        k = 2*b.pi / self.probe.wavelength
        L = b.exp(-1j * k / 2 / f * ( x_grid ** 2 + y_grid ** 2 ) )
        array = L[None,None,:,:] * array
        self._array[:,:,:,:,-1] = b.fft2(array)

    def propagate_free_space(self, dz: float):
        b = self._backend
        kx_grid, ky_grid = b.meshgrid(self._kxs, self._kys, indexing='ij')
        P = b.exp(-1j * b.pi * self.probe.wavelength * dz * (kx_grid ** 2 + ky_grid ** 2))
        self._array = P[None, None, :, :, None] * self._array

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

    def applyMask(self, radius: float, realOrReciprocal: str = "reciprocal"):
        b = self._backend
        if realOrReciprocal == "reciprocal":
            radii = b.sqrt(self._kxs[:, None] ** 2 + self._kys[None, :] ** 2)
            mask = b.zeros(radii.shape)
            mask[radii < radius] = 1
            self._array *= mask[None, None, :, :, None]
        else:
            radii = b.sqrt(
                (self._xs[:, None] - b.mean(b.asarray(self._xs))) ** 2 +
                (self._ys[None, :] - b.mean(b.asarray(self._ys))) ** 2
            )
            mask = b.zeros(radii.shape)
            mask[radii < radius] = 1
            real = b.ifft2(b.ifftshift(self._array, axes=(2, 3)), axes=(2, 3))
            real *= mask[None, None, :, :, None]
            self._array = b.fftshift(b.fft2(real, axes=(2, 3)), axes=(2, 3))

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

    def aberrate(self, aberrations: dict):
        dP = aberrationFunction(
            self._kxs, self._kys, self.probe.wavelength, aberrations, self._backend
        )
        self._array[:, :, :, :, :] *= dP[None, None, :, :, None]
