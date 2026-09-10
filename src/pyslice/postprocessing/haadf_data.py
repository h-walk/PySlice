"""
HAADF (High Angle Annular Dark Field) data structure.
"""
import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import logging
from .wf_data import WFData
from ..data.pyslice_serial import PySliceSerial, Signal, Dimensions, Dimension, Metadata
from pyslice.backend import Backend, to_numpy
from ..multislice.multislice import (
    ANTIALIAS_CUTOFF_FRACTION,
    ANTIALIAS_TAPER_WIDTH,
)

logger = logging.getLogger(__name__)


class HAADFData(PySliceSerial, Signal):
    """
    Data structure for HAADF (High Angle Annular Dark Field) imaging data.

    Inherits from Signal for sea-eco compatibility.

    Attributes:
        probe_positions: Array of (x,y) probe positions in Angstroms.
        xs: x coordinates of the HAADF image.
        ys: y coordinates of the HAADF image.
        adf: The computed ADF image (x × y).
        probe: Probe object with beam parameters.
        cache_dir: Path to cache directory.
    """

    _sea_config = {
        'tensor_attrs': ['_kxs', '_kys', '_xs', '_ys', '_array', 'data'],
        'path_attrs': ['cache_dir'],
        'tuple_list_attrs': ['probe_positions'],
        'exclude_attrs': ['probe', '_wf_array', '_backend'],
        'force_datasets': ['_array', 'probe_positions', '_kxs', '_kys', '_xs', '_ys'],
    }

    def __init__(self, wf_data: WFData) -> None:
        """
        Initialize HAADFData from WFData.

        Args:
            wf_data: WFData object containing wavefunction data
        """
        # Copy needed attributes from WFData (raw tensors for GPU ops)
        self._backend = wf_data._backend
        self.probe_positions = wf_data.probe_positions
        self._kxs = wf_data._kxs
        self._kys = wf_data._kys
        self.probe = wf_data.probe
        self.cache_dir = wf_data.cache_dir

        # Store reference to source WFData array for ADF calculation
        self._wf_array = wf_data.reshaped() # nprobes,x,y,t,kx,ky,l indices

        # Initialize ADF as None, will be computed by calculateADF
        self._array = None
        self._xs = wf_data.probe_xs
        self._ys = wf_data.probe_ys

        if Dimensions is not None:
            # Build placeholder dimensions (will be updated after calculateADF)
            self.dimensions = Dimensions([
                Dimension(name='x', space='position', units='Å', values=np.array([0])),
                Dimension(name='y', space='position', units='Å', values=np.array([0])),
            ], nav_dimensions=[0, 1], sig_dimensions=[])

            # Build metadata
            metadata_dict = {
                'General': {
                    'title': 'HAADF Image',
                    'signal_type': 'HAADF'
                },
                'Simulation': {
                    'voltage_eV': float(self.probe.eV),
                    'wavelength_A': float(self.probe.wavelength),
                    'aperture_mrad': float(self.probe.mrad),
                    'probe_positions': [list(p) for p in self.probe_positions],
                }
            }
            self.metadata = Metadata(metadata_dict)
            self.sea_type="Signal"

    @property
    def data(self):
        """Lazy conversion to numpy for Signal compatibility."""
        if self._array is None:
            return None
        return to_numpy(self._array)

    @data.setter
    def data(self, value):
        self._array = value

    @property
    def adf(self):
        """Backward compatible alias for internal ADF array."""
        return self._array

    @adf.setter
    def adf(self, value):
        self._array = value

    @property
    def array(self):
        """Alias for adf (backward compatibility)."""
        return to_numpy(self._array)

    def __getattr__(self, name):
        """Auto-convert coordinate arrays from tensor to numpy on access."""
        coord_attrs = {'kxs', 'kys', 'xs', 'ys'}
        if name in coord_attrs:
            raw = object.__getattribute__(self, f'_{name}')
            if raw is None:
                return None
            return to_numpy(raw)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def getMask(self, inner_mrad: float = 45, outer_mrad: float = 150):
        """Build the annular detector mask on the wavefunction k-grid.

        Args:
            inner_mrad: Inner collection semi-angle in milliradians.
            outer_mrad: Outer collection semi-angle in milliradians.

        Returns:
            Backend-native array shaped ``(kx, ky)`` with ones inside the
            annulus and zeros elsewhere.

        Raises:
            ValueError: If the detector angles do not define a positive
                annulus, or if the inner angle begins outside the usable
                anti-aliased reciprocal-space bandwidth.

        Warns:
            RuntimeWarning: If the requested outer angle exceeds the largest
                fully represented annulus and enters a cropped or tapered
                reciprocal-space region.
        """
        inner_mrad, outer_mrad = self._validate_detector_angles(
            inner_mrad, outer_mrad
        )
        b = self._backend
        q = b.sqrt(self._kxs[:,None]**2 + self._kys[None,:]**2)
        radius_inner = (inner_mrad * 1e-3) / self.probe.wavelength
        radius_outer = (outer_mrad * 1e-3) / self.probe.wavelength

        mask = b.zeros(q.shape, type_match=self._wf_array)
        if isinstance(self._wf_array, np.memmap):
            q = to_numpy(q)
        mask[q >= radius_inner] = 1
        mask[q >= radius_outer] = 0
        return mask

    @property
    def max_detector_mrad(self) -> float:
        """Largest fully represented annular-detector angle in milliradians.

        Multislice propagation applies a circular 2/3-Nyquist anti-aliasing
        aperture with a narrow cosine taper. Returned wavefunctions may be
        cropped further in ``kx`` or ``ky``. The smaller of the untapered
        passband and stored-grid limits determines the largest complete,
        unattenuated detector circle.
        """
        probe_kx_max = float(np.max(np.abs(to_numpy(self.probe.kxs))))
        probe_ky_max = float(np.max(np.abs(to_numpy(self.probe.kys))))
        propagation_limit = (
            (ANTIALIAS_CUTOFF_FRACTION - ANTIALIAS_TAPER_WIDTH)
            * min(probe_kx_max, probe_ky_max)
        )

        stored_kx_max = float(np.max(np.abs(to_numpy(self._kxs))))
        stored_ky_max = float(np.max(np.abs(to_numpy(self._kys))))
        stored_limit = min(stored_kx_max, stored_ky_max)

        wavelength_A = float(np.asarray(to_numpy(self.probe.wavelength)))
        return min(propagation_limit, stored_limit) * wavelength_A * 1e3

    def _validate_detector_angles(
        self, inner_mrad: float, outer_mrad: float
    ) -> Tuple[float, float]:
        """Validate an annulus against ordering and reciprocal bandwidth."""
        inner_mrad = float(inner_mrad)
        outer_mrad = float(outer_mrad)
        if not np.isfinite(inner_mrad) or not np.isfinite(outer_mrad):
            raise ValueError("ADF detector angles must be finite milliradian values")
        if inner_mrad < 0 or outer_mrad <= inner_mrad:
            raise ValueError(
                "ADF detector angles must satisfy 0 <= inner_mrad < outer_mrad"
            )

        limit_mrad = self.max_detector_mrad
        if inner_mrad >= limit_mrad:
            raise ValueError(
                f"ADF inner angle {inner_mrad:g} mrad is outside the usable "
                f"reciprocal-space bandwidth ({limit_mrad:.1f} mrad). "
                "Decrease the real-space sampling or detector angle."
            )
        if outer_mrad > limit_mrad:
            warnings.warn(
                f"ADF outer angle {outer_mrad:g} mrad exceeds the largest fully "
                f"represented detector angle ({limit_mrad:.1f} mrad); the "
                "detector enters the cropped or tapered region of the "
                "multislice anti-aliasing aperture.",
                RuntimeWarning,
                stacklevel=3,
            )
        return inner_mrad, outer_mrad

    def calculateADF(self, inner_mrad: float = 45, outer_mrad: float = 150, preview: bool = False) -> np.ndarray:
        """
        Calculate the ADF (Annular Dark Field) image.

        Args:
            inner_mrad: Inner collection angle in milliradians (default: 45).
            outer_mrad: Outer collection angle in milliradians (default: 150).
                A warning is emitted when this exceeds ``max_detector_mrad``.
            preview: If True, show a preview of the first exit wave with mask

        Returns:
            ADF image array (x × y)
        """
        # Use float_dtype to ensure MPS compatibility (float32 on MPS, float64 otherwise)
        #self._xs = xp.asarray(sorted(list(set(self.probe_positions[:,0]))), dtype=float_dtype)
        #self._ys = xp.asarray(sorted(list(set(self.probe_positions[:,1]))), dtype=float_dtype)
        b = self._backend
        self._array = b.zeros((len(self._xs), len(self._ys)), type_match=self._wf_array)

        mask = self.getMask(inner_mrad, outer_mrad)

        # recall self._wf_array is reshaped: p,t,kx,ky,l --> c,x,y,t,kx,ky,l
        if preview:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            preview_data = b.mean(b.absolute(self._wf_array), axis=(0,1,2,3,6))**.2 * mask
            kxs = to_numpy(self._kxs)
            kys = to_numpy(self._kys)
            ax.imshow(
                to_numpy(b.absolute(preview_data)).T,
                cmap="inferno",
                origin="lower",
                extent=(kxs.min(), kxs.max(), kys.min(), kys.max()),
            )
            ax.set_xlabel("kx (Å⁻¹)")
            ax.set_ylabel("ky (Å⁻¹)")
            ax.set_title("ADF collection region")
            plt.show()

        nc,_,_,nt,_,_,nl = self._wf_array.shape
        wf_intensity = b.absolute(self._wf_array)**2 ; mask = b.absolute(mask)
        self._array = b.einsum('cxytkql,kq->xy', wf_intensity, mask) / (nc*nt*nl)

        xs_np = to_numpy(self._xs)
        ys_np = to_numpy(self._ys)

        if Dimensions is not None:
            self._local_dimensions = Dimensions([
                Dimension(name='x', space='position', units='Å', values=xs_np),
                Dimension(name='y', space='position', units='Å', values=ys_np),
            ], nav_dimensions=[0, 1], sig_dimensions=[])

            # Update metadata with detector settings
            #if hasattr(self.signal.metadata, 'Simulation'):
            self.metadata.Simulation.inner_mrad = inner_mrad
            self.metadata.Simulation.outer_mrad = outer_mrad

        return self.data  # Return numpy array for backward compatibility

    def plot(self, filename=None, title=None):
        """
        Plot the HAADF image.

        Args:
            filename: If provided, save plot to this file instead of displaying
            title: Optional axes title.
        """
        import matplotlib.pyplot as plt

        if self._array is None:
            raise RuntimeError("calculateADF() must be called before plotting")

        fig, ax = plt.subplots()
        array = self.array.T  # imshow rows are y; PySlice stores x,y
        xs = to_numpy(self._xs)
        ys = to_numpy(self._ys)

        dx = (xs[-1] - xs[0]) / (len(xs) - 1) if len(xs) > 1 else 0
        dy = (ys[-1] - ys[0]) / (len(ys) - 1) if len(ys) > 1 else 0
        extent = (np.amin(xs) - dx/2, np.amax(xs) + dx/2, np.amin(ys) - dy/2, np.amax(ys) + dy/2)
        ax.imshow(np.absolute(array), cmap="inferno", extent=extent, origin="lower")
        ax.set_xlabel("x ($\\AA$)")
        ax.set_ylabel("y ($\\AA$)")

        if title is not None:
            ax.set_title(title)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
