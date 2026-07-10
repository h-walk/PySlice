"""
HAADF (High Angle Annular Dark Field) data structure.
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import logging
from .wf_data import WFData
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
            dimensions = Dimensions([
                Dimension(name='x', space='position', units='Å', values=np.array([0])),
                Dimension(name='y', space='position', units='Å', values=np.array([0])),
            ], nav_dimensions=[0, 1], det_dimensions=[])

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
            metadata = Metadata(metadata_dict)
            Signal.__init__(
                self,
                data=None,
                name='HAADF Image',
                dimensions=dimensions,
                signal_type='Image',
                metadata=metadata,
            )
            self.sea_type="Signal"
            record_pyslice_operation(
                self,
                "HAADFData.from_wf_data",
                inputs=[wf_data],
                parameters={
                    "n_probe_positions": len(self.probe_positions),
                    "k_shape": (len(self._kxs), len(self._kys)),
                    "image_shape": (len(self._xs), len(self._ys)),
                    "cache_dir": None if self.cache_dir is None else str(self.cache_dir),
                },
                callable_obj=type(self).__init__,
            )

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

    @track_pyslice_action
    def calculateADF(self, inner_mrad: float = 45, outer_mrad: float = 150, preview: bool = False) -> np.ndarray:
        """
        Calculate the ADF (Annular Dark Field) image.

        Args:
            inner_mrad: Inner collection angle in milliradians (default: 45)
            outer_mrad: Outer collection angle in milliradians (default: 150)
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
            preview_data = b.mean(b.absolute(self._wf_array), axis=(0,1,2,3,6))**.2 * (1 - mask)
            ax.imshow(to_numpy(b.absolute(preview_data)), cmap="inferno")
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
            ], nav_dimensions=[0, 1], det_dimensions=[])

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
        """
        import matplotlib.pyplot as plt

        if self._array is None:
            raise RuntimeError("calculateADF() must be called before plotting")

        fig, ax = plt.subplots()
        array = self.array.T[::-1,:]  # imshow convention: y,x. our convention: x,y, and flip y (0,0 upper-left)
        xs = to_numpy(self._xs)
        ys = to_numpy(self._ys)

        dx = (xs[-1] - xs[0]) / (len(xs) - 1) if len(xs) > 1 else 0
        dy = (ys[-1] - ys[0]) / (len(ys) - 1) if len(ys) > 1 else 0
        extent = (np.amin(xs) - dx/2, np.amax(xs) + dx/2, np.amin(ys) - dy/2, np.amax(ys) + dy/2)
        ax.imshow(np.absolute(array), cmap="inferno", extent=extent)
        ax.set_xlabel("x ($\\AA$)")
        ax.set_ylabel("y ($\\AA$)")

        if title is not None:
            ax.set_title(title)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
