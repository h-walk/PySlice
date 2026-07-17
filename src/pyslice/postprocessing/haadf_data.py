"""
HAADF (High Angle Annular Dark Field) data structure.
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import logging
from .wf_data import WFData
from ..data.pyslice_serial import PySliceSerial, Signal, Dimensions, Dimension, Metadata
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
        # Identity of each stored layer (thickness); used when calculateADF
        # returns one ADF image per layer.
        self.layers = getattr(wf_data, '_layer', None)

        # Initialize ADF as None, will be computed by calculateADF
        self._array = None
        self._xs = wf_data.probe_xs
        self._ys = wf_data.probe_ys

        if Dimensions is not None:
            self._set_dimensions()

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

    def _set_dimensions(self, layer_values=None, layer_name='layer',
                        layer_units=None):
        """Synchronise Signal dimensions with the current ADF array shape."""
        if Dimensions is None:
            return
        dimensions = []
        if layer_values is not None:
            layer_kwargs = {
                'name': layer_name,
                'space': 'position',
                'values': to_numpy(layer_values),
            }
            if layer_units is not None:
                layer_kwargs['units'] = layer_units
            dimensions.append(Dimension(**layer_kwargs))
        dimensions.extend([
            Dimension(name='x', space='position', units='Å',
                      values=to_numpy(self._xs)),
            Dimension(name='y', space='position', units='Å',
                      values=to_numpy(self._ys)),
        ])
        dims = Dimensions(
            dimensions,
            nav_dimensions=list(range(len(dimensions))),
            sig_dimensions=[],
        )
        # PySEA uses the public dimensions during normal operation and the
        # local copy during serialisation/deserialisation. Keep both congruent.
        self.dimensions = dims
        self._local_dimensions = dims

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
            radius_inner = to_numpy(radius_inner)
            radius_outer = to_numpy(radius_outer)
        mask[q >= radius_inner] = 1
        mask[q >= radius_outer] = 0
        return mask

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
        # One ADF image per stored layer (thickness). Only the exit wave
        # physically reaches the detector, but storing several layers gives ADF
        # vs thickness. Collapse to a plain 2D image for the single-layer case.
        # Probe copies already carry normalised intensity weights; sum them,
        # then average only over time. Dividing by nc would make the signal
        # vanish as more quadrature points are used.
        stack = b.einsum('cxytkql,kq->lxy', wf_intensity, mask) / nt
        self._array = stack[0] if nl == 1 else stack

        xs_np = to_numpy(self._xs)
        ys_np = to_numpy(self._ys)

        if Dimensions is not None:
            layer_values = None
            if nl > 1:
                layer_values = (self.layers if self.layers is not None
                                else np.arange(nl))
            self._set_dimensions(layer_values)

            # Update metadata with detector settings
            #if hasattr(self.signal.metadata, 'Simulation'):
            self.metadata.Simulation.inner_mrad = inner_mrad
            self.metadata.Simulation.outer_mrad = outer_mrad

        return self.data  # Return numpy array for backward compatibility

    def plot(self, filename=None, title=None, layer=-1):
        """
        Plot the HAADF image.

        Args:
            filename: If provided, save plot to this file instead of displaying
            layer: Which thickness to plot when the ADF is a per-layer stack
                (default -1, i.e. the exit wave). Ignored for a single-layer ADF.
        """
        import matplotlib.pyplot as plt

        if self._array is None:
            raise RuntimeError("calculateADF() must be called before plotting")

        fig, ax = plt.subplots()
        img = to_numpy(self._array)
        if img.ndim == 3:            # (n_layers, x, y) stack -> pick a thickness
            img = img[layer]
        array = img.T[::-1,:]  # imshow convention: y,x. our convention: x,y, and flip y (0,0 upper-left)
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
