"""
Wave function data structure.
"""
import numpy as np
from typing import List, Tuple, Optional
from ..multislice.multislice import Probe,aberrationFunction
from ..data.pyslice_serial import PySliceSerial, Signal, Dimensions, Dimension, Metadata
from pathlib import Path
from ..backend import mean,ones,zeros,reshape,absolute,sum,asarray,cumsum,randfloats,histogram

try:
    import torch ; xp = torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if device.type == 'mps':
        complex_dtype = torch.complex64
        float_dtype = torch.float32
    else:
        complex_dtype = torch.complex128
        float_dtype = torch.float64
except ImportError:
    TORCH_AVAILABLE = False
    xp = np
    print("PyTorch not available, falling back to NumPy")
    complex_dtype = np.complex128
    float_dtype = np.float64


class WFData(PySliceSerial, Signal):
    """
    Data structure for wave function data with format: probe_positions, frame, kx, ky, layer.

    Inherits from Signal for sea-eco compatibility.

    Attributes:
        probe_positions: List of (x,y) probe positions in Angstroms.
        time: Time array (frame # * timestep) in picoseconds.
        kxs: kx sampling vectors.
        kys: ky sampling vectors.
        xs: x real-space coordinates.
        ys: y real-space coordinates.
        layer: Layer indices for multi-layer calculations.
        array: Complex wavefunction array with shape (probe_positions, time, kx, ky, layer).
        probe: Probe object with beam parameters.
        cache_dir: Path to cache directory.
    """

    _sea_config = {
        'tensor_attrs': ['_kxs', '_kys', '_xs', '_ys', '_time', '_layer', '_array'],
        'path_attrs': ['cache_dir'],
        'tuple_list_attrs': ['probe_positions'],
        'exclude_attrs': ['probe'],
        'force_datasets': ['_array', 'probe_positions', '_kxs', '_kys', '_xs', '_ys', '_time', '_layer'],
    }

    def __init__(
        self,
        probe_positions: List[Tuple[float, float]],
        probe_xs: List[float],
        probe_ys: List[float],
        time: np.ndarray,
        kxs: np.ndarray,
        kys: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        layer: np.ndarray,
        array: np.ndarray,
        probe: Probe,
        cache_dir: Path,
    ):
        # Store raw attributes (may be tensors for GPU operations)
        self.probe_positions = probe_positions
        self.probe_xs = probe_xs
        self.probe_ys = probe_ys
        self._time = time
        self._kxs = kxs
        self._kys = kys
        self._xs = xs
        self._ys = ys
        self._layer = layer
        self.probe = probe
        self.cache_dir = cache_dir
        self.probability = None

        # Helper to convert tensors to numpy for Dimensions
        def to_numpy(x):
            if hasattr(x, 'cpu'):
                return x.cpu().numpy()
            return np.asarray(x)

        # Build Dimensions for Signal
        time_arr = to_numpy(time)
        kxs_arr = to_numpy(kxs)
        kys_arr = to_numpy(kys)
        layer_arr = to_numpy(layer) if layer is not None else np.array([0])

        if Dimensions is not None:
            dimensions = Dimensions([
                Dimension(name='probe', space='position',
                        values=np.arange(len(probe_positions))),
                Dimension(name='time', space='temporal', units='ps',
                        values=time_arr),
                Dimension(name='kx', space='scattering', units='Å⁻¹',
                        values=kxs_arr),
                Dimension(name='ky', space='scattering', units='Å⁻¹',
                        values=kys_arr),
                Dimension(name='layer', space='position',
                        values=layer_arr),
            ], nav_dimensions=[0, 1], sig_dimensions=[2, 3, 4])

            # Build metadata from simulation parameters
            # Flatten probe_positions for HDF5 compatibility, store n_probes to reshape on load
            pp_array = np.array(probe_positions).flatten().tolist()
            metadata_dict = {
                'General': {
                    'title': 'Multislice Wavefunction',
                    'signal_type': 'Wavefunction'
                },
                'Simulation': {
                    'voltage_eV': float(probe.eV),
                    'wavelength_A': float(probe.wavelength),
                    'aperture_mrad': float(probe.mrad),
                    'probe_positions': pp_array,
                    'n_probes': len(probe_positions),
                }
            }
            metadata = Metadata(metadata_dict)

        # Store array AFTER super().__init__ to avoid being overwritten
        self._array = array

    @property
    def data(self):
        """Lazy conversion to numpy for Signal compatibility."""
        if self._array is None:
            return None
        if hasattr(self._array, 'cpu'):
            return self._array.cpu().numpy()
        return np.asarray(self._array)

    @data.setter
    def data(self, value):
        self._array = value

    @property
    def array(self):
        """Backward compatible alias for internal array (may be tensor or numpy)."""
        return self._array

    #@property
    def reshaped(self): # where self._array is indices probe,time,kx,ky,layer, we reshape to probe_x,probe_y,time,kx,ky,layer
        nc,nptp,nx,ny = self.probe._array.shape # recall: decoherence creates duplicate probes: num_copies,num_positions,x,y indices
        nptp = len(self.probe_positions)
        npta,nt,nkx,nky,nl = self._array.shape # recall, Propagate flattens the first two, and adds time,layers: nc*npt,num_frames,x,y,nl indice
        intermediate = reshape(self._array,(nc,nptp,nt,nkx,nky,nl))
        nx,ny = len(self.probe_xs),len(self.probe_ys)
        return reshape(intermediate,(nc,ny,nx,nt,nkx,nky,nl)).swapaxes(1,2)

    @array.setter
    def array(self, value):
        self._array = value

    def __getattr__(self, name):
        """Auto-convert coordinate arrays from tensor to numpy on access."""
        coord_attrs = {'time', 'kxs', 'kys', 'xs', 'ys', 'layer'}
        if name in coord_attrs:
            raw = object.__getattribute__(self, f'_{name}')
            if raw is None:
                return None
            if hasattr(raw, 'cpu'):
                return raw.cpu().numpy()
            return np.asarray(raw)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def counts(self,N):
        if self.probability is None:
            self.probability = self._array
            npt,nt,nx,ny,nl = self._array.shape
            ary = self._array/xp.sum(xp.absolute(self._array))                  # normalized: ensure values arerelative probabilities of each voxel
            ary = xp.absolute(ary.reshape(npt*nt*nx*ny*nl))
            self.buckets = zeros(len(ary)+1,type_match=ary)
            self.buckets[1:] = cumsum(ary)                                      # cumsum means we can "select" a voxel with a random float 0-1
        detector_hits = asarray(randfloats(N))                                  # randomly "select" histogram bins based on each bin's relative size
        hist = histogram(detector_hits,bins=self.buckets)
        self._array = asarray(hist.reshape((npt,nt,nx,ny,nl)))

    def plot_reciprocal(self,filename=None,whichProbe="mean",whichTimestep="mean",powerscaling=0.25,extent=None,nuke_zerobeam=False,title=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        raw = self._array[:,:,:,:,-1] # probe, time, kx, ky, layer --> p,t,kx,ky
        npt,nt,nkx,nky = raw.shape
        array = zeros((nkx,nky), type_match=raw)
        if isinstance(whichProbe,str) and whichProbe=="mean":
            whichProbe = np.arange(npt)
        elif isinstance(whichProbe,int):
            whichProbe = [whichProbe]
        if isinstance(whichTimestep,str) and whichTimestep=="mean":
            whichTimestep = np.arange(nt)
        elif isinstance(whichTimestep,int):
            whichTimestep = [whichTimestep]
        for p in whichProbe:
            for t in whichTimestep:
                layer = absolute(raw[p,t,:,:])
                array+=layer
        array/=(len(whichTimestep)*len(whichProbe))
        #array=abs(raw) # don't do this, it pulls memmaps into ram! 
        #if isinstance(whichProbe,str) and whichProbe=="mean":
        #    array = mean(abs(array),axis=0) # p,t,kx,ky --> t,kx,ky
        #else:
        #    array = array[whichProbe] 
        #
        #if isinstance(whichTimestep,str) and whichTimestep=="mean":
        #    array = mean(array,axis=0) # t,kx,ky --> kx,ky
        #else:
        #    array = array[whichTimestep] 

        # Convert kxs and kys to numpy for indexing
        if hasattr(self.kxs, 'cpu'):
            kxs_np = self.kxs.cpu().numpy()
            kys_np = self.kys.cpu().numpy()
        else:
            kxs_np = np.asarray(self.kxs)
            kys_np = np.asarray(self.kys)

        # If extent is provided, slice the data
        if extent is not None:
            kx_min, kx_max, ky_min, ky_max = extent

            # Find indices for the requested extent
            kx_mask = (kxs_np >= kx_min) & (kxs_np <= kx_max)
            ky_mask = (kys_np >= ky_min) & (kys_np <= ky_max)

            # Slice the array and coordinate arrays
            array = array[kx_mask, :][:, ky_mask]
            kxs_np = kxs_np[kx_mask]
            kys_np = kys_np[ky_mask]
            actual_extent = (kxs_np[0], kxs_np[-1],
                           kys_np[0], kys_np[-1])
        else:
            # Use full extent
            kxs_min = float(kxs_np.min())
            kxs_max = float(kxs_np.max())
            kys_min = float(kys_np.min())
            kys_max = float(kys_np.max())
            actual_extent = (kxs_min, kxs_max, kys_min, kys_max)

        # Transpose for imshow convention
        array = array.T  # imshow convention: y,x. our convention: x,y
        if nuke_zerobeam:
            array[np.argmin(np.absolute(kys_np)),np.argmin(np.absolute(kxs_np))]=0

        # Convert to numpy array if it's a tensor
        # Apply powerscaling to intensity (|Ψ|²)
        img_data = (absolute(array)**2)**powerscaling
        if hasattr(img_data, 'cpu'):
            img_data = img_data.cpu().numpy()
        elif hasattr(img_data, '__array__'):
            img_data = np.asarray(img_data)
        ax.imshow(img_data, cmap="inferno", extent=actual_extent, origin='lower',aspect=1)
        ax.set_xlabel("kx ($\\AA^{-1}$)")
        ax.set_ylabel("ky ($\\AA^{-1}$)")

        if title is not None:
            ax.set_title(title)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    plot = plot_reciprocal

    def plot_phase(self,filename=None,whichProbe=0,whichTimestep=0,extent=None,avg=False):
        """
        Plot the phase of the wavefunction in real space.

        Args:
            whichProbe: Probe index
            whichTimestep: Timestep index
            extent: Optional (xmin, xmax, ymin, ymax) to zoom
            avg: If True, average over all timesteps before plotting
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        # Get array (with or without averaging)
        if avg:
            array = self._array[whichProbe,:,:,:,-1] # Shape: (time, kx, ky)
            if hasattr(array, 'mean'):  # torch tensor
                array = array.mean(dim=0)  # Average over time dimension
            else:  # numpy array
                array = np.mean(array, axis=0)
        else:
            array = self._array[whichProbe,whichTimestep,:,:,-1]

        # Transform to real space
        array = xp.fft.ifft2(array)
        xs_np = np.asarray(self.xs)
        ys_np = np.asarray(self.ys)

        # If extent is provided, slice the data
        if extent is not None:
            x_min, x_max, y_min, y_max = extent

            # Find indices for the requested extent
            x_mask = (xs_np >= x_min) & (xs_np <= x_max)
            y_mask = (ys_np >= y_min) & (ys_np <= y_max)

            # Slice the array
            array = array[x_mask, :][:, y_mask]
            actual_extent = (xs_np[x_mask][0], xs_np[x_mask][-1],
                           ys_np[y_mask][0], ys_np[y_mask][-1])
        else:
            # Use full extent
            actual_extent = (float(xs_np.min()), float(xs_np.max()),
                           float(ys_np.min()), float(ys_np.max()))

        # Transpose for imshow convention
        array = array.T  # imshow convention: y,x. our convention: x,y

        # Get phase
        phase_data = xp.angle(array)
        if hasattr(phase_data, 'cpu'):
            phase_data = phase_data.cpu().numpy()
        elif hasattr(phase_data, '__array__'):
            phase_data = np.asarray(phase_data)

        # Plot with phase colormap
        im = ax.imshow(phase_data, cmap='hsv', extent=actual_extent, origin='lower',
                       vmin=-np.pi, vmax=np.pi)
        plt.colorbar(im, ax=ax, label='Phase (radians)')
        ax.set_title('Phase in real space')
        ax.set_xlabel('x (Å)')
        ax.set_ylabel('y (Å)')

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    def plot_realspace(self,whichProbe="mean",whichTimestep="mean",extent=None,filename=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        array = xp.fft.ifft2(self._array[:,:,:,:,-1])

        array = xp.absolute(array) # probe, time, kx, ky, layer --> p,t,kx,ky

        if isinstance(whichProbe,str) and whichProbe=="mean":
            array = mean(abs(array),axis=0) # p,t,kx,ky --> t,kx,ky
        else:
            array = array[whichProbe]

        if isinstance(whichTimestep,str) and whichTimestep=="mean":
            array = mean(array,axis=0) # t,kx,ky --> kx,ky
        else:
            array = array[whichTimestep]

        array = array.T # imshow convention: y,x. our convention: x,y

        # Use provided extent or calculate from data
        if extent is None:
            extent = ( np.amin(self.xs) , np.amax(self.xs) , np.amin(self.ys) , np.amax(self.ys) )

        # Convert to numpy array if it's a tensor
        img_data = xp.absolute(array)**.25
        if hasattr(img_data, 'cpu'):
            img_data = img_data.cpu().numpy()
        elif hasattr(img_data, '__array__'):
            img_data = np.asarray(img_data)

        ax.imshow( img_data, cmap="inferno", extent=extent )

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    def propagate_free_space(self,dz): # UNITS OF ANGSTROM
        kx_grid, ky_grid = xp.meshgrid(self._kxs, self._kys, indexing='ij')
        k_squared = kx_grid**2 + ky_grid**2
        inner = xp.pi * self.probe.wavelength * dz * k_squared
        P = xp.exp( -1j * inner ) # not sure why, but combining this and previous line triggers a "ComplexWarning: Casting complex values to real discards the imaginary part" in python 2.9.1 but not 2.2.2
        if TORCH_AVAILABLE and isinstance(self._array, torch.Tensor):
            P = P.to(self._array.device)
        #if dz>0:
        self._array = P[None,None,:,:,None] * self._array

    def addSpatialDecoherence(self,sigma_dz,N):
        dzs = np.linspace(-2*sigma_dz,2*sigma_dz,N) # suppose N=25
        amplitudes = np.exp(-dzs**2/sigma_dz**2)
        self._array = self._array[:,None,:,:,:,:] * ones(N)[None,:,None,None,None,None] # n_probes,nt,nx,ny,nl -->
        nc,npt,nt,nx,ny,nl = self._array.shape            # suppose nc=10 (addTemporalDecoherence created 10 wavelengths)
        kx_grid, ky_grid = xp.meshgrid(self._kxs, self._kys, indexing='ij')
        k_squared = kx_grid**2 + ky_grid**2
        for i in range(N):
            inner = xp.pi * self.probe.wavelength * dzs[i] * k_squared
            P = xp.exp( -1j * inner ) # not sure why, but combining this and previous line triggers a "ComplexWarning: Casting complex values to real discards the imaginary part" in python 2.9.1 but not 2.2.2
            self._array[:,i,:,:,:,:] *= amplitudes[i]*P[None,None,:,:,None]
        self._array = self._array.reshape(nc*npt,nt,nx,ny,nl)
        #self.defocus(dzs)                           # defocus starts with 25,10,npt,nx,ny --reshapes--> 250,npt,nx,ny
        #for i in range(N):                         # reshape to flatten loops first index last: [[0,1],[2,3]] --> [0,1,2,3]
        #    for j in range(nc):
        #        self._array[i*nc+j] *= amplitudes[i]
        #nc,npt,nx,ny = self._array.shape
        #if npt==1:
        #    self.applyShifts()


    def applyMask(self, radius, realOrReciprocal="reciprocal"):
        if realOrReciprocal == "reciprocal":
            radii = xp.sqrt( self._kxs[:,None]**2 + self._kys[None,:]**2 )
            mask = zeros(radii.shape, device=self._array.device if TORCH_AVAILABLE else None)
            mask[radii<radius]=1
            self._array*=mask[None,None,:,:,None]
        else:
            # Use numpy for _xs/_ys since they're numpy arrays, then convert result
            radii_np = np.sqrt( ( self._xs[:,None] - np.mean(self._xs) )**2 +\
                ( self._ys[None,:] - np.mean(self._ys) )**2 )
            if TORCH_AVAILABLE:
                radii = xp.tensor(radii_np, dtype=self._array.real.dtype, device=self._array.device)
            else:
                radii = radii_np
            mask = zeros(radii.shape, device=self._array.device if TORCH_AVAILABLE else None)
            mask[radii<radius]=1
            kwarg = {"dim":(2,3)} if TORCH_AVAILABLE else {"axes":(2,3)}
            real = xp.fft.ifft2(xp.fft.ifftshift(self._array,**kwarg),**kwarg)
            real *= mask[None,None,:,:,None]
            self._array = xp.fft.fftshift(xp.fft.fft2(real,**kwarg),**kwarg)

    def crop(self,kx_range=None,ky_range=None):
        npt,nt,nx,ny,nl = self._array.shape
        i1=0 ; i2=nx ; j1=0 ; j2=ny
        if kx_range is not None:
            i1=xp.argwhere(self._kxs >= kx_range[0])[0]    # first element >=
            i2=xp.argwhere(self._kxs <= kx_range[1])[-1]+1 # last element <=, +1, so i1:i2 includes i2
        if ky_range is not None:
            j1=xp.argwhere(self._kys >= ky_range[0])[0]
            j2=xp.argwhere(self._kys <= ky_range[1])[-1]+1
        nx=i2-i1 ; ny=j2-j1
        self._array = self._array[:,:,i1:i2,j1:j2,:] # p,t,x,y,l indices: TODO this uses the same amount of RAM
        #self._array = xp.zeros((npt,nt,nx,ny,nl), device=self._array.device if TORCH_AVAILABLE else None) +\
        #      self._array[:,:,i1:i2,j1:j2,:] # p,t,x,y,l indices TODO this actually uses MORE RAM? 
        self._kxs = self._kxs[i1:i2]
        self._kys = self._kys[j1:j2]

    def aberrate(self,aberrations):
        dP = aberrationFunction(self._kxs,self._kys,self.probe.wavelength,aberrations)
        self._array[:,:,:,:,:] *= dP[None,None,:,:,None] # indices npt,nt,kx,ky,nl
