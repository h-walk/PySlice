import numpy as np
from tqdm import tqdm
import logging

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



logger = logging.getLogger(__name__)

m_electron = 9.109383e-31    # mass of an electron, kg
q_electron = 1.602177e-19    # charge of an electron, J / eV or kg m^2/s^2 / eV  
c_light = 299792458.0        # speed of light, m / s
h_planck = 6.62607015e-34    # m^2 kg / s


def m_effective(eV):
    """Relativistic correction: E=m*c^2, so m=E/c^2, in kg"""
    return m_electron + eV * q_electron / c_light**2

def wavelength(eV):
    return h_planck * c_light / ((eV * q_electron)**2 + 2 * eV * q_electron * m_electron * c_light**2)**0.5 * 1e10

class Probe:
    """
    PyTorch-accelerated probe class for electron microscopy.
    
    Generates probe wavefunctions on GPU for both plane wave and convergent beam modes.
    Significant speedup for large grid sizes through GPU-accelerated FFT operations.
    """
    
    def __init__(self, xs, ys, mrad, eV, array=None, device=None):
        """
        Initialize GPU-accelerated probe wavefunction.
        
        Args:
            xs, ys: Real space coordinate arrays
            mrad: Convergence semi-angle in milliradians (0.0 = plane wave)
            eV: Electron energy in eV
            device: PyTorch device (None for auto-detection)
        """
        if TORCH_AVAILABLE:
            # Auto-detect device if not specified (same logic as Potential class)
            if device is None:
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = torch.device('mps')
                else:
                    device = torch.device('cpu')
            elif isinstance(device, str):
                device = torch.device(device)
            self.device = device
            self.use_torch = True
            
            # Use float32 for MPS compatibility (same as Potential class)
            self.dtype = torch.float32 if device.type == 'mps' else torch.float64
            self.complex_dtype = torch.complex64 if device.type == 'mps' else torch.complex128
        else:
            if device is not None:
                raise ImportError("PyTorch not available. Please install PyTorch.")
            self.device = None
            self.use_torch = False
            self.dtype = np.float64
            self.complex_dtype = np.complex128
        
        self.xs = xs
        self.ys = ys
        self.mrad = mrad
        self.eV = eV
        self.wavelength = wavelength(eV)
        
        nx = len(xs)
        ny = len(ys)
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        
        # Set up device kwargs for unified xp interface (same as Potential class)
        device_kwargs = {'device': self.device, 'dtype': self.dtype} if self.use_torch else {}
        
        self.kxs = xp.fft.fftfreq(nx, d=dx, **device_kwargs)
        self.kys = xp.fft.fftfreq(ny, d=dy, **device_kwargs)

        if not array is None: # Allow construction of a Probe object with a passed array instead of building it below. used by create_batched_probes
            if self.use_torch and hasattr(array, 'to'):
                self.array = array.to(device=self.device, dtype=self.complex_dtype)
            else:
                self.array = xp.asarray(array)
            return
                    
        device_kwargs = {'device': self.device, 'dtype': self.dtype} if self.use_torch else {}
        if mrad == 0:
            self.array = xp.ones((nx, ny), **device_kwargs)
        else:
            reciprocal = xp.zeros((nx, ny), **device_kwargs)
            radius = (mrad * 1e-3) / self.wavelength  # Convert mrad to reciprocal space units
            
            kx_grid, ky_grid = xp.meshgrid(self.kxs, self.kys, indexing='ij')
            radii = xp.sqrt(kx_grid**2 + ky_grid**2)
            
            mask = radii < radius
            reciprocal[mask] = 1.0
            
            self.array = xp.fft.ifftshift(xp.fft.ifft2(reciprocal))
        
        #self.array_numpy = self.array.cpu().numpy()
    
    def copy(self):
        """Create a deep copy of the probe."""
        new_probe = ProbeTorch.__new__(ProbeTorch)
        new_probe.xs = self.xs.clone()
        new_probe.ys = self.ys.clone()
        new_probe.mrad = self.mrad
        new_probe.eV = self.eV
        new_probe.wavelength = self.wavelength
        new_probe.kxs = self.kxs.clone()
        new_probe.kys = self.kys.clone()
        new_probe.array = self.array.clone()
        new_probe.device = self.device
        new_probe.array_numpy = self.array_numpy.copy()
        return new_probe
    
    def to_cpu(self):
        """Convert probe array to CPU NumPy array."""
        return self.array.cpu().numpy()
    
    def to_device(self, device):
        """Move probe to specified device."""
        self.array = self.array.to(device)
        self.kxs = self.kxs.to(device)
        self.kys = self.kys.to(device)
        self.device = device
        return self

def probe_grid(xlims,ylims,n,m):
	x,y=np.meshgrid(np.linspace(*xlims,n),np.linspace(*ylims,m))
	return np.reshape([x,y],(2,len(x.flat))).T


def create_batched_probes(base_probe, probe_positions, device=None):
    """
    Create a batch of shifted probes for vectorized processing.
    
    Args:
        base_probe: ProbeTorch object
        probe_positions: List of (x,y) positions
        device: PyTorch device
        
    Returns:
        probe object with an array of shape (n_probes, nx, ny)
    """
    #if device is None:
    #    device = base_probe.device
        
    n_probes = len(probe_positions)
    probe_arrays = []
    
    for px, py in probe_positions:
        # Create shifted probe using phase ramp in k-space
        probe_k = xp.fft.fft2(base_probe.array)
        
        # Apply phase ramp for spatial shift
        kx_shift = xp.exp(2j * xp.pi * base_probe.kxs[:, None] * px)
        ky_shift = xp.exp(2j * xp.pi * base_probe.kys[None, :] * py)
        probe_k_shifted = probe_k * kx_shift * ky_shift
        
        # Convert back to real space
        shifted_probe_array = xp.fft.ifft2(probe_k_shifted)
        probe_arrays.append(shifted_probe_array)
    
    # Stack into batch tensor
    if TORCH_AVAILABLE:
        array = torch.stack(probe_arrays, dim=0)
    else:
        array = xp.asarray(probe_arrays)

    return Probe(base_probe.xs, base_probe.ys, base_probe.mrad, base_probe.eV, array=array, device=base_probe.device)

def Propagate(probe, potential, device=None):
    """
    PyTorch-accelerated multislice propagation function.
    Supports both single probe and batched multi-probe processing.
    
    Args:
        probe: ProbeTorch object or tensor with shape (n_probes, nx, ny)
        potential: Potential object (can be NumPy or PyTorch version)
        device: PyTorch device (None for auto-detection)
        
    Returns:
        torch.Tensor: Exit wavefunction(s) after multislice propagation
    """
    if device is not None and not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Please install PyTorch.")
    

    if len(probe.array.shape) == 2:
        probe.array = probe.array[None,:,:]
    
    # Calculate interaction parameter (Kirkland Eq 5.6)
    E0_eV = m_electron * c_light**2 / q_electron
    sigma = (2 * np.pi) / (probe.wavelength * probe.eV) * \
            (E0_eV + probe.eV) / (2 * E0_eV + probe.eV)
    if TORCH_AVAILABLE:
        #sigma_dtype = torch.float32 if device.type == 'mps' else torch.float64
        sigma = torch.tensor(sigma, dtype=float_dtype, device=device)
    
    # Get slice thickness
    dz = potential.zs[1] - potential.zs[0] if len(potential.zs) > 1 else 0.5
    
    # Initialize wavefunction with probe(s) - shape: (n_probes, nx, ny)
    array = probe.array #.clone()
    
    # Pre-compute propagation operator in k-space (Fresnel propagation)
    # All tensors should already be on the correct device from creation
    kx_grid, ky_grid = xp.meshgrid(potential.kxs, potential.kys, indexing='ij')
    k_squared = kx_grid**2 + ky_grid**2
    P = xp.exp(-1j * xp.pi * probe.wavelength * dz * k_squared)
    
    # Vectorized multislice propagation through each slice
    for z in range(len(potential.zs)):
        # Transmission function: t = exp(iσV(x,y,z))
        # All tensors should already be on the correct device from creation
        potential_slice = potential.array[:, :, z]
        t = xp.exp(1j * sigma * potential_slice)
        
        # Apply transmission to all probes: ψ' = t × ψ
        # Broadcasting: t[nx,ny] * array[n_probes,nx,ny] = array[n_probes,nx,ny]
        array = t[None, :, :] * array
        
        # Fresnel propagation to next slice (except for last slice)
        if z < len(potential.zs) - 1:
            # Vectorized FFT over spatial dimensions for all probes
            kwarg = {"dim":(-2,-1)} if TORCH_AVAILABLE else {"axes":(-2,-1)}
            fft_array = xp.fft.fft2(array, **kwarg)
            propagated_fft = P[None, :, :] * fft_array
            array = xp.fft.ifft2(propagated_fft, **kwarg)
    
    # Return single probe result if input was single, otherwise return batch
    if array.shape[0] == 1:
        return array.squeeze(0)
    return array

