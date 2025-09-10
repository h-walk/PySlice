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
    else:
        complex_dtype = torch.complex128

except ImportError:
    TORCH_AVAILABLE = False
    xp = np
    print("PyTorch not available, falling back to NumPy")
    complex_dtype = np.complex128

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
    
    def __init__(self, xs, ys, mrad, eV, device=None):
        """
        Initialize GPU-accelerated probe wavefunction.
        
        Args:
            xs, ys: Real space coordinate arrays
            mrad: Convergence semi-angle in milliradians (0.0 = plane wave)
            eV: Electron energy in eV
            device: PyTorch device (None for auto-detection)
        """
        if TORCH_AVAILABLE:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = device
        
            self.xs = torch.tensor(xs, dtype=torch.float32, device=device)
            self.ys = torch.tensor(ys, dtype=torch.float32, device=device)
        else:
            if device is not None:
                raise ImportError("PyTorch not available. Please install PyTorch.")
            self.xs = xs
            self.ys = ys
        
        self.mrad = mrad
        self.eV = eV
        self.wavelength = wavelength(eV)
        
        nx = len(xs)
        ny = len(ys)
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        
        self.kxs = xp.fft.fftfreq(nx, d=dx, device=device)
        self.kys = xp.fft.fftfreq(ny, d=dy, device=device)
                    
        if mrad == 0:
            self.array = xp.ones((nx, ny), dtype=complex_dtype, device=device)
        else:
            reciprocal = xp.zeros((nx, ny), dtype=complex_dtype, device=device)
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
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Please install PyTorch.")
    
    # Auto-detect device if not specified
    if device is None:
        device = probe.device if hasattr(probe, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle both single probe and batched multi-probe input
    if hasattr(probe, 'array'):
        # Single probe object
        if isinstance(probe.array, torch.Tensor):
            probe_array = probe.array.to(device)
            if probe_array.dim() == 2:
                probe_array = probe_array.unsqueeze(0)  # Add batch dimension
        else:
            # Convert from NumPy
            if device.type == 'mps':
                probe_array = torch.tensor(probe.array, dtype=torch.complex64, device=device)
            else:
                probe_array = torch.tensor(probe.array, dtype=torch.complex128, device=device)
            if probe_array.dim() == 2:
                probe_array = probe_array.unsqueeze(0)
                
        probe_kxs = probe.kxs.to(device) if hasattr(probe, 'kxs') else torch.fft.fftfreq(probe_array.shape[-2], device=device)
        probe_kys = probe.kys.to(device) if hasattr(probe, 'kys') else torch.fft.fftfreq(probe_array.shape[-1], device=device)
        probe_eV = probe.eV
        probe_wavelength = probe.wavelength
    else:
        # Direct tensor input (batched probes)
        probe_array = probe.to(device)
        if probe_array.dim() == 2:
            probe_array = probe_array.unsqueeze(0)
        # Need to get these from somewhere - assume they're passed or use defaults
        probe_kxs = torch.fft.fftfreq(probe_array.shape[-2], device=device)
        probe_kys = torch.fft.fftfreq(probe_array.shape[-1], device=device)
        # These would need to be passed as well for tensor input
        probe_eV = 100000  # Default 100keV
        probe_wavelength = wavelength(probe_eV)
    
    # Convert potential to PyTorch if needed
    if hasattr(potential, 'array_torch'):
        # Use PyTorch version if available
        potential_array = potential.array_torch.to(device)
        # All these are already tensors in PotentialTorch
        potential_kxs = potential.kxs.to(device)
        potential_kys = potential.kys.to(device)
        potential_zs = potential.zs.to(device)
    else:
        # Convert from NumPy version
        potential_array = torch.tensor(potential.array, dtype=torch.float32, device=device)
        potential_kxs = torch.tensor(potential.kxs, dtype=torch.float32, device=device)
        potential_kys = torch.tensor(potential.kys, dtype=torch.float32, device=device)
        potential_zs = torch.tensor(potential.zs, dtype=torch.float32, device=device)
    
    # Calculate interaction parameter (Kirkland Eq 5.6)
    E0_eV = m_electron * c_light**2 / q_electron
    sigma = (2 * np.pi) / (probe_wavelength * probe_eV) * \
            (E0_eV + probe_eV) / (2 * E0_eV + probe_eV)
    sigma_dtype = torch.float32 if device.type == 'mps' else torch.float64
    sigma = torch.tensor(sigma, dtype=sigma_dtype, device=device)
    
    # Get slice thickness
    dz = potential_zs[1] - potential_zs[0] if len(potential_zs) > 1 else 0.5
    
    # Pre-compute propagation operator in k-space (Fresnel propagation)
    kx_grid, ky_grid = torch.meshgrid(potential_kxs, potential_kys, indexing='ij')
    k_squared = kx_grid**2 + ky_grid**2
    P = torch.exp(-1j * torch.pi * probe_wavelength * dz * k_squared)
    
    # Initialize wavefunction with probe(s) - shape: (n_probes, nx, ny)
    array = probe_array.clone()
    
    # Vectorized multislice propagation through each slice
    for z in range(len(potential_zs)):
        # Transmission function: t = exp(iσV(x,y,z))
        potential_slice = potential_array[:, :, z]
        t = torch.exp(1j * sigma * potential_slice)
        
        # Apply transmission to all probes: ψ' = t × ψ
        # Broadcasting: t[nx,ny] * array[n_probes,nx,ny] = array[n_probes,nx,ny]
        array = t[None, :, :] * array
        
        # Fresnel propagation to next slice (except for last slice)
        if z < len(potential_zs) - 1:
            # Vectorized FFT over spatial dimensions for all probes
            fft_array = torch.fft.fft2(array, dim=(-2, -1))
            propagated_fft = P[None, :, :] * fft_array
            array = torch.fft.ifft2(propagated_fft, dim=(-2, -1))
    
    # Return single probe result if input was single, otherwise return batch
    if array.shape[0] == 1:
        return array.squeeze(0)
    return array


def PropagateTorchToNumpy(probe, potential, device=None):
    """
    Convenience function that runs PyTorch-accelerated propagation and returns NumPy array.
    
    Args:
        probe: Probe object (NumPy or PyTorch)
        potential: Potential object (NumPy or PyTorch)  
        device: PyTorch device (None for auto-detection)
        
    Returns:
        numpy.ndarray: Exit wavefunction as NumPy array
    """
    if not TORCH_AVAILABLE:
        # Fallback to NumPy implementation
        from .multislice_npy import Propagate
        return Propagate(probe, potential)
    
    # Run PyTorch version and convert result
    result_torch = PropagateTorch(probe, potential, device)
    return result_torch.cpu().numpy()


# Compatibility aliases for drop-in replacement
#if TORCH_AVAILABLE:
#    # PyTorch versions (recommended)
#    Probe = ProbeTorch
#    Propagate = PropagateTorchToNumpy
#else:
#    # Fallback to NumPy versions
#    from .multislice_npy import Probe, Propagate
#    print("Warning: PyTorch not available, using NumPy fallback")


def create_batched_probes(base_probe, probe_positions, device=None):
    """
    Create a batch of shifted probes for vectorized processing.
    
    Args:
        base_probe: ProbeTorch object
        probe_positions: List of (x,y) positions
        device: PyTorch device
        
    Returns:
        torch.Tensor: Batched probe array (n_probes, nx, ny)
    """
    if device is None:
        device = base_probe.device
        
    n_probes = len(probe_positions)
    probe_arrays = []
    
    for px, py in probe_positions:
        # Create shifted probe using phase ramp in k-space
        probe_k = torch.fft.fft2(base_probe.array)
        
        # Apply phase ramp for spatial shift
        kx_shift = torch.exp(2j * torch.pi * base_probe.kxs[:, None] * px)
        ky_shift = torch.exp(2j * torch.pi * base_probe.kys[None, :] * py)
        probe_k_shifted = probe_k * kx_shift * ky_shift
        
        # Convert back to real space
        shifted_probe_array = torch.fft.ifft2(probe_k_shifted)
        probe_arrays.append(shifted_probe_array)
    
    # Stack into batch tensor
    return torch.stack(probe_arrays, dim=0)


def PropagateBatch(probes_or_positions, potential, base_probe=None, device=None):
    """
    Process multiple probes in vectorized fashion on GPU.
    
    Args:
        probes_or_positions: Either list of probe positions [(x,y), ...] or batched tensor
        potential: Potential object
        base_probe: Base probe for position shifting (if using positions)
        device: PyTorch device
        
    Returns:
        torch.Tensor: Batched exit wavefunctions (n_probes, nx, ny)
    """
    if not TORCH_AVAILABLE:
        # Fallback for individual processing
        if base_probe is not None:
            results = []
            for px, py in probes_or_positions:
                # Would need to implement NumPy version of shifting
                results.append(Propagate(base_probe, potential))
            return results
        return [Propagate(probe, potential) for probe in probes_or_positions]
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle different input types
    if isinstance(probes_or_positions[0], tuple):
        # List of positions - create batched probes
        if base_probe is None:
            raise ValueError("base_probe required when using probe positions")
        probe_batch = create_batched_probes(base_probe, probes_or_positions, device)
    else:
        # Assume it's already a batched tensor or list of probe objects
        if isinstance(probes_or_positions, torch.Tensor):
            probe_batch = probes_or_positions
        else:
            # List of probe arrays - stack them
            probe_arrays = [torch.tensor(p.array) if hasattr(p, 'array') else p 
                          for p in probes_or_positions]
            probe_batch = torch.stack(probe_arrays, dim=0).to(device)
    
    # Use the vectorized propagation
    return PropagateTorch(probe_batch, potential, device)