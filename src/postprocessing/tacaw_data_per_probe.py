"""
TACAW data that creates WFData objects per probe as needed.
"""
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging
from tqdm import tqdm
from .wf_data import WFData
from .tacaw_data import TACAWData

try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        device = torch.device('cuda')
        complex_dtype = torch.complex128
        float_dtype = torch.float64
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        complex_dtype = torch.complex64  # MPS only supports complex64
        float_dtype = torch.float32
    else:
        device = torch.device('cpu')
        complex_dtype = torch.complex128
        float_dtype = torch.float64
except ImportError:
    TORCH_AVAILABLE = False
    device = None
    complex_dtype = None
    float_dtype = None

logger = logging.getLogger(__name__)

class TACAWDataPerProbe:
    """
    TACAW processor that creates WFData objects per probe as needed.
    
    This avoids the memory explosion by only loading one probe's data at a time.
    """
    
    def __init__(self, probe_positions, time, kxs, kys, layer, probe, cache_dir: Path):
        """
        Initialize with metadata and cache directory.
        
        Args:
            probe_positions: List of (x,y) probe positions
            time: Time array
            kxs, kys: k-space arrays
            layer: Layer array
            probe: Probe object
            cache_dir: Directory containing frame_*.npy files
        """
        self.probe_positions = probe_positions
        self.time = time
        self.kxs = kxs
        self.kys = kys
        self.layer = layer
        self.probe = probe
        self.cache_dir = Path(cache_dir)
        
        # Compute frequencies
        n_freq = len(time)
        dt = time[1] - time[0] if len(time) > 1 else 1.0
        self.frequencies = np.fft.fftfreq(n_freq, d=dt)
        self.frequencies = np.fft.fftshift(self.frequencies)
        
        # Cache for computed TACAW objects
        self._tacaw_cache = {}
        
        logger.info(f"Initialized per-probe TACAW with {len(probe_positions)} probes")
        if TORCH_AVAILABLE:
            logger.info(f"PyTorch acceleration enabled on device: {device}")
        else:
            logger.info("Using NumPy backend (PyTorch not available)")
        
        # Get dimensions from first frame
        test_frame = self._load_frame(0)
        self.n_frames = len(time)
        self.nx, self.ny = test_frame.shape[1], test_frame.shape[2]
        
        # Initialize memory-mapped access for efficient I/O
        self._setup_memmap_access()
    
    def _load_frame(self, frame_idx: int) -> np.ndarray:
        """Load a single frame from cache."""
        cache_file = self.cache_dir / f"frame_{frame_idx}.npy"
        if not cache_file.exists():
            raise FileNotFoundError(f"Frame {frame_idx} not found at {cache_file}")
        return np.load(cache_file)
    
    def _get_probe_wfdata(self, probe_index: int) -> WFData:
        """
        Create WFData object for a single probe by loading only that probe's data using memory mapping.
        
        Args:
            probe_index: Index of probe to load
            
        Returns:
            WFData object containing only this probe's data
        """
        # Load all frames for this probe using memory mapping
        wavefunction_data = np.zeros((1, self.n_frames, self.nx, self.ny, 1), dtype=np.complex128)
        
        for frame_idx in range(self.n_frames):
            frame_memmap = self._get_frame_memmap(frame_idx)  # Memory-mapped access
            wavefunction_data[0, frame_idx, :, :, 0] = frame_memmap[probe_index, :, :, 0, 0]
        
        # Create WFData with just this probe
        return WFData(
            probe_positions=[self.probe_positions[probe_index]],  # Single probe
            time=self.time,
            kxs=self.kxs,
            kys=self.kys,
            layer=self.layer,
            wavefunction_data=wavefunction_data,
            probe=self.probe
        )
    
    def _get_probe_tacaw(self, probe_index: int) -> TACAWData:
        """
        Get TACAWData for a single probe (with caching).
        
        Args:
            probe_index: Index of probe
            
        Returns:
            TACAWData object for this probe
        """
        if probe_index in self._tacaw_cache:
            return self._tacaw_cache[probe_index]
        
        # Create WFData for just this probe
        wf_data = self._get_probe_wfdata(probe_index)
        
        # Convert to TACAW
        tacaw_data = TACAWData(wf_data)
        
        # Cache it
        self._tacaw_cache[probe_index] = tacaw_data
        
        return tacaw_data
    
    def spectrum(self, probe_index: Optional[int] = 0, chunk_size: int = 1) -> np.ndarray:
        """
        Get spectrum for specific probe or average over all probes.
        
        Args:
            probe_index: Index of probe (default: 0), or None to average over all probes
            chunk_size: Number of probes to process simultaneously (default: 1)
                       Higher values use more memory but can be faster
            
        Returns:
            Spectrum array (frequency intensity)
        """
        if probe_index is None:
            return self._compute_average_spectrum(chunk_size)
        else:
            tacaw = self._get_probe_tacaw(probe_index)
            return tacaw.spectrum(probe_index=0)  # Always 0 since single probe
    
    def spectrum_image(self, frequency: float, probe_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Get spectrum image at specific frequency using fast memory-mapped access.
        
        Args:
            frequency: Frequency value in THz
            probe_indices: List of probe indices (default: all probes)
        
        Returns:
            Spectrum intensity for each probe position
        """
        if probe_indices is None:
            probe_indices = list(range(len(self.probe_positions)))
        
        logger.info(f"Computing spectrum image for {len(probe_indices)} probes at {frequency:.2f} THz")
        
        # Find closest frequency index
        freq_idx = np.argmin(np.abs(self.frequencies - frequency))
        
        # Process each probe individually - memory mapping makes this fast
        intensities = []
        for probe_idx in tqdm(probe_indices, desc=f"Computing spectrum image at {frequency:.1f} THz"):
            intensity = self._compute_probe_spectrum_intensity_at_frequency(probe_idx, freq_idx)
            intensities.append(intensity)
        
        return np.array(intensities)
    
    def diffraction(self, probe_index: Optional[int] = 0, chunk_size: int = 1) -> np.ndarray:
        """
        Get diffraction pattern for specific probe or average over all probes.
        
        Args:
            probe_index: Index of probe (default: 0), or None to average over all probes
            chunk_size: Number of probes to process simultaneously (default: 1)
                       Higher values use more memory but can be faster
            
        Returns:
            Diffraction pattern (kx, ky) - intensity averaged over frequencies
        """
        if probe_index is None:
            return self._compute_average_diffraction(chunk_size)
        else:
            tacaw = self._get_probe_tacaw(probe_index)
            return tacaw.diffraction(probe_index=0)
    
    def spectral_diffraction(self, frequency: float, probe_index: int = 0) -> np.ndarray:
        """Get spectral diffraction at specific frequency and probe."""
        tacaw = self._get_probe_tacaw(probe_index)
        return tacaw.spectral_diffraction(frequency, probe_index=0)
    
    def masked_spectrum(self, mask: np.ndarray, probe_index: int = 0) -> np.ndarray:
        """Get masked spectrum for specific probe."""
        tacaw = self._get_probe_tacaw(probe_index)
        return tacaw.masked_spectrum(mask, probe_index=0)
    
    def dispersion(self, kx_path: np.ndarray = None, ky_path: np.ndarray = None, 
                   probe_index: int = 0) -> np.ndarray:
        """Get dispersion for specific probe."""
        tacaw = self._get_probe_tacaw(probe_index)
        return tacaw.dispersion(kx_path, ky_path, probe_index=0)
    
    def precompute_probes(self, probe_indices: List[int]):
        """
        Precompute TACAW data for specific probes.
        
        Args:
            probe_indices: List of probe indices to precompute
        """
        logger.info(f"Precomputing TACAW for {len(probe_indices)} probes")
        
        for i, probe_idx in enumerate(probe_indices):
            if i % 10 == 0:
                logger.info(f"Precomputing probe {i+1}/{len(probe_indices)}")
            self._get_probe_tacaw(probe_idx)
        
        logger.info("Precomputation complete")
    
    def _compute_average_spectrum(self, chunk_size: int = 1) -> np.ndarray:
        """
        Compute average spectrum over all probes using chunked processing.
        
        Args:
            chunk_size: Number of probes to process simultaneously
        
        Returns:
            Average spectrum array
        """
        n_probes = len(self.probe_positions)
        logger.info(f"Computing average spectrum over {n_probes} probes (chunk_size={chunk_size})")
        
        total_spectrum = None
        n_chunks = (n_probes + chunk_size - 1) // chunk_size  # Ceiling division
        
        for chunk_idx in tqdm(range(n_chunks), desc="Computing average spectrum"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_probes)
            chunk_indices = list(range(start_idx, end_idx))
            
            # Process chunk of probes
            chunk_spectrum = self._process_spectrum_chunk(chunk_indices)
            
            # Accumulate results
            if total_spectrum is None:
                total_spectrum = chunk_spectrum
            else:
                total_spectrum += chunk_spectrum
        
        return total_spectrum / n_probes
    
    def _compute_average_diffraction(self, chunk_size: int = 1) -> np.ndarray:
        """
        Compute average diffraction over all probes using chunked processing.
        
        Args:
            chunk_size: Number of probes to process simultaneously
        
        Returns:
            Average diffraction pattern array
        """
        n_probes = len(self.probe_positions)
        logger.info(f"Computing average diffraction over {n_probes} probes (chunk_size={chunk_size})")
        
        total_diffraction = None
        n_chunks = (n_probes + chunk_size - 1) // chunk_size  # Ceiling division
        
        for chunk_idx in tqdm(range(n_chunks), desc="Computing average diffraction"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, n_probes)
            chunk_indices = list(range(start_idx, end_idx))
            
            # Process chunk of probes
            chunk_diffraction = self._process_diffraction_chunk(chunk_indices)
            
            # Accumulate results
            if total_diffraction is None:
                total_diffraction = chunk_diffraction
            else:
                total_diffraction += chunk_diffraction
        
        return total_diffraction / n_probes
    
    def _process_spectrum_chunk(self, probe_indices: List[int]) -> np.ndarray:
        """Process a chunk of probes for spectrum calculation using vectorized operations with memory mapping."""
        # Load all frames for this chunk of probes at once using memory mapping
        n_probes_chunk = len(probe_indices)
        chunk_wf_data = np.zeros((n_probes_chunk, self.n_frames, self.nx, self.ny), dtype=np.complex128)
        
        # Efficient loading: use memory-mapped files for much faster I/O
        for frame_idx in range(self.n_frames):
            frame_memmap = self._get_frame_memmap(frame_idx)  # Memory-mapped access
            for i, probe_idx in enumerate(probe_indices):
                chunk_wf_data[i, frame_idx, :, :] = frame_memmap[probe_idx, :, :, 0, 0]
        
        # Vectorized FFT and spectrum calculation for entire chunk
        return self._compute_chunk_spectrum_vectorized(chunk_wf_data)
    
    def _process_diffraction_chunk(self, probe_indices: List[int]) -> np.ndarray:
        """Process a chunk of probes for diffraction calculation using vectorized operations with memory mapping."""
        # Load all frames for this chunk of probes at once using memory mapping
        n_probes_chunk = len(probe_indices)
        chunk_wf_data = np.zeros((n_probes_chunk, self.n_frames, self.nx, self.ny), dtype=np.complex128)
        
        # Efficient loading: use memory-mapped files for much faster I/O
        for frame_idx in range(self.n_frames):
            frame_memmap = self._get_frame_memmap(frame_idx)  # Memory-mapped access
            for i, probe_idx in enumerate(probe_indices):
                chunk_wf_data[i, frame_idx, :, :] = frame_memmap[probe_idx, :, :, 0, 0]
        
        # Vectorized FFT and diffraction calculation for entire chunk
        return self._compute_chunk_diffraction_vectorized(chunk_wf_data)
    
    def _compute_probe_spectrum_intensity_at_frequency(self, probe_idx: int, freq_idx: int) -> float:
        """
        Efficiently compute spectrum intensity for a single probe at a specific frequency.
        
        Args:
            probe_idx: Index of probe
            freq_idx: Index of frequency
            
        Returns:
            Intensity value at the specified frequency
        """
        # Load single probe data using memory mapping
        probe_wf_data = np.zeros((self.n_frames, self.nx, self.ny), dtype=np.complex128)
        
        for frame_idx in range(self.n_frames):
            frame_memmap = self._get_frame_memmap(frame_idx)
            probe_wf_data[frame_idx, :, :] = frame_memmap[probe_idx, :, :, 0, 0]
        
        if TORCH_AVAILABLE:
            # GPU-accelerated processing
            probe_tensor = torch.from_numpy(probe_wf_data).to(device=device, dtype=complex_dtype)
            
            # Subtract time-average
            wf_mean = torch.mean(probe_tensor, dim=0, keepdim=True)
            probe_tensor = probe_tensor - wf_mean
            
            # FFT along time axis
            probe_fft = torch.fft.fft(probe_tensor, dim=0)
            probe_fft = torch.fft.fftshift(probe_fft, dim=0)
            
            # Compute intensity and extract at target frequency
            intensity = torch.abs(probe_fft)**2
            freq_intensity = intensity[freq_idx, :, :]  # Shape: (nx, ny)
            
            # Sum over k-space
            total_intensity = torch.sum(freq_intensity)
            return total_intensity.cpu().item()
        
        else:
            # CPU fallback
            # Subtract time-average
            wf_mean = np.mean(probe_wf_data, axis=0, keepdims=True)
            probe_wf_data = probe_wf_data - wf_mean
            
            # FFT along time axis
            probe_fft = np.fft.fft(probe_wf_data, axis=0)
            probe_fft = np.fft.fftshift(probe_fft, axes=0)
            
            # Compute intensity and extract at target frequency
            intensity = np.abs(probe_fft)**2
            freq_intensity = intensity[freq_idx, :, :]  # Shape: (nx, ny)
            
            # Sum over k-space
            return np.sum(freq_intensity)
    
    def _process_spectrum_image_chunk(self, probe_indices: List[int], freq_idx: int) -> List[float]:
        """
        Process a chunk of probes for spectrum image calculation at a specific frequency.
        
        Args:
            probe_indices: List of probe indices to process
            freq_idx: Index of the target frequency
            
        Returns:
            List of intensity values for each probe at the target frequency
        """
        # Load all frames for this chunk of probes using memory mapping
        n_probes_chunk = len(probe_indices)
        chunk_wf_data = np.zeros((n_probes_chunk, self.n_frames, self.nx, self.ny), dtype=np.complex128)
        
        # Efficient loading using memory-mapped files
        for frame_idx in range(self.n_frames):
            frame_memmap = self._get_frame_memmap(frame_idx)
            for i, probe_idx in enumerate(probe_indices):
                chunk_wf_data[i, frame_idx, :, :] = frame_memmap[probe_idx, :, :, 0, 0]
        
        # Compute spectrum intensities efficiently
        if TORCH_AVAILABLE:
            # GPU-accelerated processing
            chunk_tensor = torch.from_numpy(chunk_wf_data).to(device=device, dtype=complex_dtype)
            
            # Subtract time-average
            wf_mean = torch.mean(chunk_tensor, dim=1, keepdim=True)
            chunk_tensor = chunk_tensor - wf_mean
            
            # FFT along time axis
            chunk_wf_fft = torch.fft.fft(chunk_tensor, dim=1)
            chunk_wf_fft = torch.fft.fftshift(chunk_wf_fft, dim=1)
            
            # Compute intensity
            chunk_intensity = torch.abs(chunk_wf_fft)**2
            
            # Extract intensity at target frequency and sum over k-space
            freq_intensity = chunk_intensity[:, freq_idx, :, :]  # Shape: (n_probes_chunk, nx, ny)
            probe_intensities = torch.sum(freq_intensity, dim=(1, 2))  # Shape: (n_probes_chunk,)
            
            # Convert to numpy
            return probe_intensities.cpu().numpy().tolist()
        
        else:
            # CPU fallback
            # Subtract time-average
            wf_mean = np.mean(chunk_wf_data, axis=1, keepdims=True)
            chunk_wf_data = chunk_wf_data - wf_mean
            
            # FFT along time axis
            chunk_wf_fft = np.fft.fft(chunk_wf_data, axis=1)
            chunk_wf_fft = np.fft.fftshift(chunk_wf_fft, axes=1)
            
            # Compute intensity
            chunk_intensity = np.abs(chunk_wf_fft)**2
            
            # Extract intensity at target frequency and sum over k-space
            freq_intensity = chunk_intensity[:, freq_idx, :, :]  # Shape: (n_probes_chunk, nx, ny)
            probe_intensities = np.sum(freq_intensity, axis=(1, 2))  # Shape: (n_probes_chunk,)
            
            return probe_intensities.tolist()
    
    def _setup_memmap_access(self):
        """Setup memory-mapped access to frame files for efficient I/O."""
        logger.info("Setting up memory-mapped access to frame files")
        self._frame_memmaps = {}
        
        # Get shape info from first frame
        first_frame_path = self.cache_dir / "frame_0.npy"
        first_frame = np.load(first_frame_path)
        self.frame_shape = first_frame.shape  # (n_probes_total, nx, ny, 1, 1)
        
        # Optional: Pre-load small number of frames into memory maps
        # This is a compromise - we don't load all frames but we avoid repeated file opens
        self.memmap_cache_size = min(50, self.n_frames)  # Cache up to 50 frames as memmaps
        logger.info(f"Will use memory-mapped caching for up to {self.memmap_cache_size} frames")
    
    def _get_frame_memmap(self, frame_idx: int) -> np.ndarray:
        """Get memory-mapped access to a frame file."""
        if frame_idx in self._frame_memmaps:
            return self._frame_memmaps[frame_idx]
        
        # If cache is full, remove oldest entry
        if len(self._frame_memmaps) >= self.memmap_cache_size:
            oldest_frame = min(self._frame_memmaps.keys())
            del self._frame_memmaps[oldest_frame]
        
        # Create new memory map
        frame_path = self.cache_dir / f"frame_{frame_idx}.npy"
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame {frame_idx} not found at {frame_path}")
        
        # Use memory mapping for efficient access
        frame_memmap = np.load(frame_path, mmap_mode='r')
        self._frame_memmaps[frame_idx] = frame_memmap
        
        return frame_memmap
    
    def _compute_chunk_spectrum_vectorized(self, chunk_wf_data: np.ndarray) -> np.ndarray:
        """
        Compute spectrum for a chunk of probes using vectorized operations (PyTorch accelerated).
        
        Args:
            chunk_wf_data: Shape (n_probes_chunk, n_frames, nx, ny)
            
        Returns:
            Summed spectrum across all probes in chunk
        """
        if TORCH_AVAILABLE:
            # Convert to PyTorch tensor with appropriate dtype and move to device
            chunk_tensor = torch.from_numpy(chunk_wf_data).to(device=device, dtype=complex_dtype)
            
            # Subtract time-average to remove elastic scattering 
            wf_mean = torch.mean(chunk_tensor, dim=1, keepdim=True)  # Shape: (n_probes, 1, nx, ny)
            chunk_tensor = chunk_tensor - wf_mean
            
            # Vectorized FFT along time axis for all probes simultaneously (GPU accelerated!)
            chunk_wf_fft = torch.fft.fft(chunk_tensor, dim=1)  # FFT along time axis
            chunk_wf_fft = torch.fft.fftshift(chunk_wf_fft, dim=1)
            
            # Compute intensity |Ψ(ω,q)|² for all probes
            chunk_intensity = torch.abs(chunk_wf_fft)**2  # Shape: (n_probes_chunk, n_freq, nx, ny)
            
            # Sum over k-space for each probe, then sum across probes  
            chunk_spectra = torch.sum(chunk_intensity, dim=(2, 3))  # Shape: (n_probes_chunk, n_freq)
            total_spectrum = torch.sum(chunk_spectra, dim=0)  # Shape: (n_freq,)
            
            # Convert back to numpy
            return total_spectrum.cpu().numpy()
        
        else:
            # Fallback to NumPy
            # Subtract time-average to remove elastic scattering 
            wf_mean = np.mean(chunk_wf_data, axis=1, keepdims=True)  # Shape: (n_probes, 1, nx, ny)
            chunk_wf_data = chunk_wf_data - wf_mean
            
            # Vectorized FFT along time axis for all probes simultaneously
            chunk_wf_fft = np.fft.fft(chunk_wf_data, axis=1)  # FFT along time axis
            chunk_wf_fft = np.fft.fftshift(chunk_wf_fft, axes=1)
            
            # Compute intensity |Ψ(ω,q)|² for all probes
            chunk_intensity = np.abs(chunk_wf_fft)**2  # Shape: (n_probes_chunk, n_freq, nx, ny)
            
            # Sum over k-space for each probe, then sum across probes  
            chunk_spectra = np.sum(chunk_intensity, axis=(2, 3))  # Shape: (n_probes_chunk, n_freq)
            total_spectrum = np.sum(chunk_spectra, axis=0)  # Shape: (n_freq,)
            
            return total_spectrum
    
    def _compute_chunk_diffraction_vectorized(self, chunk_wf_data: np.ndarray) -> np.ndarray:
        """
        Compute diffraction for a chunk of probes using vectorized operations (PyTorch accelerated).
        
        Args:
            chunk_wf_data: Shape (n_probes_chunk, n_frames, nx, ny)
            
        Returns:
            Summed diffraction pattern across all probes in chunk
        """
        if TORCH_AVAILABLE:
            # Convert to PyTorch tensor with appropriate dtype and move to device
            chunk_tensor = torch.from_numpy(chunk_wf_data).to(device=device, dtype=complex_dtype)
            
            # Subtract time-average to remove elastic scattering
            wf_mean = torch.mean(chunk_tensor, dim=1, keepdim=True)  # Shape: (n_probes, 1, nx, ny)
            chunk_tensor = chunk_tensor - wf_mean
            
            # Vectorized FFT along time axis for all probes simultaneously (GPU accelerated!)
            chunk_wf_fft = torch.fft.fft(chunk_tensor, dim=1)  # FFT along time axis
            chunk_wf_fft = torch.fft.fftshift(chunk_wf_fft, dim=1)
            
            # Compute intensity |Ψ(ω,q)|² for all probes
            chunk_intensity = torch.abs(chunk_wf_fft)**2  # Shape: (n_probes_chunk, n_freq, nx, ny)
            
            # Sum over frequencies for each probe, then sum across probes
            chunk_diffractions = torch.sum(chunk_intensity, dim=1)  # Shape: (n_probes_chunk, nx, ny)
            total_diffraction = torch.sum(chunk_diffractions, dim=0)  # Shape: (nx, ny)
            
            # Convert back to numpy
            return total_diffraction.cpu().numpy()
        
        else:
            # Fallback to NumPy
            # Subtract time-average to remove elastic scattering
            wf_mean = np.mean(chunk_wf_data, axis=1, keepdims=True)  # Shape: (n_probes, 1, nx, ny)
            chunk_wf_data = chunk_wf_data - wf_mean
            
            # Vectorized FFT along time axis for all probes simultaneously
            chunk_wf_fft = np.fft.fft(chunk_wf_data, axis=1)  # FFT along time axis
            chunk_wf_fft = np.fft.fftshift(chunk_wf_fft, axes=1)
            
            # Compute intensity |Ψ(ω,q)|² for all probes
            chunk_intensity = np.abs(chunk_wf_fft)**2  # Shape: (n_probes_chunk, n_freq, nx, ny)
            
            # Sum over frequencies for each probe, then sum across probes
            chunk_diffractions = np.sum(chunk_intensity, axis=1)  # Shape: (n_probes_chunk, nx, ny)
            total_diffraction = np.sum(chunk_diffractions, axis=0)  # Shape: (nx, ny)
            
            return total_diffraction
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get info about cached probes."""
        return {
            'total_probes': len(self.probe_positions),
            'cached_probes': len(self._tacaw_cache),
            'cached_indices': list(self._tacaw_cache.keys())
        }