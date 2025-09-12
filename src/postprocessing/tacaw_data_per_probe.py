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
            # Use cached FFT if available
            if hasattr(self, '_probe_fft_cache') and probe_index in self._probe_fft_cache:
                fft_data = self._probe_fft_cache[probe_index]
                intensity = np.abs(fft_data)**2
                spectrum = np.sum(intensity, axis=(1, 2))  # Sum over k-space
                return spectrum
            else:
                tacaw = self._get_probe_tacaw(probe_index)
                return tacaw.spectrum(probe_index=0)  # Always 0 since single probe
    
    def spectrum_image(self, frequency: float, probe_indices: Optional[List[int]] = None, 
                      chunk_size: Optional[int] = None) -> np.ndarray:
        """
        Get spectrum image at specific frequency using optimized batch processing.
        
        Args:
            frequency: Frequency value in THz
            probe_indices: List of probe indices (default: all probes)
            chunk_size: Number of probes to process simultaneously (default: 100)
        
        Returns:
            Spectrum intensity for each probe position
        """
        if probe_indices is None:
            probe_indices = list(range(len(self.probe_positions)))
        
        if chunk_size is None:
            chunk_size = 100  # Default chunk size
        
        logger.info(f"Computing spectrum image for {len(probe_indices)} probes at {frequency:.2f} THz (chunk_size={chunk_size})")
        
        # Find closest frequency index
        freq_idx = np.argmin(np.abs(self.frequencies - frequency))
        
        # Check if we have cached intensity for this frequency
        cache_key = f"freq_{freq_idx}_intensities"
        if hasattr(self, '_freq_intensity_cache') and cache_key in self._freq_intensity_cache:
            logger.info(f"Using cached intensity data for frequency index {freq_idx}")
            cached_intensities = self._freq_intensity_cache[cache_key]
            return cached_intensities[probe_indices]
        
        # Process in chunks for efficiency
        n_chunks = (len(probe_indices) + chunk_size - 1) // chunk_size
        intensities = np.zeros(len(probe_indices), dtype=np.float64)
        
        for chunk_idx in tqdm(range(n_chunks), desc=f"Computing spectrum image at {frequency:.1f} THz"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(probe_indices))
            chunk_probe_indices = probe_indices[start_idx:end_idx]
            
            # Process chunk using vectorized operations
            chunk_intensities = self._process_spectrum_image_chunk(chunk_probe_indices, freq_idx)
            intensities[start_idx:end_idx] = chunk_intensities
        
        # Cache if processing all probes
        if len(probe_indices) == len(self.probe_positions):
            if not hasattr(self, '_freq_intensity_cache'):
                self._freq_intensity_cache = {}
            self._freq_intensity_cache[cache_key] = intensities.copy()
        
        return intensities
    
    def diffraction(self, probe_index: Optional[int] = 0, chunk_size: int = 1) -> np.ndarray:
        """
        Get diffraction pattern for specific probe or average over all probes.
        Note: This sums the raw wavefunction intensity over time (no FFT needed).
        
        Args:
            probe_index: Index of probe (default: 0), or None to average over all probes
            chunk_size: Number of probes to process simultaneously (default: 1)
                       Higher values use more memory but can be faster
            
        Returns:
            Diffraction pattern (kx, ky) - intensity summed over time
        """
        if probe_index is None:
            return self._compute_average_diffraction(chunk_size)
        else:
            # Load raw wavefunction data for this probe (no FFT needed)
            probe_wf_data = np.zeros((self.n_frames, self.nx, self.ny), dtype=np.complex128)
            
            for frame_idx in range(self.n_frames):
                frame_memmap = self._get_frame_memmap(frame_idx)
                probe_wf_data[frame_idx, :, :] = frame_memmap[probe_index, :, :, 0, 0]
            
            # Subtract time-average to remove elastic scattering
            wf_mean = np.mean(probe_wf_data, axis=0, keepdims=True)
            probe_wf_data = probe_wf_data - wf_mean
            
            # Compute intensity and sum over time
            intensity = np.abs(probe_wf_data)**2
            diffraction = np.sum(intensity, axis=0)  # Sum over time axis
            
            return diffraction
    
    def spectral_diffraction(self, frequency: float, probe_index: int = 0) -> np.ndarray:
        """Get spectral diffraction at specific frequency and probe."""
        tacaw = self._get_probe_tacaw(probe_index)
        return tacaw.spectral_diffraction(frequency, probe_index=0)
    
    def masked_spectrum(self, mask: np.ndarray, probe_index: int = 0) -> np.ndarray:
        """Get masked spectrum for specific probe."""
        tacaw = self._get_probe_tacaw(probe_index)
        return tacaw.masked_spectrum(mask, probe_index=0)
    
    def dispersion(self, kx_path: np.ndarray = None, ky_path: np.ndarray = None) -> np.ndarray:
        """
        Extract dispersion relation from actual TACAW intensity data.
        
        Args:
            kx_path: kx values for dispersion calculation (optional)
            ky_path: ky values for dispersion calculation (optional)
            
        Returns:
            Dispersion relation array with shape (n_frequencies, n_kx)
            Real intensity data from TACAW simulation
        """
        # Use provided kx/ky paths, or fall back to our own kxs/kys
        if kx_path is None:
            kx_array = self.kxs
            kx_indices = np.arange(len(self.kxs))
        else:
            # Find closest indices in our kxs array for the requested kx_path
            kx_indices = []
            for kx_val in kx_path:
                idx = np.argmin(np.abs(self.kxs - kx_val))
                kx_indices.append(idx)
            kx_indices = np.array(kx_indices)
            
        if ky_path is None:
            ky_array = self.kys
            ky_indices = np.arange(len(self.kys))
        else:
            # Find closest indices in our kys array for the requested ky_path
            ky_indices = []
            for ky_val in ky_path:
                idx = np.argmin(np.abs(self.kys - ky_val))
                ky_indices.append(idx)
            ky_indices = np.array(ky_indices)
        
        # Extract dispersion from probe 0 intensity data
        # Shape will be (n_frequencies, n_kx)
        n_freq = len(self.frequencies)
        n_kx = len(kx_indices)
        
        dispersion = np.zeros((n_freq, n_kx))
        
        # Get intensity data for probe 0
        for i, kx_idx in enumerate(kx_indices):
            # Use middle ky index if no specific ky_path given, otherwise use first ky_idx
            ky_idx = len(self.kys) // 2 if ky_path is None else ky_indices[min(i, len(ky_indices)-1)]
            
            # Extract intensity vs frequency for this k-point
            for j in range(n_freq):
                dispersion[j, i] = self.intensity[0, j, kx_idx, ky_idx]
        
        return dispersion
    
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
        
        # OPTIMIZED: Vectorized loading - load frame once, extract all probes at once
        for frame_idx in range(self.n_frames):
            frame_memmap = self._get_frame_memmap(frame_idx)  # Memory-mapped access
            # Vectorized indexing: extract all needed probes in one operation
            chunk_wf_data[:, frame_idx, :, :] = frame_memmap[probe_indices, :, :, 0, 0]
        
        # Vectorized FFT and spectrum calculation for entire chunk
        return self._compute_chunk_spectrum_vectorized(chunk_wf_data)
    
    def _process_diffraction_chunk(self, probe_indices: List[int]) -> np.ndarray:
        """Process a chunk of probes for diffraction calculation using vectorized operations with memory mapping."""
        # Load all frames for this chunk of probes at once using memory mapping
        n_probes_chunk = len(probe_indices)
        chunk_wf_data = np.zeros((n_probes_chunk, self.n_frames, self.nx, self.ny), dtype=np.complex128)
        
        # OPTIMIZED: Vectorized loading - load frame once, extract all probes at once
        for frame_idx in range(self.n_frames):
            frame_memmap = self._get_frame_memmap(frame_idx)  # Memory-mapped access
            # Vectorized indexing: extract all needed probes in one operation
            chunk_wf_data[:, frame_idx, :, :] = frame_memmap[probe_indices, :, :, 0, 0]
        
        # Subtract time-average to remove elastic scattering
        wf_mean = np.mean(chunk_wf_data, axis=1, keepdims=True)  # Shape: (n_probes, 1, nx, ny)
        chunk_wf_data = chunk_wf_data - wf_mean
        
        # Compute intensity and sum over time (no FFT needed for diffraction)
        chunk_intensity = np.abs(chunk_wf_data)**2
        # Sum over time axis for each probe, then sum across probes
        chunk_diffractions = np.sum(chunk_intensity, axis=1)  # Shape: (n_probes_chunk, nx, ny)
        total_diffraction = np.sum(chunk_diffractions, axis=0)  # Shape: (nx, ny)
        
        return total_diffraction
    
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
    
    def _process_spectrum_image_chunk(self, probe_indices: List[int], freq_idx: int) -> np.ndarray:
        """
        Process a chunk of probes for spectrum image calculation at a specific frequency.
        Optimized for large chunks with better memory management.
        
        Args:
            probe_indices: List of probe indices to process
            freq_idx: Index of the target frequency
            
        Returns:
            Array of intensity values for each probe at the target frequency
        """
        n_probes_chunk = len(probe_indices)
        
        # Check if we have pre-computed FFT cache for these probes
        if hasattr(self, '_probe_fft_cache'):
            cached_intensities = []
            uncached_indices = []
            uncached_positions = []
            
            for i, probe_idx in enumerate(probe_indices):
                if probe_idx in self._probe_fft_cache:
                    # Use cached FFT data
                    fft_data = self._probe_fft_cache[probe_idx]
                    intensity = np.abs(fft_data[freq_idx])**2
                    cached_intensities.append((i, np.sum(intensity)))
                else:
                    uncached_indices.append(probe_idx)
                    uncached_positions.append(i)
            
            # If all are cached, return immediately
            if not uncached_indices:
                result = np.zeros(n_probes_chunk)
                for pos, intensity in cached_intensities:
                    result[pos] = intensity
                return result
            
            # Process only uncached probes
            probe_indices = uncached_indices
            n_probes_to_process = len(uncached_indices)
        else:
            self._probe_fft_cache = {}  # Initialize cache if it doesn't exist
            n_probes_to_process = n_probes_chunk
            cached_intensities = []
            uncached_positions = list(range(n_probes_chunk))
        
        # Load all frames for uncached probes using memory mapping
        chunk_wf_data = np.zeros((n_probes_to_process, self.n_frames, self.nx, self.ny), dtype=np.complex128)
        
        # OPTIMIZED: Vectorized loading - load frame once, extract all probes at once
        for frame_idx in range(self.n_frames):
            frame_memmap = self._get_frame_memmap(frame_idx)
            # Vectorized indexing: extract all needed probes in one operation
            chunk_wf_data[:, frame_idx, :, :] = frame_memmap[probe_indices, :, :, 0, 0]
        
        # Compute spectrum intensities efficiently
        if TORCH_AVAILABLE:
            # GPU-accelerated processing with memory optimization
            # Process in sub-chunks if needed to avoid GPU memory issues
            max_gpu_chunk = 200  # Process at most 200 probes at once on GPU
            
            if n_probes_to_process <= max_gpu_chunk:
                # Process all at once
                chunk_tensor = torch.from_numpy(chunk_wf_data).to(device=device, dtype=complex_dtype)
                
                # Subtract time-average
                wf_mean = torch.mean(chunk_tensor, dim=1, keepdim=True)
                chunk_tensor = chunk_tensor - wf_mean
                
                # FFT along time axis
                chunk_wf_fft = torch.fft.fft(chunk_tensor, dim=1)
                chunk_wf_fft = torch.fft.fftshift(chunk_wf_fft, dim=1)
                
                # Cache the FFT data for future use
                chunk_fft_np = chunk_wf_fft.cpu().numpy()
                for i, probe_idx in enumerate(probe_indices):
                    self._probe_fft_cache[probe_idx] = chunk_fft_np[i]
                
                # Compute intensity
                chunk_intensity = torch.abs(chunk_wf_fft)**2
                
                # Extract intensity at target frequency and sum over k-space
                freq_intensity = chunk_intensity[:, freq_idx, :, :]
                probe_intensities = torch.sum(freq_intensity, dim=(1, 2))
                
                # Convert to numpy
                probe_intensities_np = probe_intensities.cpu().numpy()
            else:
                # Process in smaller GPU chunks
                probe_intensities_np = np.zeros(n_probes_to_process)
                for gpu_start in range(0, n_probes_to_process, max_gpu_chunk):
                    gpu_end = min(gpu_start + max_gpu_chunk, n_probes_to_process)
                    
                    sub_chunk_tensor = torch.from_numpy(chunk_wf_data[gpu_start:gpu_end]).to(device=device, dtype=complex_dtype)
                    
                    # Subtract time-average
                    wf_mean = torch.mean(sub_chunk_tensor, dim=1, keepdim=True)
                    sub_chunk_tensor = sub_chunk_tensor - wf_mean
                    
                    # FFT along time axis
                    sub_chunk_fft = torch.fft.fft(sub_chunk_tensor, dim=1)
                    sub_chunk_fft = torch.fft.fftshift(sub_chunk_fft, dim=1)
                    
                    # Cache the FFT data for future use
                    sub_chunk_fft_np = sub_chunk_fft.cpu().numpy()
                    for i in range(gpu_end - gpu_start):
                        probe_idx = probe_indices[gpu_start + i]
                        self._probe_fft_cache[probe_idx] = sub_chunk_fft_np[i]
                    
                    # Compute intensity
                    sub_chunk_intensity = torch.abs(sub_chunk_fft)**2
                    
                    # Extract intensity at target frequency and sum over k-space
                    freq_intensity = sub_chunk_intensity[:, freq_idx, :, :]
                    sub_intensities = torch.sum(freq_intensity, dim=(1, 2))
                    
                    probe_intensities_np[gpu_start:gpu_end] = sub_intensities.cpu().numpy()
                    
                    # Clear GPU memory
                    del sub_chunk_tensor, sub_chunk_fft, sub_chunk_intensity
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
        else:
            # CPU fallback with optimized NumPy operations
            # Subtract time-average
            wf_mean = np.mean(chunk_wf_data, axis=1, keepdims=True)
            chunk_wf_data = chunk_wf_data - wf_mean
            
            # FFT along time axis
            chunk_wf_fft = np.fft.fft(chunk_wf_data, axis=1)
            chunk_wf_fft = np.fft.fftshift(chunk_wf_fft, axes=1)
            
            # Cache the FFT data for future use
            for i, probe_idx in enumerate(probe_indices):
                self._probe_fft_cache[probe_idx] = chunk_wf_fft[i]
            
            # Compute intensity
            chunk_intensity = np.abs(chunk_wf_fft)**2
            
            # Extract intensity at target frequency and sum over k-space
            freq_intensity = chunk_intensity[:, freq_idx, :, :]
            probe_intensities_np = np.sum(freq_intensity, axis=(1, 2))
        
        # Combine cached and newly computed results
        result = np.zeros(n_probes_chunk)
        
        # Fill in cached values
        for pos, intensity in cached_intensities:
            result[pos] = intensity
        
        # Fill in newly computed values
        for i, pos in enumerate(uncached_positions):
            result[pos] = probe_intensities_np[i]
        
        return result
    
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
    
    
    def precompute_all_probe_ffts(self, chunk_size: int = 200):
        """
        Pre-compute FFTs for all probes to enable fast spectrum image generation.
        This is a one-time cost that dramatically speeds up subsequent operations.
        
        Args:
            chunk_size: Number of probes to process at once
        """
        if not hasattr(self, '_probe_fft_cache'):
            self._probe_fft_cache = {}
        
        logger.info(f"Pre-computing FFTs for {len(self.probe_positions)} probes (chunk_size={chunk_size})")
        
        n_chunks = (len(self.probe_positions) + chunk_size - 1) // chunk_size
        
        for chunk_idx in tqdm(range(n_chunks), desc="Pre-computing FFTs"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(self.probe_positions))
            probe_indices = list(range(start_idx, end_idx))
            
            # Skip already cached probes
            uncached_indices = [idx for idx in probe_indices if idx not in self._probe_fft_cache]
            if not uncached_indices:
                continue
            
            # Load wavefunction data for uncached probes
            n_probes_chunk = len(uncached_indices)
            chunk_wf_data = np.zeros((n_probes_chunk, self.n_frames, self.nx, self.ny), dtype=np.complex128)
            
            for frame_idx in range(self.n_frames):
                frame_memmap = self._get_frame_memmap(frame_idx)
                chunk_wf_data[:, frame_idx, :, :] = frame_memmap[uncached_indices, :, :, 0, 0]
            
            if TORCH_AVAILABLE:
                # GPU-accelerated FFT computation
                # Process in sub-chunks if needed for GPU memory
                max_gpu_chunk = 100
                
                for gpu_start in range(0, n_probes_chunk, max_gpu_chunk):
                    gpu_end = min(gpu_start + max_gpu_chunk, n_probes_chunk)
                    sub_chunk = chunk_wf_data[gpu_start:gpu_end]
                    
                    sub_chunk_tensor = torch.from_numpy(sub_chunk).to(device=device, dtype=complex_dtype)
                    
                    # Subtract time-average
                    wf_mean = torch.mean(sub_chunk_tensor, dim=1, keepdim=True)
                    sub_chunk_tensor = sub_chunk_tensor - wf_mean
                    
                    # FFT along time axis
                    sub_chunk_fft = torch.fft.fft(sub_chunk_tensor, dim=1)
                    sub_chunk_fft = torch.fft.fftshift(sub_chunk_fft, dim=1)
                    
                    # Store in cache
                    sub_chunk_fft_np = sub_chunk_fft.cpu().numpy()
                    for i in range(gpu_end - gpu_start):
                        probe_idx = uncached_indices[gpu_start + i]
                        self._probe_fft_cache[probe_idx] = sub_chunk_fft_np[i]
                    
                    # Clear GPU memory
                    del sub_chunk_tensor, sub_chunk_fft
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
            else:
                # CPU fallback
                # Subtract time-average
                wf_mean = np.mean(chunk_wf_data, axis=1, keepdims=True)
                chunk_wf_data = chunk_wf_data - wf_mean
                
                # FFT along time axis
                chunk_fft = np.fft.fft(chunk_wf_data, axis=1)
                chunk_fft = np.fft.fftshift(chunk_fft, axes=1)
                
                # Store in cache
                for i, probe_idx in enumerate(uncached_indices):
                    self._probe_fft_cache[probe_idx] = chunk_fft[i]
        
        logger.info(f"FFT pre-computation complete. Cached {len(self._probe_fft_cache)} probes")
    
    def spectrum_image_fast(self, frequency: float) -> np.ndarray:
        """
        Ultra-fast spectrum image using pre-computed FFT cache.
        Call precompute_all_probe_ffts() first for best performance.
        
        Args:
            frequency: Frequency value in THz
            
        Returns:
            Spectrum intensity for all probe positions
        """
        # Ensure FFTs are pre-computed
        if not hasattr(self, '_probe_fft_cache') or len(self._probe_fft_cache) < len(self.probe_positions):
            logger.info("FFT cache incomplete. Pre-computing now...")
            self.precompute_all_probe_ffts()
        
        # Find closest frequency index
        freq_idx = np.argmin(np.abs(self.frequencies - frequency))
        
        # Extract intensities from cached FFT data
        intensities = np.zeros(len(self.probe_positions), dtype=np.float64)
        
        for probe_idx in range(len(self.probe_positions)):
            if probe_idx in self._probe_fft_cache:
                fft_data = self._probe_fft_cache[probe_idx]
                intensity = np.abs(fft_data[freq_idx])**2
                intensities[probe_idx] = np.sum(intensity)
            else:
                # Fallback to computing on-demand
                intensities[probe_idx] = self._compute_probe_spectrum_intensity_at_frequency(probe_idx, freq_idx)
        
        return intensities
    
    def clear_fft_cache(self):
        """Clear the FFT cache to free memory."""
        if hasattr(self, '_probe_fft_cache'):
            self._probe_fft_cache.clear()
            logger.info("Cleared FFT cache")
        if hasattr(self, '_freq_intensity_cache'):
            self._freq_intensity_cache.clear()
            logger.info("Cleared frequency intensity cache")
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get info about cached probes."""
        status = {
            'total_probes': len(self.probe_positions),
            'tacaw_cached_probes': len(self._tacaw_cache),
            'tacaw_cached_indices': list(self._tacaw_cache.keys())
        }
        
        if hasattr(self, '_probe_fft_cache'):
            status['fft_cached_probes'] = len(self._probe_fft_cache)
            status['fft_cache_memory_mb'] = sum(
                fft.nbytes / 1e6 for fft in self._probe_fft_cache.values()
            )
        
        if hasattr(self, '_freq_intensity_cache'):
            status['freq_intensity_cached'] = len(self._freq_intensity_cache)
        
        return status

    @property
    def intensity(self):
        """
        Provide array-like access to intensity data for backward compatibility.
        
        Returns a lazy-loaded intensity array that computes FFTs on-demand.
        Expected interface: intensity[probe_index, frequency_index, kx, ky]
        """
        if not hasattr(self, '_intensity_array'):
            self._intensity_array = LazyIntensityArray(self)
        return self._intensity_array


class LazyIntensityArray:
    """
    Lazy-loaded array that provides numpy-like indexing for intensity data.
    Computes FFTs on-demand and caches results for efficiency.
    """
    
    def __init__(self, tacaw_per_probe):
        self.tacaw = tacaw_per_probe
        self._cache = {}  # Cache for computed intensity data
        
    def __getitem__(self, key):
        """
        Handle array indexing: intensity[probe_idx, freq_idx, kx, ky]
        
        Args:
            key: Can be int, slice, or tuple of indices
            
        Returns:
            Intensity data at the specified indices
        """
        if isinstance(key, tuple):
            if len(key) == 4:
                probe_idx, freq_idx, kx_idx, ky_idx = key
                
                # Handle slice notation (e.g., intensity[0, freq_idx, :, :])
                if kx_idx == slice(None) and ky_idx == slice(None):
                    return self._get_frequency_slice(probe_idx, freq_idx)
                else:
                    return self._get_single_element(probe_idx, freq_idx, kx_idx, ky_idx)
            
            elif len(key) == 3:
                probe_idx, freq_idx, spatial_idx = key
                if spatial_idx == slice(None):
                    # Handle intensity[probe_idx, freq_idx, :]
                    return self._get_frequency_slice(probe_idx, freq_idx).flatten()
                else:
                    raise NotImplementedError("3D indexing not fully implemented")
                    
            elif len(key) == 2:
                probe_idx, freq_idx = key
                return self._get_frequency_slice(probe_idx, freq_idx)
                
        elif isinstance(key, int):
            # Return all data for this probe
            return self._get_probe_intensity(key)
        
        else:
            raise NotImplementedError(f"Indexing pattern {key} not supported")
    
    def _get_frequency_slice(self, probe_idx: int, freq_idx: int) -> np.ndarray:
        """Get intensity[probe_idx, freq_idx, :, :] - 2D k-space slice."""
        cache_key = (probe_idx, freq_idx)
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Compute FFT for this probe if not cached
        if not hasattr(self.tacaw, '_probe_fft_cache') or probe_idx not in self.tacaw._probe_fft_cache:
            self._compute_probe_fft(probe_idx)
        
        # Extract intensity at this frequency
        fft_data = self.tacaw._probe_fft_cache[probe_idx]
        intensity_2d = np.abs(fft_data[freq_idx])**2  # Shape: (nx, ny)
        
        # Cache the result
        self._cache[cache_key] = intensity_2d
        
        return intensity_2d
    
    def _get_single_element(self, probe_idx: int, freq_idx: int, kx_idx: int, ky_idx: int) -> float:
        """Get intensity[probe_idx, freq_idx, kx_idx, ky_idx] - single element."""
        freq_slice = self._get_frequency_slice(probe_idx, freq_idx)
        return freq_slice[kx_idx, ky_idx]
    
    def _get_probe_intensity(self, probe_idx: int) -> np.ndarray:
        """Get all intensity data for a probe: shape (n_freq, nx, ny)."""
        cache_key = f"probe_{probe_idx}_all"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Compute FFT for this probe if not cached
        if not hasattr(self.tacaw, '_probe_fft_cache') or probe_idx not in self.tacaw._probe_fft_cache:
            self._compute_probe_fft(probe_idx)
        
        # Extract all frequency data
        fft_data = self.tacaw._probe_fft_cache[probe_idx]  # Shape: (n_freq, nx, ny)
        intensity_3d = np.abs(fft_data)**2
        
        # Cache the result
        self._cache[cache_key] = intensity_3d
        
        return intensity_3d
    
    def _compute_probe_fft(self, probe_idx: int):
        """Compute FFT for a single probe and cache it."""
        if not hasattr(self.tacaw, '_probe_fft_cache'):
            self.tacaw._probe_fft_cache = {}
        
        # Load wavefunction data for this probe
        probe_wf_data = np.zeros((self.tacaw.n_frames, self.tacaw.nx, self.tacaw.ny), dtype=np.complex128)
        
        for frame_idx in range(self.tacaw.n_frames):
            frame_memmap = self.tacaw._get_frame_memmap(frame_idx)
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
            
            # Store in cache
            self.tacaw._probe_fft_cache[probe_idx] = probe_fft.cpu().numpy()
        else:
            # CPU fallback
            # Subtract time-average
            wf_mean = np.mean(probe_wf_data, axis=0, keepdims=True)
            probe_wf_data = probe_wf_data - wf_mean
            
            # FFT along time axis
            probe_fft = np.fft.fft(probe_wf_data, axis=0)
            probe_fft = np.fft.fftshift(probe_fft, axes=0)
            
            # Store in cache
            self.tacaw._probe_fft_cache[probe_idx] = probe_fft
    
    @property
    def shape(self):
        """Return the shape of the full intensity array."""
        return (len(self.tacaw.probe_positions), len(self.tacaw.frequencies), self.tacaw.nx, self.tacaw.ny)