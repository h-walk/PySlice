"""
Core data structure for TACAW EELS calculations.
"""
from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import logging
from .wf_data import WFData

logger = logging.getLogger(__name__)

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

@dataclass
class TACAWData(WFData):
    # inherit all attributes from parent object
    def __init__(self, WFData, layer_index: int = None) -> object:
        self.__class__ = type(WFData.__class__.__name__,
                              (self.__class__, WFData.__class__),
                              {})
        self.__dict__ = WFData.__dict__
        self.fft_from_wf_data(layer_index)

    """
    Data structure for storing TACAW EELS results with format: probe_positions, frequency, kx, ky.

    Attributes:
        probe_positions: List of (x,y) probe positions in Angstroms.
        frequency: Frequencies in THz.
        kx: kx sampling vectors (e.g., in Å⁻¹).
        ky: ky sampling vectors (e.g., in Å⁻¹).
        intensity: Intensity array |Ψ(ω,q)|² (probe_positions, frequency, kx, ky).
    """
    probe_positions: List[Tuple[float, float]]
    frequency: np.ndarray  # frequencies in THz
    kx: np.ndarray  # kx sampling vectors
    ky: np.ndarray  # ky sampling vectors
    intensity: np.ndarray  # Intensity array |Ψ(ω,q)|² (probe_positions, frequency, kx, ky)

    def fft_from_wf_data(self, layer_index: int = None):
        """
        Perform FFT along the time axis for a specific layer to convert to TACAW data.
        This implements the JACR method: Ψ(t,q,r) → |Ψ(ω,q,r)|² via FFT.

        Args:
            layer_index: Index of the layer to compute FFT for (default: last layer)

        Returns:
            TACAWData object with intensity data |Ψ(ω,q)|² for the specified layer
        """

        # Default to last layer if not specified
        if layer_index is None:
            layer_index = len(self.layer) - 1

        # Validate layer index
        if layer_index < 0 or layer_index >= len(self.layer):
            raise ValueError(f"layer_index {layer_index} out of range [0, {len(self.layer)-1}]")

        # Compute frequencies from time sampling
        n_freq = len(self.time)
        dt = self.time[1] - self.time[0] 
        self.frequencies = np.fft.fftfreq(n_freq, d=dt)
        self.frequencies = np.fft.fftshift(self.frequencies)

        # Extract wavefunction data for the specified layer
        # Shape: (probe_positions, time, kx, ky, layer)
        wf_layer = self.wavefunction_data[:, :, :, :, layer_index]
        
        # Perform FFT along time axis (axis=1) for each probe position and k-point
        # Following abeels.py approach: subtract mean to avoid high zero-frequency peak
        if TORCH_AVAILABLE and hasattr(wf_layer, 'dim'):  # Check if it's a torch tensor
            wf_mean = torch.mean(wf_layer, dim=1, keepdim=True)
            wf_fft = torch.fft.fft(wf_layer - wf_mean, dim=1)
            wf_fft = torch.fft.fftshift(wf_fft, dim=1)
        else:
            wf_mean = np.mean(wf_layer, axis=1, keepdims=True)
            wf_fft = np.fft.fft(wf_layer - wf_mean, axis=1)
            wf_fft = np.fft.fftshift(wf_fft, axes=1)
        
        # Compute intensity |Ψ(ω,q)|² from the frequency-domain wavefunction
        if TORCH_AVAILABLE and hasattr(wf_fft, 'dim'):  # Check if it's a torch tensor
            self.intensity = torch.abs(wf_fft)**2
        else:
            self.intensity = np.abs(wf_fft)**2


    def spectrum(self, probe_index: int = None) -> np.ndarray:
        """
        Extract spectrum for a specific probe position by summing over all k-space.

        Args:
            probe_index: Index of probe position (default: 0). If None, averages over all probes.

        Returns:
            Spectrum array (frequency intensity)
        """
        if probe_index is None:
            # Average over all probe positions
            all_spectra = []
            for i in range(len(self.probe_positions)):
                probe_intensity = self.intensity[i]  # Shape: (frequency, kx, ky)
                spectrum = xp.sum(probe_intensity, axis=(1, 2))  # Sum over kx, ky
                all_spectra.append(spectrum)
            
            # Average all spectra
            if TORCH_AVAILABLE and hasattr(all_spectra[0], 'cpu'):
                all_spectra = [s.cpu().numpy() for s in all_spectra]
            spectrum = np.mean(all_spectra, axis=0)
        else:
            if probe_index >= len(self.probe_positions):
                raise ValueError(f"Probe index {probe_index} out of range")

            # Sum intensity data over all k-space for this probe position
            probe_intensity = self.intensity[probe_index]  # Shape: (frequency, kx, ky)
            spectrum = xp.sum(probe_intensity, axis=(1, 2))  # Sum over kx, ky
            
            # Convert to numpy if PyTorch tensor
            if TORCH_AVAILABLE and hasattr(spectrum, 'cpu'):
                spectrum = spectrum.cpu().numpy()
        
        return spectrum

    def spectrum_image(self, frequency: float, probe_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Extract spectrum image at a specific frequency showing intensity in real space (probe positions).
        
        Args:
            frequency: Frequency value in THz
            probe_indices: List of probe indices to include (default: all probes)
            
        Returns:
            Spectrum intensity for each probe position (real space map)
        """
        # Find closest frequency index
        freq_idx = np.argmin(np.abs(self.frequencies - frequency))

        # Use all probes if none specified
        if probe_indices is None:
            probe_indices = list(range(len(self.probe_positions)))

        # Extract intensity at this frequency for each selected probe position
        spectrum_intensities = []
        for probe_idx in probe_indices:
            # Sum intensity data over all k-space for this probe at this frequency
            probe_intensity = self.intensity[probe_idx, freq_idx, :, :]
            
            # Sum over k-space using appropriate method
            if TORCH_AVAILABLE and hasattr(probe_intensity, 'sum'):
                probe_intensity_sum = probe_intensity.sum()
                if hasattr(probe_intensity_sum, 'cpu'):
                    probe_intensity_sum = probe_intensity_sum.cpu().numpy()
            else:
                probe_intensity_sum = np.sum(probe_intensity)
                
            spectrum_intensities.append(probe_intensity_sum)
        
        return np.array(spectrum_intensities)



    def diffraction(self, probe_index: int = None) -> np.ndarray:
        """
        Extract diffraction pattern for a specific probe position by summing over all frequencies.
        
        Args:
            probe_index: Index of probe position (default: 0). If None, averages over all probes.
            
        Returns:
            Diffraction pattern (kx, ky) - intensity summed over all frequencies
        """
        if probe_index is None:
            # Average over all probe positions
            all_diffractions = []
            for i in range(len(self.probe_positions)):
                probe_intensity = self.intensity[i]  # Shape: (frequency, kx, ky)
                diffraction_pattern = xp.sum(probe_intensity, axis=0)  # Sum over frequencies
                all_diffractions.append(diffraction_pattern)
            
            # Average all diffraction patterns
            if TORCH_AVAILABLE and hasattr(all_diffractions[0], 'cpu'):
                all_diffractions = [d.cpu().numpy() for d in all_diffractions]
            diffraction_pattern = np.mean(all_diffractions, axis=0)
        else:
            if probe_index >= len(self.probe_positions):
                raise ValueError(f"Probe index {probe_index} out of range")

            # Sum intensity data over all frequencies for this probe position
            probe_intensity = self.intensity[probe_index]  # Shape: (frequency, kx, ky)
            diffraction_pattern = xp.sum(probe_intensity, axis=0)  # Sum over frequencies
            
            # Convert to numpy if PyTorch tensor
            if TORCH_AVAILABLE and hasattr(diffraction_pattern, 'cpu'):
                diffraction_pattern = diffraction_pattern.cpu().numpy()
        
        return diffraction_pattern

    def spectral_diffraction(self, frequency: float, probe_index: int = None) -> np.ndarray:
        """
        Extract spectral diffraction pattern at a specific frequency.

        Args:
            frequency: Frequency value in THz
            probe_index: Index of probe position (default: None). If None, averages over all probes.

        Returns:
            Spectral diffraction pattern (kx, ky) at the specified frequency
        """
        # Find closest frequency index
        freq_idx = np.argmin(np.abs(self.frequencies - frequency))

        if probe_index is None:
            # Average over all probe positions
            all_spectral_diffractions = []
            for i in range(len(self.probe_positions)):
                spectral_diffraction = self.intensity[i, freq_idx, :, :]
                all_spectral_diffractions.append(spectral_diffraction)
            
            # Average all spectral diffraction patterns
            if TORCH_AVAILABLE and hasattr(all_spectral_diffractions[0], 'cpu'):
                all_spectral_diffractions = [sd.cpu().numpy() for sd in all_spectral_diffractions]
            spectral_diffraction = np.mean(all_spectral_diffractions, axis=0)
        else:
            if probe_index >= len(self.probe_positions):
                raise ValueError(f"Probe index {probe_index} out of range")

            # Extract intensity data at this frequency and probe position
            spectral_diffraction = self.intensity[probe_index, freq_idx, :, :]
            
            # Convert to numpy if PyTorch tensor
            if TORCH_AVAILABLE and hasattr(spectral_diffraction, 'cpu'):
                spectral_diffraction = spectral_diffraction.cpu().numpy()
        
        return spectral_diffraction

    def masked_spectrum(self, mask: np.ndarray, probe_index: int = None) -> np.ndarray:
        """
        Extract spectrum with spatial masking in k-space.

        Args:
            mask: Spatial mask array with shape (kx, ky)
            probe_index: Index of probe position (default: None). If None, averages over all probes.

        Returns:
            Masked spectrum (frequency intensity) with k-space mask applied
        """
        if mask.shape != (len(self.kx), len(self.ky)):
            raise ValueError(f"Mask shape {mask.shape} doesn't match k-space shape ({len(self.kx)}, {len(self.ky)})")

        if probe_index is None:
            # Average over all probe positions
            all_masked_spectra = []
            for i in range(len(self.probe_positions)):
                probe_intensity = self.intensity[i]  # Shape: (frequency, kx, ky)
                masked_intensity = probe_intensity * mask[None, :, :]  # Broadcast mask to all frequencies
                masked_spectrum = xp.sum(masked_intensity, axis=(1, 2))  # Sum over masked k-space
                all_masked_spectra.append(masked_spectrum)
            
            # Average all masked spectra
            if TORCH_AVAILABLE and hasattr(all_masked_spectra[0], 'cpu'):
                all_masked_spectra = [ms.cpu().numpy() for ms in all_masked_spectra]
            masked_spectrum = np.mean(all_masked_spectra, axis=0)
        else:
            if probe_index >= len(self.probe_positions):
                raise ValueError(f"Probe index {probe_index} out of range")

            # Extract intensity data for this probe
            probe_intensity = self.intensity[probe_index]  # Shape: (frequency, kx, ky)
            
            # Apply spatial mask in k-space
            masked_intensity = probe_intensity * mask[None, :, :]  # Broadcast mask to all frequencies
            masked_spectrum = xp.sum(masked_intensity, axis=(1, 2))  # Sum over masked k-space
            
            # Convert to numpy if PyTorch tensor
            if TORCH_AVAILABLE and hasattr(masked_spectrum, 'cpu'):
                masked_spectrum = masked_spectrum.cpu().numpy()

        return masked_spectrum

    def dispersion(self, kx_path: np.ndarray, ky_path: np.ndarray, probe_index: int = None) -> np.ndarray:
        """
        Extract dispersion relation from actual TACAW intensity data.
        
        Args:
            kx_path: kx values for dispersion calculation
            ky_path: ky values for dispersion calculation  
            probe_index: Index of probe position (default: None). If None, averages over all probes.
            
        Returns:
            Dispersion relation array with shape (n_frequencies, n_k_points)
            Real intensity data from TACAW simulation
        """
        # Find closest indices in our kxs/kys arrays for the requested paths
        kx_indices = []
        for kx_val in kx_path:
            idx = np.argmin(np.abs(self.kxs - kx_val))
            kx_indices.append(idx)
        kx_indices = np.array(kx_indices)
            
        ky_indices = []
        for ky_val in ky_path:
            idx = np.argmin(np.abs(self.kys - ky_val))
            ky_indices.append(idx)
        ky_indices = np.array(ky_indices)
        
        # Create dispersion array
        n_frequencies = len(self.frequencies)
        n_k_points = len(kx_indices)
        dispersion = np.zeros((n_frequencies, n_k_points))
        
        if probe_index is None:
            # Average over all probe positions
            for i, (kx_idx, ky_idx) in enumerate(zip(kx_indices, ky_indices)):
                all_probe_intensities = []
                for probe_idx in range(len(self.probe_positions)):
                    intensity_at_k = self.intensity[probe_idx, :, kx_idx, ky_idx]
                    if TORCH_AVAILABLE and hasattr(intensity_at_k, 'cpu'):
                        intensity_at_k = intensity_at_k.cpu().numpy()
                    all_probe_intensities.append(intensity_at_k)
                dispersion[:, i] = np.mean(all_probe_intensities, axis=0)
        else:
            if probe_index >= len(self.probe_positions):
                raise ValueError(f"Probe index {probe_index} out of range")
                
            # Extract dispersion for specific probe
            for i, (kx_idx, ky_idx) in enumerate(zip(kx_indices, ky_indices)):
                intensity_at_k = self.intensity[probe_index, :, kx_idx, ky_idx]
                if TORCH_AVAILABLE and hasattr(intensity_at_k, 'cpu'):
                    intensity_at_k = intensity_at_k.cpu().numpy()
                dispersion[:, i] = intensity_at_k
        
        return dispersion


# Example usage (for testing within this file)
if __name__ == '__main__':
    # Create dummy data matching the new format
    probe_positions = [(0.0, 0.0), (1.5, 0.0), (0.0, 1.5), (1.5, 1.5)]
    frequencies = np.linspace(0, 50, 100)  # THz
    kx = np.linspace(-1, 1, 32)
    ky = np.linspace(-1, 1, 32)

    # Create dummy intensity data for TACAW (frequency domain)
    intensity = np.random.rand(len(probe_positions), len(frequencies), len(kx), len(ky))

    tacaw_obj = TACAWData(
        probe_positions=probe_positions,
        frequency=frequencies,
        kx=kx,
        ky=ky,
        intensity=intensity
    )

    print("TACAWData object created with simplified format.")
    print(f"Probe positions: {tacaw_obj.probe_positions}")
    print(f"Frequency range (THz): {tacaw_obj.frequency.min():.2f} - {tacaw_obj.frequency.max():.2f}")
    print(f"kx range: {tacaw_obj.kx.min():.2f} - {tacaw_obj.kx.max():.2f}")
    print(f"ky range: {tacaw_obj.ky.min():.2f} - {tacaw_obj.ky.max():.2f}")

    # Test the postprocessing methods
    print("\n--- Postprocessing Method Examples ---")

    # Test spectrum
    spectrum_data = tacaw_obj.spectrum(probe_index=0)
    print(f"Spectrum for probe 0: {spectrum_data.shape} array")

    # Test spectrum image (real space intensity at specific frequency)
    spec_img = tacaw_obj.spectrum_image(frequency=10.0, probe_indices=[0, 1])
    print(f"Spectrum image at 10 THz for 2 probes: {spec_img.shape} array (real space intensities)")

    # Test diffraction
    diff_pattern = tacaw_obj.diffraction(probe_index=0)
    print(f"Diffraction pattern for probe 0: {diff_pattern.shape} array")

    # Test spectral diffraction
    spec_diff = tacaw_obj.spectral_diffraction(frequency=15.0, probe_index=0)
    print(f"Spectral diffraction at 15 THz: {spec_diff.shape} array")

    # Test masked spectrum
    mask = np.ones((len(tacaw_obj.kx), len(tacaw_obj.ky)))
    mask[:len(tacaw_obj.kx)//2, :] = 0  # Mask first half of k-space
    masked_spec = tacaw_obj.masked_spectrum(mask, probe_index=0)
    print(f"Masked spectrum: {masked_spec.shape} array")

    # Test dispersion
    disp = tacaw_obj.dispersion()
    print(f"Dispersion relation: {disp.shape} array")

    print("\nAll postprocessing methods working!") 