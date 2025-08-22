"""
Core data structure for TACAW EELS calculations.
"""
from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path
import logging
import pickle
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class TACAWData:
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

    def spectrum(self, probe_index: int = 0) -> np.ndarray:
        """
        Extract spectrum for a specific probe position by summing over all k-space.

        Args:
            probe_index: Index of probe position (default: 0)

        Returns:
            Spectrum array (frequency intensity)
        """
        if probe_index >= len(self.probe_positions):
            raise ValueError(f"Probe index {probe_index} out of range")

        # Sum intensity data over all k-space for this probe position
        probe_intensity = self.intensity[probe_index]  # Shape: (frequency, kx, ky)
        spectrum = np.sum(probe_intensity, axis=(1, 2))  # Sum over kx, ky
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
        freq_idx = np.argmin(np.abs(self.frequency - frequency))

        # Use all probes if none specified
        if probe_indices is None:
            probe_indices = list(range(len(self.probe_positions)))

        # Extract intensity at this frequency for each selected probe position
        spectrum_intensities = []
        for probe_idx in probe_indices:
            # Sum intensity data over all k-space for this probe at this frequency
            probe_intensity = np.sum(self.intensity[probe_idx, freq_idx, :, :])
            spectrum_intensities.append(probe_intensity)
        
        return np.array(spectrum_intensities)

    def spectrum_image_2d(self, frequency: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a 2D spectrum image map when probe positions are arranged in a grid.
        
        Args:
            frequency: Frequency value in THz
            
        Returns:
            Tuple of (X_grid, Y_grid, intensity_map) for 2D plotting
        """
        # Get intensities for all probe positions
        intensities = self.spectrum_image(frequency)
        
        # Extract x and y coordinates of probe positions
        x_coords = np.array([pos[0] for pos in self.probe_positions])
        y_coords = np.array([pos[1] for pos in self.probe_positions])
        
        # Create unique sorted coordinates for gridding
        unique_x = np.unique(x_coords)
        unique_y = np.unique(y_coords)
        
        # Create 2D grid
        X_grid, Y_grid = np.meshgrid(unique_x, unique_y)
        intensity_map = np.zeros_like(X_grid)
        
        # Fill intensity map
        for i, (x, y) in enumerate(self.probe_positions):
            x_idx = np.where(unique_x == x)[0][0]
            y_idx = np.where(unique_y == y)[0][0]
            intensity_map[y_idx, x_idx] = intensities[i]
        
        return X_grid, Y_grid, intensity_map

    def diffraction(self, probe_index: int = 0) -> np.ndarray:
        """
        Extract diffraction pattern for a specific probe position by summing over all frequencies.
        
        Args:
            probe_index: Index of probe position (default: 0)
            
        Returns:
            Diffraction pattern (kx, ky) - intensity summed over all frequencies
        """
        if probe_index >= len(self.probe_positions):
            raise ValueError(f"Probe index {probe_index} out of range")

        # Sum intensity data over all frequencies for this probe position
        probe_intensity = self.intensity[probe_index]  # Shape: (frequency, kx, ky)
        diffraction_pattern = np.sum(probe_intensity, axis=0)  # Sum over frequencies
        return diffraction_pattern

    def spectral_diffraction(self, frequency: float, probe_index: int = 0) -> np.ndarray:
        """
        Extract spectral diffraction pattern at a specific frequency.

        Args:
            frequency: Frequency value in THz
            probe_index: Index of probe position (default: 0)

        Returns:
            Spectral diffraction pattern (kx, ky) at the specified frequency
        """
        if probe_index >= len(self.probe_positions):
            raise ValueError(f"Probe index {probe_index} out of range")

        # Find closest frequency index
        freq_idx = np.argmin(np.abs(self.frequency - frequency))

        # Extract intensity data at this frequency and probe position
        spectral_diffraction = self.intensity[probe_index, freq_idx, :, :]
        return spectral_diffraction

    def masked_spectrum(self, mask: np.ndarray, probe_index: int = 0) -> np.ndarray:
        """
        Extract spectrum with spatial masking in k-space.

        Args:
            mask: Spatial mask array with shape (kx, ky)
            probe_index: Index of probe position (default: 0)

        Returns:
            Masked spectrum (frequency intensity) with k-space mask applied
        """
        if probe_index >= len(self.probe_positions):
            raise ValueError(f"Probe index {probe_index} out of range")

        # Extract intensity data for this probe
        probe_intensity = self.intensity[probe_index]  # Shape: (frequency, kx, ky)
        
        # Apply spatial mask in k-space
        if mask.shape == (len(self.kx), len(self.ky)):
            masked_intensity = probe_intensity * mask[None, :, :]  # Broadcast mask to all frequencies
            masked_spectrum = np.sum(masked_intensity, axis=(1, 2))  # Sum over masked k-space
        else:
            raise ValueError(f"Mask shape {mask.shape} doesn't match k-space shape ({len(self.kx)}, {len(self.ky)})")

        return masked_spectrum

    def dispersion(self) -> np.ndarray:
        """
        Calculate phonon dispersion from TACAW data.

        Returns:
            Dispersion relation array
        """
        # For now, return a simple dispersion calculation
        # In a real implementation, this would compute phonon dispersion
        kx_mesh, ky_mesh = np.meshgrid(self.kx, self.ky)
        k_magnitude = np.sqrt(kx_mesh**2 + ky_mesh**2)

        # Simple linear dispersion example: ω = v * |k|
        sound_velocity = 5000  # m/s (example)
        omega = sound_velocity * k_magnitude * 1e-12  # Convert to THz

        return omega

    def save(self, filepath: Union[str, Path], auto_generate_filename: bool = False,
             trajectory=None, sim_params: dict = None, probe_positions: list = None) -> bool:
        """
        Save TACAWData object to disk using pickle.

        Args:
            filepath: Path where to save the file (or directory if auto_generate_filename=True)
            auto_generate_filename: If True, generate a unique filename based on simulation parameters
            trajectory: Trajectory object (required if auto_generate_filename=True)
            sim_params: Simulation parameters dict (required if auto_generate_filename=True)
            probe_positions: List of probe positions (required if auto_generate_filename=True)

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)

            # Auto-generate filename if requested
            if auto_generate_filename:
                if trajectory is None or sim_params is None or probe_positions is None:
                    logger.error("trajectory, sim_params, and probe_positions required for auto filename generation")
                    return False

                # Generate unique filename based on simulation parameters
                param_string = f"{trajectory.n_frames}_{trajectory.n_atoms}_{len(probe_positions)}"
                param_string += f"_{sim_params.get('aperture', 0.0)}_{sim_params.get('voltage_kv', 100.0)}"
                param_string += f"_{sim_params.get('sampling', 0.1)}_{sim_params.get('slice_thickness', 0.5)}"
                param_string += f"_{trajectory.timestep}"

                # Add probe positions to the hash
                for pos in probe_positions:
                    param_string += f"_{pos[0]:.3f}_{pos[1]:.3f}"

                # Create hash for unique identification
                param_hash = hashlib.md5(param_string.encode()).hexdigest()[:8]
                filename = f"tacaw_data_{trajectory.n_frames}f_{len(probe_positions)}p_{param_hash}.pkl"
                filepath = filepath / filename

            # Ensure parent directory exists
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"TACAWData saved to: {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save TACAWData to {filepath}: {e}")
            return False

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> Optional['TACAWData']:
        """
        Load TACAWData object from disk using pickle.

        Args:
            filepath: Path to the file to load

        Returns:
            TACAWData object if successful, None otherwise
        """
        try:
            filepath = Path(filepath)

            if not filepath.exists():
                logger.error(f"File does not exist: {filepath}")
                return None

            with open(filepath, 'rb') as f:
                obj = pickle.load(f)

            # Validate that we loaded the correct type
            if not isinstance(obj, cls):
                logger.error(f"Loaded object is not a {cls.__name__} instance")
                return None

            logger.info(f"TACAWData loaded from: {filepath}")
            return obj

        except Exception as e:
            logger.error(f"Failed to load TACAWData from {filepath}: {e}")
            return None

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
    
    # Test 2D spectrum image
    X, Y, intensity_map = tacaw_obj.spectrum_image_2d(frequency=10.0)
    print(f"2D spectrum image at 10 THz: {intensity_map.shape} spatial map")

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