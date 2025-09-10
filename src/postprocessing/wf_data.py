"""
Wave function data structure.
"""
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple
from ..multislice.multislice import Probe

@dataclass
class WFData:
    """
    Data structure for wave function data with format: probe_positions, frame, kx, ky, layer.

    Attributes:
        probe_positions: List of (x,y) probe positions in Angstroms.
        frame: Time array (frame # * timestep) in picoseconds.
        kx: kx sampling vectors.
        ky: ky sampling vectors.
        layer: Layer indices for multi-layer calculations.
        wavefunction_data: Complex wavefunction array with shape (probe_positions, time, kx, ky, layer).
    """
    probe_positions: List[Tuple[float, float]]
    time: np.ndarray  # Time in picoseconds (frame # * timestep)
    kxs: np.ndarray    # kx sampling vectors
    kys: np.ndarray    # ky sampling vectors
    layer: np.ndarray # Layer indices
    wavefunction_data: np.ndarray  # Complex wavefunction array (probe_positions, time, kx, ky, layer)
    probe: Probe

    def fft_to_tacaw_data(self, layer_index: int = None):
        """
        Perform FFT along the time axis for a specific layer to convert to TACAW data.
        This implements the JACR method: Ψ(t,q,r) → |Ψ(ω,q,r)|² via FFT.

        Args:
            layer_index: Index of the layer to compute FFT for (default: last layer)

        Returns:
            TACAWData object with intensity data |Ψ(ω,q)|² for the specified layer
        """
        from .tacaw_data import TACAWData

        # Default to last layer if not specified
        if layer_index is None:
            layer_index = len(self.layer) - 1

        # Validate layer index
        if layer_index < 0 or layer_index >= len(self.layer):
            raise ValueError(f"layer_index {layer_index} out of range [0, {len(self.layer)-1}]")

        # Compute frequencies from time sampling
        n_freq = len(self.time)
        dt = self.time[1] - self.time[0] 
        frequencies_thz = np.fft.fftfreq(n_freq, d=dt)
        frequencies_thz = np.fft.fftshift(frequencies_thz)

        # Extract wavefunction data for the specified layer
        # Shape: (probe_positions, time, kx, ky, layer)
        wf_layer = self.wavefunction_data[:, :, :, :, layer_index]
        
        # Perform FFT along time axis (axis=1) for each probe position and k-point
        # Following abeels.py approach: subtract mean to avoid high zero-frequency peak
        wf_mean = np.mean(wf_layer, axis=1, keepdims=True)
        wf_fft = np.fft.fft(wf_layer - wf_mean, axis=1)
        wf_fft = np.fft.fftshift(wf_fft, axes=1)
        
        # Compute intensity |Ψ(ω,q)|² from the frequency-domain wavefunction
        intensity = np.abs(wf_fft)**2

        return TACAWData(
            probe_positions=self.probe_positions,
            frequency=frequencies_thz,
            kx=self.kxs,
            ky=self.kys,
            intensity=intensity  # Store the intensity data |Ψ(ω,q)|²
        )


# Example usage (for testing within this file)
if __name__ == '__main__':
    # Create dummy data matching the new format
    probe_positions = [(0.0, 0.0), (1.5, 0.0), (0.0, 1.5)]
    timesteps = np.arange(100) * 0.001  # frame # * timestep (ps)
    kx = np.linspace(-1, 1, 32)
    ky = np.linspace(-1, 1, 32)
    layers = np.arange(5)  # 5 layers

    # Create dummy wavefunction data
    wavefunction_data = np.random.complex128((len(probe_positions), len(timesteps), len(kx), len(ky), len(layers)))
    
    wf_data = WFData(
        probe_positions=probe_positions,
        time=timesteps,
        kx=kx,
        ky=ky,
        layer=layers,
        wavefunction_data=wavefunction_data
    )

    print("WFData object created with specified format.")
    print(f"Probe positions: {wf_data.probe_positions}")
    print(f"Time range (ps): {wf_data.time.min():.4f} - {wf_data.time.max():.4f}")
    print(f"kx range: {wf_data.kx.min():.2f} - {wf_data.kx.max():.2f}")
    print(f"ky range: {wf_data.ky.min():.2f} - {wf_data.ky.max():.2f}")
    print(f"Layer range: {wf_data.layer.min()} - {wf_data.layer.max()}")

    # Demonstrate FFT conversion to TACAW data
    print(f"\n--- FFT Conversion Examples ---")

    # Convert default layer (last layer)
    tacaw_data = wf_data.fft_to_tacaw_data()
    last_layer = len(wf_data.layer) - 1
    print(f"Default (last layer {last_layer}) converted to TACAW data:")
    print(f"Frequency range (THz): {tacaw_data.frequency.min():.2f} - {tacaw_data.frequency.max():.2f}")
    print(f"Probe positions: {tacaw_data.probe_positions}")
    print(f"kx range: {tacaw_data.kx.min():.2f} - {tacaw_data.kx.max():.2f}")
    print(f"ky range: {tacaw_data.ky.min():.2f} - {tacaw_data.ky.max():.2f}")

    # Convert specific layer if needed
    if len(wf_data.layer) > 1:
        tacaw_data_specific = wf_data.fft_to_tacaw_data(layer_index=0)
        print(f"\nSpecific layer 0 converted to TACAW data:")
        print(f"Frequency range (THz): {tacaw_data_specific.frequency.min():.2f} - {tacaw_data_specific.frequency.max():.2f} THz")
