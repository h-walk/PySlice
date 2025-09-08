"""
Optimized multislice calculation functions for electron microscopy.


Key components:
- Probe class: Creates probe wavefunctions (plane wave or convergent beam)
- propagate function: Implements multislice propagation algorithm
- Physical constants: Accurate relativistic electron wavelength calculation
"""

import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# Physical constants (from numpyslice.py)
m_electron = 9.109383e-31    # mass of an electron, kg
q_electron = 1.602177e-19    # charge of an electron, J / eV or kg m^2/s^2 / eV  
c_light = 299792458.0        # speed of light, m / s
h_planck = 6.62607015e-34    # m^2 kg / s


def m_effective(eV):
    """Relativistic correction: E=m*c^2, so m=E/c^2, in kg"""
    return m_electron + eV * q_electron / c_light**2


def wavelength(eV):
    """
    Calculate wavelength (in Å) from electron energy (in eV) using relativistic formula.
    
    Args:
        eV: Electron energy in eV
        
    Returns:
        Wavelength in Angstroms
    """
    # EINSTEINIAN (relativistic): 
    # https://virtuelle-experimente.de/en/elektronenbeugung/wellenlaenge/de-broglie-relativistisch.php
    return h_planck * c_light / ((eV * q_electron)**2 + 2 * eV * q_electron * m_electron * c_light**2)**0.5 * 1e10


class Probe:
    """
    Optimized probe class for electron microscopy.
    
    Generates probe wavefunctions for both plane wave and convergent beam modes.

    """
    
    def __init__(self, xs, ys, mrad, eV):
        """
        Initialize probe wavefunction.
        
        Args:
            xs, ys: Real space coordinate arrays
            mrad: Convergence semi-angle in milliradians (0.0 = plane wave)
            eV: Electron energy in eV
        """
        self.xs = xs
        self.ys = ys
        self.mrad = mrad
        self.eV = eV
        self.wavelength = wavelength(eV)
        
        nx = len(xs)
        ny = len(ys)
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0]
        
        # Set up k-space frequencies
        self.kxs = np.fft.fftfreq(nx, d=dx)
        self.kys = np.fft.fftfreq(ny, d=dy)
        
        logger.info(f"Creating probe: {mrad}mrad aperture, {eV/1000:.1f}kV ({self.wavelength:.4f}Å)")
        
        # PREPARE PROBE: plane wave, or disks in reciprocal space
        if mrad == 0:
            # Plane wave
            self.array = np.ones((nx, ny), dtype=complex)
            logger.info("  Probe type: Plane wave")
        else:
            # Convergent beam - create disk in reciprocal space
            reciprocal = np.zeros((nx, ny), dtype=complex)
            radius = (mrad * 1e-3) / self.wavelength  # Convert mrad to reciprocal space units
            radii = np.sqrt(self.kxs[:, None]**2 + self.kys[None, :]**2)
            reciprocal[radii < radius] = 1
            self.array = np.fft.ifftshift(np.fft.ifft2(reciprocal))
            logger.info(f"  Probe type: Convergent beam, radius={radius:.6f} Å⁻¹")


def Propagate(probe, potential):
    """
    Optimized multislice propagation function.
    
    Implements the multislice method with proper transmission and propagation functions.
    Maintains the same interface as numpyslice.propagate but with optimized calculations.
    
    Args:
        probe: Probe object containing the incident wavefunction
        potential: Potential object containing the specimen potential
        
    Returns:
        numpy.ndarray: Exit wavefunction after multislice propagation
    """
    # Calculate interaction parameter (Kirkland Eq 5.6)
    # σ = 2π/(λE) × (m₀c² + E)/(2m₀c² + E)
    sigma = (2 * np.pi) / (probe.wavelength * probe.eV) * \
            (m_electron * c_light**2 + probe.eV) / (2 * m_electron * c_light**2 + probe.eV)
    
    # Get slice thickness
    dz = potential.zs[1] - potential.zs[0] if len(potential.zs) > 1 else 0.5
    
    # Pre-compute propagation operator in k-space (Fresnel propagation)
    # P = exp(-iπλΔz(k_x² + k_y²))
    P = np.exp(-1j * np.pi * probe.wavelength * dz * 
               (potential.kxs[:, None]**2 + potential.kys[None, :]**2))
    
    logger.info(f"Propagating through {len(potential.zs)} slices")
    logger.info(f"  Interaction parameter σ = {sigma:.6e}")
    logger.info(f"  Slice thickness Δz = {dz:.3f} Å")
    
    # Initialize wavefunction with probe
    array = probe.array.copy()
    
    # Multislice propagation through each slice
    for z in tqdm(range(len(potential.zs)), desc="Multislice propagation"):
        # Transmission function: t = exp(iσV(x,y,z))
        # where V(x,y,z) is the projected potential for slice z
        t = np.exp(1j * sigma * potential.array[:, :, z])
        
        # Apply transmission: ψ' = t × ψ
        array = t * array
        
        # Fresnel propagation to next slice (except for last slice)
        # ψ_next = IFFT(P × FFT(ψ'))
        if z < len(potential.zs) - 1:
            array = np.fft.ifft2(P * np.fft.fft2(array))
    
    logger.info("Multislice propagation complete")
    return array