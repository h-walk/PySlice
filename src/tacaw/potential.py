import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def kirkland(qsq, Z):
    """
    Generate the Kirkland structure factor (form factor) in reciprocal space.
    
    Args:
        qsq: |q|² array in units of (1/Angstrom)²
        Z: Atomic number (or element name string)
        
    Returns:
        Form factor array with same shape as qsq
    """
    global kirklandABCDs
    
    if len(kirklandABCDs) == 0:
        loadKirkland()
        
    if isinstance(Z, str):
        Z = getZfromElementName(Z)
    Z -= 1  # Convert to 0-based indexing
    
    # Grab columns for a,b,c,d parameters
    ABCDs = kirklandABCDs[Z, :, :]  # atomic number "1" = "H" = index 0
    a = ABCDs[:, 0]
    b = ABCDs[:, 1] 
    c = ABCDs[:, 2]
    d = ABCDs[:, 3]
    
    return np.sum(a[:, None, None] / (qsq[None, :, :] + b[:, None, None]), axis=0) + \
           np.sum(c[:, None, None] * np.exp(-d[:, None, None] * qsq[None, :, :]), axis=0)


def getZfromElementName(element):
    """Return atomic number (Z) from element name."""
    elements = ["H", "He",
                "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
                "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
                "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
                "Cs", "Ba",
                "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
                "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Ti", "Pb", "Bi", "Po", "At", "Rn",
                "Fr", "Ra",
                "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No",
                "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"]
    return elements.index(element) + 1


# Global storage for Kirkland parameters
kirklandABCDs = []

def loadKirkland():
    """Load Kirkland parameters from kirkland.txt file."""
    global kirklandABCDs
    
    # Try to find kirkland.txt in the project directory
    kirkland_file = None
    search_paths = [
        'kirkland.txt',
        '../kirkland.txt', 
        '../../kirkland.txt',
        Path(__file__).parent.parent.parent / 'kirkland.txt'
    ]
    
    for path in search_paths:
        if Path(path).exists():
            kirkland_file = str(path)
            break
            
    if kirkland_file is None:
        raise FileNotFoundError("Could not find kirkland.txt file")
    
    # Parse Kirkland parameters
    kirkland_params = []
    
    for i in range(103):  # Elements 1-103
        rows = (i * 4 + 1, i * 4 + 5)  # Skip header, read 3 lines
        try:
            abcd = np.loadtxt(kirkland_file, skiprows=rows[0], max_rows=3)
            # ORDERING IS: (Kirkland page 291)
            # a1 b1 a2 b2
            # a3 b4 c1 d1
            # c2 d2 c3 d3
            a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3 = abcd.flat
            # reorder so four columns are a,b,c,d
            abcd = [[a1, b1, c1, d1], [a2, b2, c2, d2], [a3, b3, c3, d3]]
            kirkland_params.append(abcd)
        except Exception as e:
            # Fill with zeros if parameters not available
            kirkland_params.append([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    
    kirklandABCDs = np.asarray(kirkland_params)


class Potential:
    """
    Optimized potential class for multislice electron microscopy.

    """
    
    def __init__(self, xs, ys, zs, positions, atomTypes, kind="kirkland"):
        """
        Initialize optimized potential calculation.
        
        Args:
            xs, ys, zs: Coordinate arrays for the 3D grid
            positions: Array of atomic positions (N_atoms, 3)
            atomTypes: Array of atomic types (atomic numbers or strings)
            kind: Type of potential ("kirkland" or "gauss")
        """
        self.xs = xs
        self.ys = ys 
        self.zs = zs
        
        nx = len(xs)
        ny = len(ys)
        nz = len(zs)
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0] 
        dz = zs[1] - zs[0] if nz > 1 else 0.5
        
        # Set up k-space frequencies
        self.kxs = np.fft.fftfreq(nx, d=dx)
        self.kys = np.fft.fftfreq(ny, d=dy)
        qsq = self.kxs[:, None]**2 + self.kys[None, :]**2
        
        
        # Initialize potential array
        reciprocal = np.zeros((nx, ny, nz), dtype=complex)
        
        # Convert atom types to atomic numbers if needed
        unique_atom_types = set(atomTypes)
        atomic_numbers = []
        for at in atomTypes:
            if isinstance(at, str):
                atomic_numbers.append(getZfromElementName(at))
            else:
                atomic_numbers.append(at)
        atomic_numbers = np.array(atomic_numbers)
        
        # OPTIMIZATION 1: Compute form factors once per atom type
        form_factors = {}
        for at in unique_atom_types:
            if kind == "kirkland":
                if isinstance(at, str):
                    Z = getZfromElementName(at) 
                else:
                    Z = at
                form_factors[at] = kirkland(qsq, Z)
            elif kind == "gauss":
                form_factors[at] = np.exp(-1**2 * qsq / 2)
        
        # Process each atom type separately (reuse form factors)
        for at in unique_atom_types:
            form_factor = form_factors[at]
            
            # OPTIMIZATION 2: Vectorized atom type masking
            if isinstance(at, str):
                type_mask = np.array([atom_type == at for atom_type in atomTypes])
            else:
                type_mask = (atomic_numbers == at)
            
            # Process each z-slice
            for z in range(nz):
                # OPTIMIZATION 3: Vectorized spatial masking
                z_min = zs[z] - dz/2 if z > 0 else 0
                z_max = zs[z] + dz/2 if z < nz-1 else zs[-1] + dz
                
                spatial_mask = (positions[:, 2] >= z_min) & (positions[:, 2] < z_max)
                combined_mask = spatial_mask & type_mask
                
                if not np.any(combined_mask):
                    continue  # Skip empty slices
                
                atomsx = positions[combined_mask, 0]
                atomsy = positions[combined_mask, 1]
                
                if len(atomsx) == 0:
                    continue
                
                # Compute structure factors (same as original)
                expx = np.exp(-1j * 2 * np.pi * self.kxs[None, :] * atomsx[:, None])
                expy = np.exp(-1j * 2 * np.pi * self.kys[None, :] * atomsy[:, None])
                shape_factor = np.einsum('ax,ay->xy', expx, expy, optimize=True)
                
                reciprocal[:, :, z] += shape_factor * form_factor
        
        # OPTIMIZATION 4: Slice-by-slice IFFT for better memory efficiency
        potential_real = np.zeros((nx, ny, nz))
        for z in range(nz):
            potential_slice = np.fft.ifft2(reciprocal[:, :, z])
            potential_real[:, :, z] = np.real(potential_slice)
        
        # Apply proper normalization (same as k-space NumPy)
        self.array = potential_real * dx * dy
        
