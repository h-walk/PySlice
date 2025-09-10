import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm

try:
    import torch  ; xp = torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    if device.type == 'mps': # Use float32 for MPS (doesn't support float64), float64 for CPU/CUDA
        complex_dtype = torch.complex64
        float_dtype = torch.float32
    else:
        complex_dtype = xp.complex128
        float_dtype = xp.float64


except ImportError:
    TORCH_AVAILABLE = False
    import numpy as np ; xp = np
    print("PyTorch not available, falling back to NumPy")
    device=None
    complex_dtype = xp.complex128
    float_dtype = xp.float64
    np.fft._fft=np.fft.fft
    def fft(ary,device):
        return np.fft._fft(ary)
    xp.fft.fft=fft
    np._zeros=np.zeros
    def zeros(tup,dtype=float_dtype,device=None):
        return np._zeros(tup,dtype=dtype)
    xp.zeros=zeros
    np._sum=np.sum
    def sum(ary,dim=None,axis=None): # WATCH OUT: imports apply throughout: if we alias a kwarg, then the calling function might still expect to find the unaliased kwarg
        if axis is not None:
            return np._sum(ary,axis=axis)
        return np._sum(ary,axis=dim)
    np.sum=sum

logger = logging.getLogger(__name__)

# Global storage for Kirkland parameters on GPU - store per device
kirklandABCDs = []
def kirkland(qsq, Z):
    """
    GPU-accelerated Kirkland structure factor calculation using PyTorch.
    
    Args:
                if device is not None and not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Please install PyTorch.")
        
: |q|² tensor in units of (1/Angstrom)²
        Z: Atomic number (or element name string)
        device: PyTorch device ('cpu' or 'cuda')
        
    Returns:
        Form factor tensor with same shape as qsq
    """
    global kirklandABCDs

    if len(kirklandABCDs)==0:
        loadKirkland()

    if isinstance(Z, str):
        Z = getZfromElementName(Z)
    Z -= 1  # Convert to 0-based indexing

    # Grab columns for a,b,c,d parameters - already on correct device
    ABCDs = kirklandABCDs[Z, :, :]  
    a = ABCDs[:, 0]
    b = ABCDs[:, 1] 
    c = ABCDs[:, 2]
    d = ABCDs[:, 3]
    
    # Vectorized computation on GPU
    a_expanded = a[:, None, None]
    b_expanded = b[:, None, None]
    c_expanded = c[:, None, None]
    d_expanded = d[:, None, None]
    qsq_expanded = qsq[None, :, :]
    
    term1 = xp.sum(a_expanded / (qsq_expanded + b_expanded), dim=0)
    term2 = xp.sum(c_expanded * xp.exp(-d_expanded * qsq_expanded), dim=0)
    
    return term1 + term2

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

def gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5):
    lx, ly, lz = trajectory.box_matrix[0,0], trajectory.box_matrix[1,1], trajectory.box_matrix[2,2]
    
    # Create grids based on sampling
    nx = int(lx / sampling) + 1
    ny = int(ly / sampling) + 1  
    nz = int(lz / slice_thickness) + 1
     
    xs = np.linspace(0, lx, nx, endpoint=False)
    ys = np.linspace(0, ly, ny, endpoint=False)
    zs = np.linspace(0, lz, nz, endpoint=False)

    return xs,ys,zs,lx,ly,lz


def loadKirkland(device='cpu'):
    """Load Kirkland parameters from kirkland.txt file and move to GPU."""
    global kirklandABCDs
    
    # Convert device to string for dictionary key
    device_key = str(device)
    
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
    
    # Convert to PyTorch tensor and move to device - store per device
    if TORCH_AVAILABLE:
        kirklandABCDs = torch.tensor(kirkland_params, dtype=torch.float64, device=device)
    else:
        kirklandABCDs = np.asarray(kirkland_params)

class Potential:    
    def __init__(self, xs, ys, zs, positions, atomTypes, kind="kirkland", device=None):
        if TORCH_AVAILABLE:
            # Auto-detect device if not specified
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = device
        
            # Convert inputs to PyTorch tensors on device
            self.xs = torch.tensor(xs, dtype=torch.float64, device=device)
            self.ys = torch.tensor(ys, dtype=torch.float64, device=device)
            self.zs = torch.tensor(zs, dtype=torch.float64, device=device)
        
            positions = torch.tensor(positions, dtype=torch.float64, device=device)
        else:
            if device is not None:
                raise ImportError("PyTorch not available. Please install PyTorch.")
            self.xs = xs ; self.ys = ys ; self.zs = zs

        nx = len(xs)
        ny = len(ys)
        nz = len(zs)
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0] 
        dz = zs[1] - zs[0] if nz > 1 else 0.5
        
        # Set up k-space frequencies on GPU
        self.kxs = xp.fft.fftfreq(nx, d=dx, device=device)
        self.kys = xp.fft.fftfreq(ny, d=dy, device=device)
        qsq = self.kxs[:, None]**2 + self.kys[None, :]**2
        
        # Initialize potential array on GPU 
        reciprocal = xp.zeros((nx, ny, nz), dtype=complex_dtype, device=device)
        
        # Convert atom types to atomic numbers if needed
        unique_atom_types = set(atomTypes)
        atomic_numbers = []
        for at in atomTypes:
            if isinstance(at, str):
                atomic_numbers.append(getZfromElementName(at))
            else:
                atomic_numbers.append(at)
        if TORCH_AVAILABLE:
            atomic_numbers = torch.tensor(atomic_numbers, device=device)

        # OPTIMIZATION 1: Compute form factors once per atom type on GPU
        form_factors = {}
        for at in unique_atom_types:
            if kind == "kirkland":
                if isinstance(at, str):
                    Z = getZfromElementName(at) 
                else:
                    Z = at
                form_factors[at] = kirkland(qsq, Z)
            elif kind == "gauss":
                form_factors[at] = torch.exp(-1**2 * qsq / 2)
        
        # Process each atom type separately (reuse form factors)
        for at in unique_atom_types:
            form_factor = form_factors[at]
            
            # OPTIMIZATION 2: Vectorized atom type masking on GPU
            if isinstance(at, str):
                type_mask=[atom_type == at for atom_type in atomTypes]
                if TORCH_AVAILABLE:
                    type_mask = torch.tensor(type_mask, 
                                       dtype=torch.bool, device=device)
            else:
                type_mask = (atomic_numbers == at)
            
            # OPTIMIZATION 3: Batch process all z-slices for this atom type
            # Create z-slice masks for all slices at once
            z_coords = positions[type_mask, 2]  # Get z-coordinates for this atom type
            
            if len(z_coords) == 0:
                continue
                
            for z in range(nz):
                # Vectorized spatial masking on GPU
                z_min = self.zs[z] - dz/2 if z > 0 else 0
                z_max = self.zs[z] + dz/2 if z < nz-1 else self.zs[-1] + dz
                
                spatial_mask = (z_coords >= z_min) & (z_coords < z_max)
                
                if not xp.any(spatial_mask):
                    continue  # Skip empty slices
                
                # Get positions for atoms in this slice and type
                type_positions = positions[type_mask]
                slice_positions = type_positions[spatial_mask]
                
                if len(slice_positions) == 0:
                    continue
                
                atomsx = slice_positions[:, 0]
                atomsy = slice_positions[:, 1]
                
                # Compute structure factors - match NumPy pattern exactly
                expx = xp.exp(-1j * 2 * xp.pi * self.kxs[None, :] * atomsx[:, None])
                expy = xp.exp(-1j * 2 * xp.pi * self.kys[None, :] * atomsy[:, None])
                
                # Einstein summation - match NumPy
                kwarg={True:{},False:{"optimize":True}}[TORCH_AVAILABLE]
                shape_factor = xp.einsum('ax,ay->xy', expx, expy, **kwarg)
                
                reciprocal[:, :, z] += shape_factor * form_factor
        
        # Slice-by-slice IFFT to match NumPy implementation exactly
        potential_real = xp.zeros((nx, ny, nz), dtype=float_dtype, device=device)
        for z in range(nz):
            potential_slice = xp.fft.ifft2(reciprocal[:, :, z])
            potential_real[:, :, z] = xp.real(potential_slice)
        
        # Apply proper normalization factor (dx²×dy²) to match reference implementation
        dx = self.xs[1] - self.xs[0]
        dy = self.ys[1] - self.ys[0] 
        potential_real = potential_real / (dx**2 * dy**2)
        
        #if TORCH_AVAILABLE:
        #   self.array = potential_real.cpu().numpy()  # Move to CPU and convert to NumPy
        
        # Store tensor version for potential GPU operations
        self.array = potential_real
        
    def to_cpu(self):
        """Convert tensors back to CPU NumPy arrays."""
        return self.array
    
    def to_device(self, device):
        """Move tensor data to specified device."""
        if hasattr(self, 'array_torch'):
            self.array_torch = self.array_torch.to(device)
        self.device = device
        return self