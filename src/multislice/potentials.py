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
    #np.fft._fft=np.fft.fft # ALIASING IS TERRIBLE. THIS BLOCK, FOR EXAMPLE, THROWS A PSYCHO "postprocessing/tacaw_data.py", line 96, in fft_from_wf_data \n wf_fft = xp.fft.fft(wf_layer - wf_mean[:,None,:,:], axis=1) \n TypeError: fft() got an unexpected keyword argument 'axis'" BECAUSE WHEN WE ALIASED THE FUNCTION WE FAILED TO PREDICT ALL THE KWARGS WE MIGHT USE ELSEWHERE IN THE CODE
    #def fft(ary,device):
    #    return np.fft._fft(ary)
    #xp.fft.fft=fft
    #np._zeros=np.zeros
    #def zeros(tup,dtype=float_dtype,device=None):
    #    return np._zeros(tup,dtype=dtype)
    #xp.zeros=zeros
    #np._sum=np.sum
    #def sum(ary,dim=None,axis=None): # WATCH OUT: imports apply throughout: if we alias a kwarg, then the calling function might still expect to find the unaliased kwarg
    #    if axis is not None:
    #        return np._sum(ary,axis=axis)
    #    return np._sum(ary,axis=dim)
    #np.sum=sum

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
        # Get device from qsq tensor if it's a PyTorch tensor
        if hasattr(qsq, 'device'):
            loadKirkland(qsq.device)
        else:
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
    
    kwarg = {"dim":0} if TORCH_AVAILABLE else {"axis":0}
    term1 = xp.sum(a_expanded / (qsq_expanded + b_expanded), **kwarg)
    term2 = xp.sum(c_expanded * xp.exp(-d_expanded * qsq_expanded), **kwarg)
    
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
    # Use box matrix diagonal elements for orthogonal simulation cells
    box_matrix = trajectory.box_matrix
    
    # Extract box dimensions from diagonal elements (assuming orthogonal box)
    lx = box_matrix[0, 0]  # X dimension
    ly = box_matrix[1, 1]  # Y dimension  
    lz = box_matrix[2, 2]  # Z dimension
    
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
    
    # Convert to appropriate tensor format with device handling
    if TORCH_AVAILABLE:
        # Use float32 for MPS compatibility (Apple Silicon doesn't support float64)
        if isinstance(device, str):
            device = torch.device(device)
        dtype = torch.float32 if device.type == 'mps' else torch.float64
        kirklandABCDs = torch.tensor(kirkland_params, dtype=dtype, device=device)
    else:
        kirklandABCDs = np.asarray(kirkland_params)

class Potential:    
    def __init__(self, xs, ys, zs, positions, atomTypes, kind="kirkland", device=None, slice_axis=2):
        # Set up device and backend first
        if TORCH_AVAILABLE:
            # Auto-detect device if not specified
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
            
            # Use float32 for MPS compatibility (Apple Silicon doesn't support float64)
            self.dtype = torch.float32 if device.type == 'mps' else torch.float64
            self.complex_dtype = torch.complex64 if device.type == 'mps' else torch.complex128
            
            # Convert inputs to PyTorch tensors on device
            self.xs = torch.tensor(xs, dtype=self.dtype, device=device)
            self.ys = torch.tensor(ys, dtype=self.dtype, device=device)
            self.zs = torch.tensor(zs, dtype=self.dtype, device=device)
            positions = torch.tensor(positions, dtype=self.dtype, device=device)
        else:
            if device is not None:
                raise ImportError("PyTorch not available. Please install PyTorch.")
            self.device = None
            self.use_torch = False
            self.dtype = np.float64
            self.complex_dtype = np.complex128
            self.xs = xs
            self.ys = ys 
            self.zs = zs

        nx = len(xs)
        ny = len(ys)
        nz = len(zs)
        dx = xs[1] - xs[0]
        dy = ys[1] - ys[0] 
        dz = zs[1] - zs[0] if nz > 1 else 0.5
        
        # Store slice axis for later use
        self.slice_axis = slice_axis
        
        # Determine in-plane axes based on slice axis
        all_axes = [0, 1, 2]
        all_axes.remove(slice_axis)
        self.inplane_axis1, self.inplane_axis2 = all_axes
        
        # Store coordinate arrays and spacing for the slice axis
        coord_arrays = [xs, ys, zs]
        spacings = [dx, dy, dz]
        self.slice_coords = coord_arrays[slice_axis]
        self.slice_spacing = spacings[slice_axis]
        self.n_slices = len(self.slice_coords)
        
        # Set up device kwargs for unified xp interface
        device_kwargs = {'device': self.device, 'dtype': self.dtype} if self.use_torch else {}
        
        # Set up k-space frequencies using xp with conditional device
        self.kxs = xp.fft.fftfreq(nx, d=dx, **device_kwargs)
        self.kys = xp.fft.fftfreq(ny, d=dy, **device_kwargs)
        qsq = self.kxs[:, None]**2 + self.kys[None, :]**2
        
        # Initialize potential array using xp with conditional device
        device_kwargs = {'device': self.device } if self.use_torch else {}
        reciprocal = xp.zeros((nx, ny, self.n_slices), dtype=self.complex_dtype, **device_kwargs)
        
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
            
            # OPTIMIZATION 3: Batch process all slices for this atom type along the specified axis
            # Create slice masks for all slices at once
            slice_coords = positions[type_mask, slice_axis]  # Get coordinates along slice axis for this atom type
            
            if len(slice_coords) == 0:
                continue
                
            for slice_idx in range(self.n_slices):
                # Vectorized spatial masking using correct slice coordinates
                slice_min = self.slice_coords[slice_idx] - self.slice_spacing/2 if slice_idx > 0 else 0
                slice_max = self.slice_coords[slice_idx] + self.slice_spacing/2 if slice_idx < self.n_slices-1 else self.slice_coords[-1] + self.slice_spacing
                
                spatial_mask = (slice_coords >= slice_min) & (slice_coords < slice_max)
                
                if not xp.any(spatial_mask):
                   continue  # Skip empty slices
                
                # Get positions for atoms in this slice and type
                type_positions = positions[type_mask]
                slice_positions = type_positions[spatial_mask]
                
                if len(slice_positions) == 0:
                    continue
                
                atomsx = slice_positions[:, self.inplane_axis1]
                atomsy = slice_positions[:, self.inplane_axis2]
                
                # Compute structure factors - match NumPy pattern exactly
                expx = xp.exp(-1j * 2 * np.pi * self.kxs[None, :] * atomsx[:, None])
                expy = xp.exp(-1j * 2 * np.pi * self.kys[None, :] * atomsy[:, None])
                
                # Einstein summation - match NumPy
                kwarg={True:{},False:{"optimize":True}}[TORCH_AVAILABLE]
                shape_factor = xp.einsum('ax,ay->xy', expx, expy, **kwarg)
                
                reciprocal[:, :, slice_idx] += shape_factor * form_factor
        
        # Slice-by-slice IFFT to match NumPy implementation exactly
        potential_real = xp.zeros((nx, ny, self.n_slices), dtype=float_dtype, device=device)
        for slice_idx in range(self.n_slices):
            potential_slice = xp.fft.ifft2(reciprocal[:, :, slice_idx])
            potential_real[:, :, slice_idx] = xp.real(potential_slice)
        
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
        if self.use_torch:
            return self.array.cpu().numpy()
        else:
            return self.array
    
    def to_device(self, device):
        """Move tensor data to specified device."""
        if hasattr(self, 'array_torch'):
            self.array_torch = self.array_torch.to(device)
        self.device = device
        return self