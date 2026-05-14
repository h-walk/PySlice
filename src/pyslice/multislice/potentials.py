import numpy as np
from pathlib import Path
import logging,os
from tqdm import tqdm
from ..backend import to_cpu,mean

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
    else:
        # Move kirklandABCDs to match qsq device if needed
        if hasattr(qsq, 'device') and hasattr(kirklandABCDs, 'device'):
            if qsq.device != kirklandABCDs.device:
                kirklandABCDs = kirklandABCDs.to(qsq.device)

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
        Path(__file__).parent.parent.parent / 'kirkland.txt',
        Path(__file__).parent.parent / 'data/kirkland.txt' # relative path from potentials.py to data folder? i think this is the *right* place for kirkland.txt, not at the root of PySlice...
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
    def __init__(self, xs, ys, zs, positions=None, atomTypes=None, array=None, kind="kirkland", device=None, slice_axis=2, progress=False, cache_dir=None, frame_idx=None):
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
            if positions is not None:
                positions = torch.tensor(positions, dtype=self.dtype, device=device)
            if array is not None:
                self._array = torch.tensor(array, device=device)
            else:
                self._array = None
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
            self._array = array

        self.nx = len(xs)
        self.ny = len(ys)
        self.nz = len(zs)
        self.dx = xs[1] - xs[0]
        self.dy = ys[1] - ys[0] 
        self.dz = zs[1] - zs[0] if self.nz > 1 else 0.5
        
        # Store slice axis for later use
        self.slice_axis = slice_axis
        
        # Determine in-plane axes based on slice axis
        all_axes = [0, 1, 2]
        all_axes.remove(slice_axis)
        self.inplane_axis1, self.inplane_axis2 = all_axes
        
        # Store coordinate arrays and spacing for the slice axis
        coord_arrays = [xs, ys, zs]
        spacings = [self.dx, self.dy, self.dz]
        self.slice_coords = coord_arrays[slice_axis]
        self.slice_spacing = spacings[slice_axis]
        self.n_slices = len(self.slice_coords)
        slice_coords_np = np.asarray(self.slice_coords, dtype=np.float64)
        slice_edges_np = np.empty(self.n_slices + 1, dtype=np.float64)
        slice_edges_np[0] = 0.0
        if self.n_slices > 1:
            slice_edges_np[1:-1] = 0.5 * (slice_coords_np[:-1] + slice_coords_np[1:])
            slice_edges_np[-1] = slice_coords_np[-1] + 0.5 * self.slice_spacing
        else:
            slice_edges_np[-1] = slice_coords_np[-1] + 0.5 * self.slice_spacing
        self.slice_edges = (
            xp.tensor(slice_edges_np, dtype=self.dtype, device=self.device)
            if TORCH_AVAILABLE
            else slice_edges_np
        )

        def slice_membership_mask(coords, slice_idx):
            if TORCH_AVAILABLE and isinstance(coords, torch.Tensor):
                slice_indices = torch.bucketize(coords.contiguous(), self.slice_edges, right=True) - 1
            else:
                slice_indices = np.searchsorted(self.slice_edges, coords, side="right") - 1
            return slice_indices == slice_idx

        if positions is not None:
            cell_lengths_np = np.array([self.nx * self.dx, self.ny * self.dy, slice_edges_np[-1]])
            cell_lengths = (
                xp.tensor(cell_lengths_np, dtype=self.dtype, device=self.device)
                if TORCH_AVAILABLE
                else cell_lengths_np
            )
            positions = positions.clone() if TORCH_AVAILABLE else positions.copy()
            for ax in range(3):
                positions[:, ax] = positions[:, ax] % cell_lengths[ax]
        
        # Set up device kwargs for unified xp interface
        device_kwargs = {'device': self.device, 'dtype': self.dtype} if self.use_torch else {}
        
        # Set up k-space frequencies using xp with conditional device
        self.kxs = xp.fft.fftfreq(self.nx, d=self.dx, **device_kwargs)
        self.kys = xp.fft.fftfreq(self.ny, d=self.dy, **device_kwargs)
        qsq = self.kxs[:, None]**2 + self.kys[None, :]**2
     
        # Convert atom types to atomic numbers if needed
        if atomTypes is not None:
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
        
        self.cache_dir = cache_dir
        self.frame_idx = frame_idx

        def calculateSlice(slice_idx):
            if self._array is not None:
                return self._array[:,:,slice_idx]
            # check for caching
            cache_file = None
            if self.cache_dir is not None:
                cache_file = self.cache_dir / ("potential_"+str(frame_idx)+"_"+str(slice_idx)+".npy")
            if cache_file is not None and os.path.exists(cache_file):
                Z = np.load(cache_file)
                if TORCH_AVAILABLE:
                    return xp.from_numpy(Z).to(device)
                return Z

            # Initialize slice of potential array using xp with conditional device
            device_kwargs = {'device': self.device } if self.use_torch else {}
            reciprocal = xp.zeros((self.nx, self.ny), dtype=self.complex_dtype, **device_kwargs)

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
            
                spatial_mask = slice_membership_mask(slice_coords, slice_idx)
                
                if not xp.any(spatial_mask):
                    continue #return xp.zeros((len(self.kxs),len(self.kys)))
                
                # Get positions for atoms in this slice and type
                type_positions = positions[type_mask]
                slice_positions = type_positions[spatial_mask]
                
                if len(slice_positions) == 0:
                    continue #return xp.zeros((len(self.kxs),len(self.kys)))
                
                atomsx = slice_positions[:, self.inplane_axis1]
                atomsy = slice_positions[:, self.inplane_axis2]
                
                # TODO i'm hard-coding the chunk size is 2000 atoms per layer which is HUGE, so this shouldn't affect anyone but me, but we really ought to do a "smarter" job of picking the chunk size
                chunk_indices = list(np.arange(len(atomsx)))[::2000]+[len(atomsx)]
                shape_factor = xp.zeros( (self.nx,self.ny) , dtype=self.complex_dtype, **device_kwargs )
                for i1,i2 in zip(chunk_indices[:-1],chunk_indices[1:]):
                    atx = atomsx[i1:i2]
                    aty = atomsy[i1:i2]

                    # Compute structure factors - match NumPy pattern exactly
                    # exp(2 i pi (kx * x + ky * y) ) = exp(2 i pi kx x) * exp(2 i pi ky y), summed over all atoms (hence einsum below)
                    expx = xp.exp(-1j * 2 * np.pi * self.kxs[None, :] * atx[:, None])
                    expy = xp.exp(-1j * 2 * np.pi * self.kys[None, :] * aty[:, None])
                
                    # Einstein summation - match NumPy
                    kwarg={True:{},False:{"optimize":True}}[TORCH_AVAILABLE]
                    shape_factor += xp.einsum('ax,ay->xy', expx, expy, **kwarg)
                
                reciprocal += shape_factor * form_factor

            real = xp.fft.ifft2(reciprocal)
            real = xp.real(real)
            # Convert from electron scattering factor f_e to projected potential V_proj.
            # Kirkland Eq C.1: f_e(q) = (m_e/2πℏ²) FT[V], so FT[V] = (2πℏ²/m_e) f_e.
            # On discrete grid: V(r) = (1/A) Σ_k FT[V](k) exp(2πi k·r)
            # = (2πℏ²/m_e) / (dx·dy) × ifft2(S · f_e)
            # where A = Lx·Ly, and ifft2 already includes 1/N = 1/(Lx·Ly)×(dx·dy).
            dx = self.xs[1] - self.xs[0]
            dy = self.ys[1] - self.ys[0]
            fe_to_V = 47.87764737  # 2πℏ²/m_e in V·Å² (Kirkland, Eq C.1 prefactor)
            Z = real * fe_to_V / (dx * dy)
            if cache_file is not None:
                if TORCH_AVAILABLE and hasattr(Z, 'cpu'):
                    Z_cpu = Z.cpu().numpy()
                else:
                    Z_cpu = Z
                np.save(cache_file, Z_cpu)
            return Z
        
        self.calculateSlice = calculateSlice
        #self._array = None
       
    def build(self,progress=False):
        if self._array is not None:
            return
        # Initialize potential array using xp with conditional device
        device_kwargs = {'device': self.device } if self.use_torch else {}
        potential_real = xp.zeros((self.nx, self.ny, self.n_slices), dtype=self.dtype, **device_kwargs)

        if progress:
            localtqdm = tqdm
            print("generating potential for slices")
        else:
            def localtqdm(iterator):
                return iterator

        # cycle through layers generating reciprocal slice
        for slice_idx in localtqdm(range(self.n_slices)):
            potential_real[:, :, slice_idx] += self.calculateSlice(slice_idx)
         
        # Store tensor version for potential GPU operations
        self._array = potential_real

    def flatten(self):
        self._array = mean(self._array,axis=2,keepdims=True)
        self.nz = 1
        self.zs=self.zs[:1]

    @property
    def array(self):
        return to_cpu(self._array)

    def to_cpu(self):
        return to_cpu(self._array)

    def to_device(self, device):
        """Move tensor data to specified device."""
        if hasattr(self, 'array_torch'):
            self.array_torch = self.array_torch.to(device)
        self.device = device
        return self

    def plot(self,filename=None):
        if self._array is None:
            self.build()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        array = np.sum(np.absolute(self.array),axis=2).T[::-1,:] # imshow convention: y,x. our convention: x,y, and flip y (0,0 upper-left)
        # Convert to CPU if on GPU/MPS device
        #if hasattr(array, 'cpu'):
        #    array = array.cpu()

        # Convert extent values to CPU if needed
        xs_min = xp.amin(self.xs)
        xs_max = xp.amax(self.xs)
        ys_min = xp.amin(self.ys)
        ys_max = xp.amax(self.ys)

        if hasattr(xs_min, 'cpu'):
            xs_min = xs_min.cpu()
            xs_max = xs_max.cpu()
            ys_min = ys_min.cpu()
            ys_max = ys_max.cpu()

        extent = (xs_min, xs_max, ys_min, ys_max)
        ax.imshow(array, cmap="inferno", extent=extent)
        ax.set_xlabel("x ($\\AA$)")
        ax.set_ylabel("y ($\\AA$)")

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()
