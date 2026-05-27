import numpy as np
from tqdm import tqdm
import logging,time
from ..backend import zeros,mean,ones,to_cpu,asarray,absolute,sum,reshape,midcrop,einsum,ceil,clone
#from line_profiler import profile

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



logger = logging.getLogger(__name__)

def antialias_aperture(kxs, kys, cutoff_fraction=2/3, taper_width=0.02):
    """Compute a 2/3 Nyquist anti-aliasing aperture in k-space.

    Bandwidth-limits to ``cutoff_fraction`` of k_max with a smooth cosine
    taper to avoid ringing from a hard cutoff.  Returns a 2D real-valued
    array (1 inside, tapers to 0 outside).
    """
    kx_max = xp.amax(xp.abs(kxs))
    ky_max = xp.amax(xp.abs(kys))
    k_max = min(float(kx_max), float(ky_max))  # Nyquist = 1/(2*sampling)
    k_cutoff = cutoff_fraction * k_max

    kx_grid, ky_grid = xp.meshgrid(kxs, kys, indexing='ij')
    k_r = xp.sqrt(kx_grid**2 + ky_grid**2)

    # Smooth cosine taper from (k_cutoff - taper) to k_cutoff, strictly zero above k_cutoff
    taper = taper_width * k_max
    aperture = xp.ones_like(k_r)
    mask_taper = (k_r > k_cutoff - taper) & (k_r < k_cutoff)
    mask_outer = k_r >= k_cutoff
    if TORCH_AVAILABLE and hasattr(k_r, 'device'):
        aperture[mask_taper] = 0.5 * (1 + torch.cos(torch.pi * (k_r[mask_taper] - k_cutoff + taper) / taper))
    else:
        aperture[mask_taper] = 0.5 * (1 + np.cos(np.pi * (k_r[mask_taper] - k_cutoff + taper) / taper))
    aperture[mask_outer] = 0.0
    return aperture

m_electron = 9.109383e-31    # mass of an electron, kg
q_electron = 1.602177e-19    # charge of an electron, J / eV or kg m^2/s^2 / eV
c_light = 299792458.0        # speed of light, m / s
h_planck = 6.62607015e-34    # m^2 kg / s

#what if we use units of Å throughout, instead of m?
#m_electron = 9.109383e-19   # mass of an electron, nano-gram (ng)
#q_electron = 1.602177e1      # charge of an electron, kg Å^2/s^2 / eV
#c_light = 2.99792458e18      # speed of light, Å / s
#h_planck = 6.62607015e-14    # Å^2 ng / s

def m_effective(eV):
    """Relativistic correction: E=m*c^2, so m=E/c^2, in kg"""
    return m_electron + eV * q_electron / c_light**2
    # units [ kg ]     [ eV ] [ kg m² s⁻² eV⁻¹ ] [ m⁻² s² ]

def wavelength(eV):
    """
    Compute relativistic electron wavelength in Angstroms.

    Uses float64 for intermediate calculations to avoid underflow on MPS/float32.
    The term m_electron * c_light^2 can underflow in float32 if computed in wrong order.
    """
    # Convert to numpy float64 for calculation, then convert back if needed
    if TORCH_AVAILABLE and isinstance(eV, torch.Tensor):
        eV_np = eV.detach().cpu().numpy().astype(np.float64)
        lam_np = h_planck * c_light / ((eV_np * q_electron)**2 + 2 * eV_np * q_electron * m_electron * c_light**2)**0.5 * 1e10
        return torch.tensor(lam_np, dtype=eV.dtype, device=eV.device)
    else:
        return h_planck * c_light / ((eV * q_electron)**2 + 2 * eV * q_electron * m_electron * c_light**2)**0.5 * 1e10

class Probe:
    """
    PyTorch-accelerated probe class for electron microscopy.
    
    Generates probe wavefunctions on GPU for both plane wave and convergent beam modes.
    Significant speedup for large grid sizes through GPU-accelerated FFT operations.
    """
    
    def __init__(self, xs, ys, mrad, eV, array=None, device=None, gaussianVOA=0, preview=False, probe_xs=None, probe_ys=None, probe_positions=None, cropping=False, defer_shifts=False, stay_reciprocal = False, crop_reciprocal=False):
        """
        Initialize GPU-accelerated probe wavefunction.
        
        Args:
            xs, ys: Real space coordinate arrays
            mrad: Convergence semi-angle in milliradians (0.0 = plane wave)
            eV: Electron energy in eV
            device: PyTorch device (None for auto-detection)
        """
        # TORCH DEVICES AND DTYPES
        if TORCH_AVAILABLE:
            # Auto-detect device if not specified (same logic as Potential class)
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

            # Use float32 for MPS compatibility (same as Potential class)
            self.dtype = torch.float32 if device.type == 'mps' else torch.float64
            self.complex_dtype = torch.complex64 if device.type == 'mps' else torch.complex128
        else:
            if device is not None:
                raise ImportError("PyTorch not available. Please install PyTorch.")
            self.device = None
            self.use_torch = False
            self.dtype = np.float64
            self.complex_dtype = np.complex128
        
        # SET UP SPATIAL GRIDS
        # Convert coordinate arrays to tensors if using torch (same as Potential class)
        if self.use_torch:
            # Use as_tensor to avoid copy warning when input is already a tensor
            self.xs = torch.as_tensor(xs, dtype=self.dtype, device=self.device)
            self.ys = torch.as_tensor(ys, dtype=self.dtype, device=self.device)
        else:
            self.xs = xs
            self.ys = ys

        nx = len(xs) ; ny = len(ys)
        dx = xs[1] - xs[0] ; dy = ys[1] - ys[0]
        lx = nx*dx ; ly = ny*dy
        self.nx = nx ; self.dx = dx ; self.lx = lx
        self.ny = ny ; self.dy = dy ; self.ly = ly

        # HANDLE PROBE POSTIONS
        self.probe_xs = probe_xs
        self.probe_ys = probe_ys
        self.probe_positions = probe_positions

        # Preferred to pass probe_xs and probe_ys from which we will define a grid. copied from probe_grid (now defunct)
        if self.probe_xs is not None and self.probe_ys is not None:
            x,y = np.meshgrid(self.probe_xs,self.probe_ys)
            self.probe_positions = np.reshape([x,y],(2,len(x.flat))).T

        # Set up default probe position if not provided
        if self.probe_positions is None:
            self.probe_positions = [(lx/2, ly/2)]  # Center probe
            self.probe_xs = [lx/2] ; self.probe_ys = [ly/2]

        # HANDLE BEAM PARAMS
        self.mrad = mrad
        #if isinstance(eV,(float,int)):
        #    n = 1 if array is None else len(array)
        #    eV = [ eV ]*n
        self.eV = eV ; self.wavelength=wavelength(eV)
        self.eVs = np.asarray([eV])
        if self.use_torch:
            self.eVs = torch.as_tensor(self.eVs, dtype=self.dtype, device=self.device)
        self.wavelengths = wavelength(self.eVs)
        self.temporal_decoherence = None
        self.spatial_decoherence = None
        self.gaussianVOA = gaussianVOA
        
        # Set up device kwargs for unified xp interface (same as Potential class)
        device_kwargs = {'device': self.device, 'dtype': self.dtype} if self.use_torch else {}
        
        self.stay_reciprocal = stay_reciprocal
        self.crop_reciprocal = crop_reciprocal
        #if self.crop_reciprocal: # user asked for kspace to be, say, 100 pixels, but right now it's 350...
        #    self.crop_reciprocal = (min(nx,ny)-self.crop_reciprocal)//2 # so we need to chop 125 off each side


        self.kxs = xp.fft.fftfreq(nx, d=dx, **device_kwargs)
        self.kys = xp.fft.fftfreq(ny, d=dy, **device_kwargs)

        if not array is None: # Allow construction of a Probe object with a passed array instead of building it below. used by create_batched_probes
            if self.use_torch and hasattr(array, 'to'):
                self._array = array.to(device=self.device, dtype=self.complex_dtype)
            else:
                self._array = xp.asarray(array)
        else:
            #self._array = zeros((len(self.eV),1,nx,ny))
            #for i,w in enumerate(self.wavelength):
            #   self._array[i,0,:,:] = self.generate_single_probe(mrad,w,gaussianVOA,preview=preview)
            #self._array = zeros((1,1,nx,ny),dtype=complex_dtype)
            self._array= self.generate_single_probe(mrad,self.wavelength,self.gaussianVOA,preview=preview)[None,None,:,:]*ones((1,1), device=self.device)[:,:,None,None]

        self.cropping = cropping
        #self.offsets = [[0,0] for i in range(len(self.probe_positions)) ]
        self.offsets = np.zeros((len(self.probe_positions),2),dtype="int") # these are used when we have cropped the probe

        # NEW PHILOSOPHY: we used to build out the probe cube (npt,nx,ny) no matter what, but if you have
        # a bajillion probes, then this cube might be huge! instead, callers (e.g. calculator) pass
        # defer_shifts=True and call applyShifts when ready. This means calculators' loop_probes can
        # handle them one at a time, without building out the entire cube.
        if not defer_shifts:
            self.applyShifts()

    def generate_single_probe(self,mrad,wavelength,gaussianVOA,preview=False):
        kxs,kys = self.kxs,self.kys
        if self.crop_reciprocal:           # unshifted kx ky: 0,1,2,3,....-3,-2,-1, midcrop gets rid of high-k: 0,1,2,-2,-1
            kxs = midcrop(self.kxs,self.crop_reciprocal[0])
            kys = midcrop(self.kys,self.crop_reciprocal[1])

        nx,ny = len(kxs) , len(kys)
        if mrad == 0:
            return zeros((nx, ny), device=self.device)+1

        radius = (mrad * 1e-3) / wavelength  # Convert mrad to reciprocal space units

        reciprocal = zeros((nx, ny), device=self.device)
        kx_grid, ky_grid = xp.meshgrid(kxs, kys, indexing='ij') # unshifted kx ky: 0,1,2,3,....-3,-2,-1
        radii = xp.sqrt(kx_grid**2 + ky_grid**2)

        if gaussianVOA == 0:
            mask = radii < radius
            reciprocal[mask] = 1.0          # mask covers the corners (unshifted in reciprocal space)
        else:
            from scipy.special import erf
            reciprocal = 1-erf((radii-radius)/(gaussianVOA*radius))

        if preview:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots() ; print(radius)
            extent = (xp.min(self.kxs), xp.max(self.kxs), xp.min(self.kys), xp.max(self.kys))
            ax.imshow(xp.fft.fftshift(reciprocal.T), cmap="inferno",extent=extent) # shift to visualize with k=0 in the center
            ax.set_xlabel("kx ($\\AA^{-1}$)")
            ax.set_ylabel("ky ($\\AA^{-1}$)")
            plt.show()

        if self.stay_reciprocal: # if we would've done a real-space shift, we should apply a phase ramp in reciprocal space
            return reciprocal * xp.exp(-2j * xp.pi * kxs[:, None] * self.lx/2 ) * xp.exp(-2j * xp.pi * kys[None, :] * self.ly/2 )

        return xp.fft.ifftshift(xp.fft.ifft2(reciprocal)) # iFFT --> realspace --> shift --> zero in the center
        #self.array_numpy = self.array.cpu().numpy()
    
    def copy(self,selected_probes=None):
        """Create a deep copy of the probe."""
        new_probe = Probe.__new__(Probe)
        for attr in self.__dict__.keys():
            if attr[0]=="_" or "array" in attr:
                continue
            val = getattr(self,attr)
            val = clone(val)
            setattr(new_probe,attr,val)
        if selected_probes is not None:
            selected_probes = to_cpu(selected_probes)
            nc,npt,nx,ny = self._array.shape
            if npt == 1:
                new_probe._array = clone(self._array[:,:,:,:])
            else:
                new_probe._array = clone(self._array[:,selected_probes,:,:])
            new_probe.offsets = self.offsets[selected_probes,:]
            new_probe.probe_positions = self.probe_positions[to_cpu(selected_probes),:]
            #print("new",new_probe.offsets.shape,new_probe.probe_positions.shape)
        else:
            new_probe._array = clone(self._array)
            #print("no selected used")
        #new_probe.device = self.device
        #new_probe.array_numpy = self.array_numpy.copy()
        return new_probe

    @property
    def array(self):
        return self.to_cpu()
    
    def to_cpu(self):
        """Convert probe array to CPU NumPy array."""
        if hasattr(self._array, 'cpu'):
            return self._array.cpu().numpy()
        return self._array
    
    def to_device(self, device):
        """Move probe to specified device (similar to Potential.to_device)."""
        if not self.use_torch:
            raise RuntimeError("to_device() requires PyTorch")

        # MPS doesn't support float64
        if hasattr(device, 'type') and device.type == 'mps':
            dtype, complex_dtype = torch.float32, torch.complex64
        else:
            dtype, complex_dtype = torch.float64, torch.complex128

        self._array = self._array.to(device=device, dtype=complex_dtype)
        self.xs = self.xs.to(device=device, dtype=dtype)
        self.ys = self.ys.to(device=device, dtype=dtype)
        self.kxs = self.kxs.to(device=device, dtype=dtype)
        self.kys = self.kys.to(device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype
        self.complex_dtype = complex_dtype
        return self

    def plot(self,filename=None,title=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # calling self.array should convert to CPU/numpy
        array = np.mean(np.absolute(self.array[:,:,::-1,:]),axis=1)[0,:,:] # positional,summable,x,y indices
        array=array.T # imshow convention: y,x. our convention: x,y
        plot_array = np.absolute(array)**.25

        # Convert extent values to CPU if needed (use xp for torch/numpy compatibility)
        xs_min = xp.amin(self.xs)
        xs_max = xp.amax(self.xs) # TODO technically this should be xs[-1]+dx??
        ys_min = xp.amin(self.ys)
        ys_max = xp.amax(self.ys)

        if hasattr(xs_min, 'cpu'):
            xs_min = xs_min.cpu()
            xs_max = xs_max.cpu()
            ys_min = ys_min.cpu()
            ys_max = ys_max.cpu()

        extent = (xs_min, xs_max, ys_min, ys_max)
        ax.imshow(plot_array, cmap="inferno",extent=extent)
        ax.set_xlabel("x ($\\AA$)")
        ax.set_ylabel("y ($\\AA$)")
        if title is not None:
            ax.set_title(title)

        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    def defocus(self,dz): # POSITIVE DEFOCUS PUTS BEAM WAIST ABOVE SAMPLE, UNITS OF ANGSTROM
        if isinstance(dz,(int,float)):
            dz = zeros(len(self._array), device=self.device)+dz
        kx_grid, ky_grid = xp.meshgrid(self.kxs, self.kys, indexing='ij')
        k_squared = kx_grid**2 + ky_grid**2
        P = xp.exp(-1j * xp.pi * self.wavelength * dz[:,None,None] * k_squared[None,:,:])
        nz = len(dz) ; nc,npt,nx,ny = self._array.shape #; print("nc,npt,nx,ny",nc,npt,nx,ny)
        self._array = xp.fft.ifft2( P[:,None,None,:,:] * xp.fft.fft2( self._array )[None,:,:,:,:] )
        self._array = self._array.reshape((nz*nc,npt,nx,ny))
        #print("defocus",dz,"new shape",self._array.shape)

    # ORDER OF OPERATIONS IS IMPORTANT. MUST DO: addTemporalDecoherence, addSpatialDecoherence, create_batched_probes
    # addTemporalDecoherence - creates new standard probes (must come first)
    # addSpatialDecoherence - applies defocus (applies to existing probe(s))
    # create_batched_probes - applied shift to each probe
    def addTemporalDecoherence(self,sigma_eV,N):
        nc,npt,nx,ny = self._array.shape #; print("addTemporalDecoherence shape was",nc,npt,nx,ny)
        if self.temporal_decoherence is not None:
            print("WARNING: calling addTemporalDecoherence twice will overwrite previous")
        self.temporal_decoherence = (sigma_eV,N)
        eV = self.eV
        self.eVs = np.linspace(eV-2*sigma_eV,eV+2*sigma_eV,N)
        if self.use_torch:
            self.eVs = torch.as_tensor(self.eVs, dtype=self.dtype, device=self.device)
        self.wavelengths = wavelength(self.eVs)
        amplitudes = np.exp(-(eV-self.eVs)**2/sigma_eV**2)
        self._array = zeros((N,1,nx,ny), device=self.device)
        for n,eV in enumerate(self.eVs):
            self._array[n,0,:,:] = amplitudes[n] * self.generate_single_probe(self.mrad,wavelength(eV),self.gaussianVOA)
        nc,npt,nx,ny = self._array.shape #; print("addTemporalDecoherence expands to",nc,npt,nx,ny)
        if self.spatial_decoherence is not None:
            self.addSpatialDecoherence(*self.spatial_decoherence)
        nc,npt,nx,ny = self._array.shape
        if npt==1:
            self.applyShifts()

    def addSpatialDecoherence(self,sigma_dz,N):
        nc,npt,nx,ny = self._array.shape #; print("addSpatialDecoherence shape was",nc,npt,nx,ny)
        if self.temporal_decoherence is not None:
            print("WARNING: calling addSpatialDecoherence twice will overwrite previous")
        self.spatial_decoherence = (sigma_dz,N)
        dzs = np.linspace(-2*sigma_dz,2*sigma_dz,N) # suppose N=25
        amplitudes = np.exp(-dzs**2/sigma_dz**2)
        nc,npt,nx,ny = self._array.shape            # suppose nc=10 (addTemporalDecoherence created 10 wavelengths)
        if self.use_torch:
            dzs = torch.as_tensor(dzs, dtype=self.dtype, device=self.device)
        self.defocus(dzs)                           # defocus starts with 25,10,npt,nx,ny --reshapes--> 250,npt,nx,ny
        for i in range(N):                         # reshape to flatten loops first index last: [[0,1],[2,3]] --> [0,1,2,3]
            for j in range(nc):
                self._array[i*nc+j] *= amplitudes[i]
        nc,npt,nx,ny = self._array.shape #; print("addSpatialDecoherence expands to",nc,npt,nx,ny)
        self.eVs = ones(N, device=self.device)[:,None]*self.eVs[None,:] # defocus expands into nz,nc then flattens to nz*nc
        self.eVs = self.eVs.reshape(nc)
        self.wavelengths = ones(N, device=self.device)[:,None]*self.wavelengths[None,:]
        self.wavelengths = self.wavelengths.reshape(nc)
        if npt==1:
            self.applyShifts()

    def applyShifts(self):
        nc,npt,nx,ny = self._array.shape #; print("applyShifts shape was",nc,npt,nx,ny,"len(self.probe_positions)",len(self.probe_positions))
        if npt>1: # TODO ALSO NEED SOMETHING TO DETERMINE IF SHIFTS HAVE ALREADY BEEN APPLIED. EG A LIST WHICH IS ALWAYS UPDATED WHEN ARRAY IS RESET?
            return

        # inflate self._array to store probe cube (npt,nx,ny)
        if self.cropping:
            i1=nx//2-self.cropping//2 ; i2=i1+self.cropping     # |_______i1___.___i2_______| for initial centered probe at lx/2,ly/2
            j1=ny//2-self.cropping//2 ; j2=j1+self.cropping
            self._array = self._array[:,0,None,i1:i2,j1:j2] * ones(len(self.probe_positions), device=self.device)[None,:,None,None]
        else:
            self._array = self._array[:,0,None,:,:] * ones(len(self.probe_positions), device=self.device)[None,:,None,None]
        # loop through probe positions
        for i, (px,py) in enumerate(self.probe_positions):
            if px-self.lx/2 == 0 and py-self.ly/2 == 0:
                    continue

            self._array[:,i,:,:],(dpx,dpy) = self.placeProbe(self._array[:,i,:,:], px, py )
            self.offsets[i,0] = int(dpx) ; self.offsets[i,1] = int(dpy)

        nc,npt,nx,ny = self._array.shape #; print("applyShifts expands to",nc,npt,nx,ny)

    def placeProbe(self,array,x,y):
        dx = (x-self.lx/2) ; dy = (y-self.ly/2)                 # probe started in the center
        if self.cropping:
            i1=self.nx//2-self.cropping//2 ; i2=i1+self.cropping  # |_______i1___.___i2_______| for initial centered probe at lx/2,ly/2
            j1=self.ny//2-self.cropping//2 ; j2=j1+self.cropping
            device_kwargs = {'device': self.device, 'dtype': self.dtype} if self.use_torch else {}
            kxs = xp.fft.fftfreq(self.cropping, d=self.dx, **device_kwargs)
            kys = xp.fft.fftfreq(self.cropping, d=self.dy, **device_kwargs)
            dpx = dx//self.dx ; dpy = dy//self.dy               # pixel shifts
            offset_x = i1+dpx ; offset_y = j1+dpy
            dx-=dpx*self.dx ; dy-=dpy*self.dy                   # update subpixel shifts
        elif self.crop_reciprocal:           # unshifted kx ky: 0,1,2,3,....-3,-2,-1, midcrop gets rid of high-k: 0,1,2,-2,-1
            kxs = midcrop(self.kxs,self.crop_reciprocal[0])
            kys = midcrop(self.kys,self.crop_reciprocal[1])
            offset_x = 0 ; offset_y=0
        else:
            kxs,kys=self.kxs,self.kys
            offset_x = 0 ; offset_y=0
        if not self.stay_reciprocal:
            probe_k = xp.fft.fft2(array) # positional,summable,x,y
        else:
            probe_k = array
        kx_shift = xp.exp(-2j * xp.pi * kxs[None,:, None] * dx )
        ky_shift = xp.exp(-2j * xp.pi * kys[None,None, :] * dy )
        probe_k_shifted = probe_k * kx_shift * ky_shift

        if self.stay_reciprocal:
             return probe_k_shifted,(offset_x,offset_y)
        return xp.fft.ifft2(probe_k_shifted),(offset_x,offset_y)


    def aberrate(self,aberrations):
        dP = aberrationFunction(self.kxs,self.kys,self.wavelength,aberrations)
        # recall, in Probe.__init__, we created the real-space array via:
        # self.array = xp.fft.ifftshift(xp.fft.ifft2(reciprocal))
        # Aberrations are defined at the aperture plane, so we must apply them in reciprocal space. 
	    # (or do a convolution in real-space)
        reciprocal = xp.fft.fft2(xp.fft.fftshift(self._array)) # centered-real --> zero at corner --> FFT --> kx,ky zero at corner
        reciprocal *= dP
        self._array = xp.fft.ifftshift(xp.fft.ifft2(reciprocal))

# See Kirkland Eq 2.10: 
# χ(k,ϕ) = π/2 Cs λ³ k⁴ - π Δf λ k²
# + π fa2 λ k² sin(2*(ϕ-ϕa2)) + 2π/3 fa3 λ² k³ sin(3*(ϕ-ϕa3)) 
# + 2π/3 fc3 λ² k³ sin(ϕ-ϕc3)
# where fa is astig, fc is coma, with orientations ϕa or ϕc
# or https://doi-org.ornl.idm.oclc.org/10.1016/S0304-3991(99)00013-3 Eq. A1
# χ(rᵤ,rᵥ) = 2π/λ { C₁₀ ( rᵤ² + rᵥ² ) / 2
# + C₁₂ᵤ ( rᵤ² - rᵥ² ) / 2 + C₁₂ᵥ rᵤ rᵥ
# + C₂₁ᵤ rᵤ ( rᵤ² + rᵥ² ) / 3 + C₂₁ᵥ rᵥ ( rᵤ² + rᵥ² ) / 3 
# + C₂₃ᵤ rᵤ ( rᵤ² - 3 rᵥ² ) / 3 + C₂₃ᵥ rᵥ ( 3 rᵤ² - rᵥ² ) / 3
# + C₃₀ ( rᵤ² + rᵥ² )² / 4
# + C₃₂ᵤ ( rᵤ⁴ - rᵥ⁴ ) / 4 + C₃₂ᵥ 2 rᵤ rᵥ ( rᵤ² + rᵥ² ) / 4
# + C₃₄ᵤ ( rᵤ⁴ - 6 rᵤ² rᵥ² + rᵥ² ) / 4 + C₃₄ᵥ ( rᵤ³ rᵥ - rᵤ rᵥ³ ) } 
# where rᵤ = r cos(ϕ) and rᵥ = r sin(ϕ), r² = rᵤ² + rᵥ²
# χ(rᵤ,rᵥ) = 2π/λ { C₁₀ r² / 2
# + C₁₂ᵤ ( rᵤ² - rᵥ² ) / 2 + C₁₂ᵥ rᵤ rᵥ
# + C₂₁ᵤ rᵤ r² / 3 + C₂₁ᵥ rᵥ r² / 3 
# + C₂₃ᵤ rᵤ ( rᵤ² - 3 rᵥ² ) / 3 + C₂₃ᵥ rᵥ ( 3 rᵤ² - rᵥ² ) / 3
# + C₃₀ r⁴ / 4
# + C₃₂ᵤ ( rᵤ⁴ - rᵥ⁴ ) / 4 + C₃₂ᵥ 2 rᵤ rᵥ ( rᵤ² + rᵥ² ) / 4
# + C₃₄ᵤ ( rᵤ⁴ - 6 rᵤ² rᵥ² + rᵥ² ) / 4 + C₃₄ᵥ ( rᵤ³ rᵥ - rᵤ rᵥ³ ) } 
# or https://doi-org.ornl.idm.oclc.org/10.1016/j.ultramic.2010.04.006 Eq A1
# χ(u,v) = C₀₁ u + C₀₁ v
# + 1/2 [ C₁₀ ( u² + v² ) + C₁₂ᵤ ( u² - v² ) + 2 C₁₂ᵥ u v ]
# +1/3 [ C₂₃ᵤ
# ₀₁₂₃₄ᵤᵥ
# or comparing to: https://abtem.readthedocs.io/en/latest/user_guide/walkthrough/contrast_transfer_function.html
# χ(k,ϕ) = π/2/λ 1/(n+1) C ( k λ )^(n+1) cos(m*(ϕ-ϕa))
# Aberrations are an adjustment to the phase of the wave ("dPhi"), to be applied in reciprocal space.
# this is done by multiplying the complex wave (be it a probe or an exit wave) by xp.exp(-1j * dPhi)
def aberrationFunction(kxs,kys,wavelength,aberrations): # aberrations should be a dict of Cnm following https://abtem.readthedocs.io/en/latest/user_guide/walkthrough/contrast_transfer_function.html
    dPhi = xp.zeros_like(kxs[:,None] * kys[None,:])
    ks = xp.sqrt( kxs[:,None]**2 + kys[None,:]**2 ) # unshifted: 0,1,2,3,...-3,-2,-1, reciprocal origin at corner
    theta = xp.arctan2( kys[None,:] , kxs[:,None] )
    for k in aberrations.keys():
        n,m = int(k[1]),int(k[2]) # C03 --> 0,3
        C = aberrations[k] ; phi0 = 0
        if not isinstance(C,(int,float)):
            C,phi0 = C
        dPhi += 2*xp.pi/wavelength * \
            (1/(n+1)) * C * ( ks * wavelength ) ** (n+1) * \
            xp.cos( m * (theta-phi0) )
    return xp.exp(-1j * dPhi)

#def probe_grid(xlims,ylims,n,m):
#	x,y=np.meshgrid(np.linspace(*xlims,n),np.linspace(*ylims,m))
#	return np.reshape([x,y],(2,len(x.flat))).T


def create_batched_probes(base_probe, probe_positions, device=None):
    """
    Create a batch of shifted probes for vectorized processing.

    Args:
        base_probe: ProbeTorch object
        probe_positions: List of (x,y) positions
        device: PyTorch device

    Returns:
        probe object with an array of shape (n_probes, nx, ny)
    """
    # Move probe to correct device if needed
    if device is not None and TORCH_AVAILABLE:
        # Always move to ensure array is actually on the device
        # (checking base_probe.device may not reflect actual array device)
        base_probe.to_device(device)

    n_probes = len(probe_positions)
    probe_arrays = []

    nx = len(base_probe.xs)
    ny = len(base_probe.ys)

    # Compute dx, dy on same device as probe
    if TORCH_AVAILABLE and hasattr(base_probe.xs, 'device'):
        dx = (base_probe.xs[1] - base_probe.xs[0]).item()
        dy = (base_probe.ys[1] - base_probe.ys[0]).item()
    else:
        dx = base_probe.xs[1] - base_probe.xs[0]
        dy = base_probe.ys[1] - base_probe.ys[0]

    lx = nx*dx ; ly = ny*dy

    for px, py in probe_positions:
        # Create shifted probe using phase ramp in k-space
        probe_k = xp.fft.fft2(base_probe._array)

        # Apply phase ramp for spatial shift (negative sign = shift right)
        kx_shift = xp.exp(-2j * xp.pi * base_probe.kxs[:, None] * (px-lx/2) )
        ky_shift = xp.exp(-2j * xp.pi * base_probe.kys[None, :] * (py-ly/2) )
        probe_k_shifted = probe_k * kx_shift * ky_shift
        
        # Convert back to real space
        shifted_probe_array = xp.fft.ifft2(probe_k_shifted)
        probe_arrays.append(shifted_probe_array)
    
    # Stack into batch tensor
    if TORCH_AVAILABLE:
        array = torch.stack(probe_arrays, dim=0)
    else:
        array = xp.asarray(probe_arrays)

    return Probe(base_probe.xs, base_probe.ys, base_probe.mrad, base_probe.eV, array=array, device=base_probe.device)


class PrismProbe:
    """
    Where Probe object creates a series of real-space probes (n,nx,ny cube, for n probe positions), the Prism algorithm propagates a series of sinusoids (fourier components shared by all real-space probes), then reconstructs each probe's exit wave.

    PrismProbe object should serve as a stand-in for Probe, meaning it can be propagated through a potential (via the Propagate function), the probe cube (n,nx,ny) generated via self.applyShifts, and a subset of probes selected via self.copy, which enables chunked processing. where Probe.probe_positions stores real-space x,y pairs for positions, PrismProbe stores reciprocal-space kx,ky pairs to denote the sinusoid.

    """
    def __init__(self, xs, ys, mrad, eV, array=None, device=None, gaussianVOA=0, preview=False, nkx = 25, nky=None, kth=1):

        # TORCH DEVICES AND DTYPES
        if TORCH_AVAILABLE:
            # Auto-detect device if not specified (same logic as Potential class)
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
            # Use float32 for MPS compatibility (same as Potential class)
            self.dtype = torch.float32 if device.type == 'mps' else torch.float64
            self.complex_dtype = torch.complex64 if device.type == 'mps' else torch.complex128
        else:
            if device is not None:
                raise ImportError("PyTorch not available. Please install PyTorch.")
            self.device = None
            self.use_torch = False
            self.dtype = np.float64
            self.complex_dtype = np.complex128

        # FULL-SIZED STUFF: USED FOR INTERACTING WITH THE POTENTIAL AND ALL THAT, real-space and reciprocal-space
        self.dx = xs[1]-xs[0] ; self.dy = ys[1]-ys[0]
        self.nx = len(xs) ; self.ny = len(ys)
        # Convert coordinate arrays to tensors if using torch (same as Potential class)
        if self.use_torch:
            # Use as_tensor to avoid copy warning when input is already a tensor
            self.xs = torch.as_tensor(xs, dtype=self.dtype, device=self.device)
            self.ys = torch.as_tensor(ys, dtype=self.dtype, device=self.device)
        else:
            self.xs = xs
            self.ys = ys
        device_kwargs = {'device': self.device, 'dtype': self.dtype} if self.use_torch else {}
        self.kxs = xp.fft.fftshift(xp.fft.fftfreq(self.nx, d=self.dx, **device_kwargs)) # # 0,1,2,3...-3,-2.-1 -shift-> ...-3,-2,-1,0,1,2,3...
        self.kys = xp.fft.fftshift(xp.fft.fftfreq(self.ny, d=self.dy, **device_kwargs))
        self._array = zeros((1,1,self.nx,self.ny),dtype=self.complex_dtype)
        # SPARSIFIED STUFF, USED FOR CONSTRUCTING SPARSE SINUSOIDS IN REAL SPACE
        if nky is None:
            nky = nkx
        self.nx_cropped = nkx ; self.ny_cropped = nky # indices for cropping i1,i2,j1,j2

        self.i1 = self.nx//2-self.nx_cropped//2 #; self.i2 = self.i1+self.nx_cropped
        self.j1 = self.ny//2-self.ny_cropped//2 #; self.j2 = self.j1+self.ny_cropped
        self.nx_cropped  = self.nx - 2*self.i1 ; self.ny_cropped  = self.ny - 2*self.j1
        self.probe_positions=zeros((self.nx_cropped,self.ny_cropped,2))
        for i,kx in enumerate(self.kxs[self.i1:self.nx-self.i1]):           # looping across a sparsified k-grid
            for j,ky in enumerate(self.kys[self.j1:self.ny-self.j1]):
                self.probe_positions[i,j,0]=kx
                self.probe_positions[i,j,1]=ky
        self.probe_positions = reshape(self.probe_positions,(self.nx_cropped*self.ny_cropped,2))

        # HANDLE BEAM PARAMS (copied from Probe just in case anyone asks for them)
        self.mrad = mrad
        self.eV = eV ; self.wavelength=wavelength(eV)
        self.eVs = np.asarray([eV])
        if self.use_torch:
            self.eVs = torch.as_tensor(self.eVs, dtype=self.dtype, device=self.device)
        self.wavelengths = wavelength(self.eVs)
        self.temporal_decoherence = None
        self.spatial_decoherence = None
        self.gaussianVOA = gaussianVOA
        self.cropping = False
        self.kth = kth

    # where Probe.applyShifts looks at real-space x,y pairs in probe_positions and applies a phase ramp to shift a template probe, PrismProbe.applyShifts looks at reciprocal-space kx,ky pairs in probe_positions to construct sinusoids
    def applyShifts(self):
        # inflate self._array to store probe cube (npt,nx,ny)
        self._array = self._array[:,0,None,:,:] * ones(len(self.probe_positions), device=self.device)[None,:,None,None]
        # loop through probe positions
        for n,(kx,ky) in enumerate(self.probe_positions):
            self._array[:,n,:,:] = xp.exp(2j * xp.pi * self.xs[:, None] * kx ) * xp.exp(2j * xp.pi * self.ys[None,:] * ky )
            # numpy appears to use exp(i2pixk) convention for FFT: xs = np.linspace(0,100,10000) ; ys = np.sin(xs) ; fft = np.fft.fft(ys) ; freq=np.fft.fftfreq(len(xs),d=xs[1]-xs[0]) ; fft2 = np.sum(ys[:,None]*np.exp(2j*np.pi*xs[:,None]*freq[None,:]),axis=0)

    # if a PrismProbe object (a whole bunch of sinusoidal entrance waves) is propagated through a potential, then the potential exit waves for a whole bunch of realistic probes can be calculated from the exit waves for each entrance wave
    # SHIFTING: array is shifted, factors is NOT shifted,
    #@profile
    def calculateProbesFromS(self,array,positions,chunksize=100,load_into=None,ADF=False): # array comes in p,x,y,l,1 where p is our 50*50 grid of sinusoids
        if load_into is None and not ADF:
            result = zeros((len(positions),ceil(self.nx/self.kth),ceil(self.ny/self.kth)),dtype="complex") # full-res kx,ky for each probe position
        elif not ADF:
            result = load_into
        else:
            ADF,ADFmask,ADFindex = ADF ; result = None
        npt,nkx,nky,_,_ = array.shape
        array = reshape(array,(self.nx_cropped,self.ny_cropped,nkx,nky)) # eikx,eiky,kx,ky
        # preview an arbitrary exit wave? (note: calculator will have done shift(fft(realspace)), so we should invert those steps)
        #import matplotlib.pyplot as plt
        #fig, ax = plt.subplots()
        #i,j=self.nx_cropped//2+1,self.ny_cropped//2-1
        #ax.imshow(np.real(np.fft.ifft2(np.fft.ifftshift(to_cpu(array[i,j,:,:])))).T, cmap="inferno")
        #plt.show()
        # strategy: we need each probe's array (generated, shifted to position), FFT'd, cropped, so we can select fourier components
        # you can 1) create a dummy probe, call generate_single_probe and placeProbe, and reuse it every time, and this can be done in real or reciprocal space. or 2) you can just create a new probe each time (for a chunk of positions). i'm choosing 2 because it makes chunking easier
        #probe = Probe(self.xs, self.ys, self.mrad, self.eV, defer_shifts=True) # dummy probe so we can directly access Probe class functions
        chunksize=max(1,chunksize) # handle 0 as chunksize
        for n,(x,y) in enumerate(tqdm(positions)):
            if n%chunksize!=0:
                continue
            # strategy 1, real-space
            #ary = probe.generate_single_probe(self.mrad,self.wavelength,self.gaussianVOA,preview=False)
            #probe_k = xp.fft.fftshift(xp.fft.fft2(probe.placeProbe(ary,x,y)[0][0,:,:])) # shiff(fft()) to match what calculators did to array
            # strategy 2, keep things in reciprocal space, saves two ffts?
            # realspace single probe was xp.fft.ifftshift(xp.fft.ifft2(reciprocal))
            # placed was xp.fft.fft2(array), then phase ramp, then xp.fft.ifft2(array)
            #ary = probe.generate_single_probe(self.mrad,self.wavelength,self.gaussianVOA,preview=False,keep_reciprocal=True)
            #probe_k = xp.fft.fftshift(probe.placeProbe(ary,x,y,realspace=False)[0][0,:,:])
            # strategy 3, stack of probes
            #probes = Probe(self.xs, self.ys, self.mrad, self.eV, probe_positions = positions[n:n+chunksize])
            #kwarg = {"dim":(-2,-1)} if TORCH_AVAILABLE else {"axes":(-2,-1)}
            #probe_ks = xp.fft.fftshift(xp.fft.fft2(probes._array[0,:,:,:],**kwarg),**kwarg)

            # note you CAN generate the probe pre-cropped (use arg crop_reciprocal, then skip the self.i1:-self.i1 indexing), but this doesn't seem to save a whole lot of time...
            probes = Probe(self.xs, self.ys, self.mrad, self.eV, probe_positions = positions[n:n+chunksize],stay_reciprocal=True)#,crop_reciprocal=(self.i1,self.j1))
            kwarg = {"dim":(-2,-1)} if TORCH_AVAILABLE else {"axes":(-2,-1)}
            probe_ks = xp.fft.fftshift(probes._array[0,:,:,:],**kwarg)

            # fourier components of FFT'd and cropped probe are the contribution of each exit wave
            factors = probe_ks[:,self.i1:self.nx-self.i1,self.j1:self.ny-self.j1] # this is unshifted since ij_lookup used unshifted kxs,kys
            #factors = probe_ks#[:,self.i1:-self.i1,self.j1:-self.j1] # this is unshifted since ij_lookup used unshifted kxs,kys
            chunked = einsum('pkq,kqxy->pxy',factors,array) # sum over all sinusoids
            if isinstance(result,np.memmap):
                chunked = to_cpu(chunked)
            if ADF:
                intensities = einsum('pxy,xy->p',absolute(chunked)**2,ADFmask)
                for i,pp in zip(intensities,range(n,n+chunksize)):
                    ADF._array[ADFindex==pp] += i
            else:
                result[n:n+chunksize,:,:] = chunked

            #dx = (x-probe.lx/2) ; dy = (y-probe.ly/2)
            #if abs(dx)<.1 and abs(dy)<.1:
            #    print("\nposition",x,y)
            #    print("dx",dx,dy)
            #    import matplotlib.pyplot as plt
            #    fig, ax = plt.subplots()
            #    ax.imshow(np.real(probe_ks[0]).T, cmap="inferno")
            #    plt.show()
            # preview our sparse-k reconstructed probe? fft --> downsample --> ifft
            #if n<=len(positions)//3<n+chunksize:
            #    print("plotting reconstructed probe for",x,y)
            #    probe_r = xp.fft.ifft2(xp.fft.ifftshift(factors[0,:,:]))
            #    import matplotlib.pyplot as plt
            #    fig, ax = plt.subplots()
            #    extent = (xp.min(self.xs), xp.max(self.xs), xp.min(self.ys), xp.max(self.ys))
            #    ax.imshow(to_cpu(xp.real(probe_r)).T[::-1,:], cmap="inferno",extent=extent)
            #    plt.show()
            # result from this probe is it's downsampled fourier component scaling/phase term, multiplied by each fourier component's raw exit
            #factors = xp.fft.fftshift(factors) # array will have been shift(fft())'d in MultisliceCalculator

        return result

    def copy(self,selected_probes=None):
        #print("creating copy",selected_probes)
        """Create a deep copy of the probe."""
        new_probe = PrismProbe.__new__(PrismProbe)
        for attr in self.__dict__.keys():
            if attr[0]=="_" or "array" in attr:
                continue
            val = getattr(self,attr)
            val = clone(val)
            setattr(new_probe,attr,val)
        if selected_probes is not None:
            nc,npt,nx,ny = self._array.shape
            if npt == 1:
                new_probe._array = clone(self._array[:,:,:,:])
            else:
                new_probe._array = clone(self._array[:,selected_probes,:,:])
            #new_probe.offsets = self.offsets[selected_probes,:]
            new_probe.probe_positions = self.probe_positions[selected_probes,:]
            #print("new",new_probe.offsets.shape,new_probe.probe_positions.shape)
        else:
            new_probe._array = clone(self._array)
            #print("no selected used")
        #new_probe.device = self.device
        #new_probe.array_numpy = self.array_numpy.copy()
        return new_probe

    @property
    def array(self):
        return to_cpu(self._array)


# Given a real-space entrance wave, and a potential (or object), calculate the exit wave: ψ₁ -> O -> ψ₂
# From Kirkland2010:
# propagator P = exp(-i π λ dz q²), Eq 6.65
# transmission function t = exp(i σ O) where O is our object (or potential slice), Eq 6.59
# ψ₂ = ℱ⁻¹[ P * ℱ[ t * ψ₁ ] ], Eq 6.67, noting the relationship: 𝒞[ f(x),g(x) ] = ℱ⁻¹[ ℱ[f(x)] * ℱ[g(x)] ] = ℱ⁻¹[ f(k) * g(k) ]
# or as code: array = t * array ; fft_array = fft(array) ; propagated_fft = P * fft_array ; array = ifft(propagated_fft)
def Propagate(
    probe,
    potential,
    device=None,
    progress=False,
    onthefly=True,
    store_all_slices=False,
    propagate_exit=False,
    stored_slice_indices=None,
):
    """
    PyTorch-accelerated multislice propagation function.
    Supports both single probe and batched multi-probe processing.

    Args:
        probe: ProbeTorch object or tensor with shape (n_probes, nx, ny)
        potential: Potential object (can be NumPy or PyTorch version)
        device: PyTorch device (None for auto-detection)
        progress: Show progress bar
        onthefly: If True, calculate potential slices on the fly. If False, build full array
        store_all_slices: If True, return wavefunction at each slice instead of just exit wave
        propagate_exit: If True, apply a final free-space propagation after the last slice
        stored_slice_indices: Optional 0-based slice indices to return when store_all_slices=True

    Returns:
        torch.Tensor: Exit wavefunction(s) after multislice propagation
                     If store_all_slices=True, shape is (stored_slices, n_probes, nx, ny)
                     Otherwise, shape is (n_probes, nx, ny) or (nx, ny) for single probe
    """
    if device is not None and not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Please install PyTorch.")
    if device is None and hasattr(probe, 'device'):
        device = probe.device

    # Initialize wavefunction with probe(s) - shape: (n_probes, nx, ny)
    nc,npt,nx,ny = probe._array.shape #; print("nc,npt,nx,ny",nc,npt,nx,ny)
    array = probe._array.reshape((nc*npt,nx,ny)) # "flatten" first two indices
    probe_wavelengths = probe.wavelengths[:,None]*ones(npt, dtype=probe.dtype, device=device)[None,:] # also expand wavelengths and eVs arrays to cover all probe positions npt
    probe_wavelengths = probe_wavelengths.reshape(nc*npt)
    probe_eVs = probe.eVs[:,None]*ones(npt, dtype=probe.dtype, device=device)[None,:]
    probe_eVs = probe_eVs.reshape(nc*npt)

    # Calculate interaction parameter (Kirkland Eq 5.6)
    E0_eV = m_electron * c_light**2 / q_electron
    sigma = (2 * np.pi) / (probe_wavelengths * probe_eVs) * \
            (E0_eV + probe_eVs) / (2 * E0_eV + probe_eVs) # wavelength and eVs now have length of n_probes
    #print("propagate sigma",sigma)
    #if TORCH_AVAILABLE:
    #    #sigma_dtype = torch.float32 if device.type == 'mps' else torch.float64
    #    sigma = torch.tensor(sigma, dtype=float_dtype, device=device)
    
    # Get slice thickness
    dz = potential.zs[1] - potential.zs[0] if len(potential.zs) > 1 else 0.5
    
    # Pre-compute propagation operator in k-space (Fresnel propagation)
    # All tensors should already be on the correct device from creation
    kx,ky = potential.kxs, potential.kys
    if probe.cropping:
        device_kwargs = {'device': probe.device, 'dtype': probe.dtype} if probe.use_torch else {}
        kx = xp.fft.fftfreq(probe.cropping, d=probe.dx, **device_kwargs)
        ky = xp.fft.fftfreq(probe.cropping, d=probe.dy, **device_kwargs)
    kx_grid, ky_grid = xp.meshgrid(kx, ky, indexing='ij')
    k_squared = kx_grid**2 + ky_grid**2

    # Precompute 2/3 Nyquist anti-aliasing aperture for the propagated wavefunction
    aa_aperture = antialias_aperture(kx, ky)

    # Fold anti-aliasing aperture into propagator to bandwidth-limit wavefunction at every slice
    P = xp.exp(-1j * xp.pi * probe_wavelengths[:,None,None] * dz * k_squared[None,:,:]) * aa_aperture[None,:,:] # Kirkland2010 Eq 6.65

    #print("(done)",time.time()-start)


    if progress:
        localtqdm = tqdm
        print("propagating through slices")
    else:
        def localtqdm(iterator):
            return iterator

    if not onthefly:
        potential.build()

    if store_all_slices and stored_slice_indices is not None:
        stored_slice_indices = set(int(i) for i in stored_slice_indices)
        invalid_slices = [i for i in stored_slice_indices if i < 0 or i >= len(potential.zs)]
        if invalid_slices:
            raise ValueError(
                f"stored_slice_indices contains out-of-range slices {sorted(invalid_slices)}; "
                f"valid range is [0, {len(potential.zs) - 1}]"
            )

    slice_wavefunctions = [] if store_all_slices else None

    # Vectorized multislice propagation through each slice
    for z in localtqdm(range(len(potential.zs))):
        # Transmission function: t = exp(iσV(x,y,z))
        # All tensors should already be on the correct device from creation
        if onthefly:
            #print("calculateSlice") ; start = time.time()
            potential_slice = potential.calculateSlice(z)
            #print("(done)",time.time()-start)
        else:
            potential_slice = potential._array[:, :, z]

        if probe.cropping:
            #t = zeros( (len(sigma), probe.cropping, probe.cropping ), type_match=P)
            #print("probe rolling") ; start = time.time() 
            # Usually you want vectorized, but this inflates to full npt,nx,ny before cropping: bad on RAM
            #rollx = list(probe.offsets[:,0]) ; z = [1]*len(rollx)
            #pot = xp.roll(potential_slice[None,:,:],rollx,z)[:,:probe.cropping,:]
            #rolly = list(probe.offsets[:,1]) ; z = [2]*len(rolly)
            #pot = xp.roll(pot[:,:,:],rolly,z)[:,:,:probe.cropping]
            nx,ny = potential_slice.shape
            #xr = xp.arange(nx) ; yr = xp.arange(ny)
            #pot_stack = zeros( (len(sigma),probe.cropping,probe.cropping) )
            #for p,o in enumerate(probe.offsets): # We want to go from i1,j2 to i1+cropping,j1+cropping, but sometimes i1 or j1 is negatuve
            #    # rolling the whole thing: slow (5s per on Nick's particles)
            #    #pot = xp.roll(potential_slice,int(-o[0]),0)[:probe.cropping,:]
            #    #pot = xp.roll(pot,int(-o[1]),1)[:,:probe.cropping]
            #    # rolling indices: faster (0.7s per on Nick's particles)
            #    xi = xp.roll(xr,-o[0])[:probe.cropping]
            #    yi = xp.roll(yr,-o[1])[:probe.cropping]
            #    pot_stack[p] = potential_slice[xi,:][:,yi]
            # full indexing is even faster
            xi = xp.zeros((len(sigma),probe.cropping),dtype=int) ; yi = xp.zeros((len(sigma),probe.cropping),dtype=int)
            xr = xp.arange(nx) ; yr = xp.arange(ny)
            for p,(x,y) in enumerate(probe.offsets):
                xi[p,:] = xp.roll(xr,-x)[:probe.cropping]
                yi[p,:] = xp.roll(yr,-y)[:probe.cropping]
            pot_stack=potential_slice[xi[:,:,None],yi[:,None,:]]
            #print("(done)",time.time()-start)
            t=xp.exp(1j*sigma[:,None,None]*pot_stack)

        else:
            t = xp.exp(1j * sigma[:,None,None] * potential_slice[None,:,:]) # Kirkland2010 Eq 6.59. n,x,y indices

        # Apply transmission to all probes: ψ' = t × ψ
        # Broadcasting: t[n_probes,nx,ny] * array[n_probes,nx,ny] = array[n_probes,nx,ny]
        array = t * array

        # Store wavefunction at this slice if requested (after transmission)
        if store_all_slices and (
            stored_slice_indices is None or z in stored_slice_indices
        ):
            # Clone/copy to avoid reference issues
            slice_wavefunctions.append(clone(array))

        # Fresnel propagation to next slice (except for last slice)
        if z < len(potential.zs) - 1 or propagate_exit:
            # Vectorized FFT over spatial dimensions for all probes
            kwarg = {"dim":(-2,-1)} if TORCH_AVAILABLE else {"axes":(-2,-1)}
            #print(kwarg,array.dtype,array.shape)
            #print("FFT / multiply / iFFT") ; start = time.time()
            fft_array = xp.fft.fft2(array, **kwarg)
            propagated_fft = P * fft_array
            array = xp.fft.ifft2(propagated_fft, **kwarg)
            #print("(done)",time.time()-start)

    # Return results based on what was requested
    if store_all_slices:
        # Stack the list into a tensor with slices as a new dimension
        # Shape will be (n_slices, n_probes, nx, ny) - more conventional ordering
        if TORCH_AVAILABLE:
            return torch.stack(slice_wavefunctions, dim=0)
        else:
            return xp.stack(slice_wavefunctions, axis=0)

    #array = array.reshape((nc,npt,nx,ny))

    # Return single probe result if input was single, otherwise return batch
    #if array.shape[0] == 1:
    #    return array.squeeze(0)
    return array # okay for Propagate to return a Tensor. we probably don't want to move things off-gpu yet

# Given a real-space entrance and real-space exit wave, calculate the object the wave must have passed through: ψ₁ -> O -> ψ₂, given ψ₁,ψ₂, find O
# From Kirkland2010:
# propagator P = exp(-i π λ dz q²), Eq 6.65
# transmission function t = exp(i σ O) where O is our object (or potential slice), Eq 6.59
# ψ₂ = ℱ⁻¹[ P * ℱ[ t * ψ₁ ] ], Eq 6.67, noting the relationship: 𝒞[ f(x),g(x) ] = ℱ⁻¹[ ℱ[f(x)] * ℱ[g(x)] ] = ℱ⁻¹[ f(k) * g(k) ]
# or as code: array = t * array ; fft_array = fft(array) ; propagated_fft = P * fft_array ; array = ifft(propagated_fft)
# SO, to determine O from ψ₁ and ψ₂:
# ℱ[ ψ₂ ] = P * ℱ[ t * ψ₁ ]
# ℱ[ ψ₂ ]/P = ℱ[ t * ψ₁ ]
# ℱ⁻¹[ ℱ[ ψ₂ ]/P ] = t * ψ₁
# t = ℱ⁻¹[ ℱ[ ψ₂ ]/P ]/ψ₁
# exp(i σ O) = ℱ⁻¹[ ℱ[ ψ₂ ]/P ]/ψ₁
# i σ O = log( ℱ⁻¹[ ℱ[ ψ₂ ]/P ]/ψ₁ )
# O = log( ℱ⁻¹[ ℱ[ ψ₂ ]/P ]/ψ₁ ) / i / σ
def calculateObject(probe,exitwave,guessedObject,weighting=.5,dz=0.5,damping=.01):

    import matplotlib.pyplot as plt
    #fig, axs = plt.subplots(1,2)
    #axs[0].imshow(to_cpu(xp.absolute(exitwave)), cmap="inferno") ; axs[0].set_title("exit wave")
    #axs[1].imshow(to_cpu(xp.absolute(probe._array[0,0,:,:])), cmap="inferno") ; axs[1].set_title("entrance wave")
    #plt.show()

    nc,npt,nx,ny = probe._array.shape #; print("nc,npt,nx,ny",nc,npt,nx,ny)
    psi1 = probe._array[0,0,:,:] # select first probe array only for now!
    lamda = probe.wavelengths[0]
    eV = probe.eVs[0]

    # Calculate interaction parameter (Kirkland Eq 5.6)
    E0_eV = m_electron * c_light**2 / q_electron
    sigma = (2 * np.pi) / (lamda * eV) * \
            (E0_eV + eV) / (2 * E0_eV + eV) # wavelength and eVs now have length of n_probes

    # Pre-compute propagation operator in k-space (Fresnel propagation)
    # All tensors should already be on the correct device from creation
    kx_grid, ky_grid = xp.meshgrid(probe.kxs, probe.kys, indexing='ij') # TODO use probe.kxs instead, to free ourselves of the need to pass a potential
    k_squared = kx_grid**2 + ky_grid**2

    P = xp.exp(-1j * xp.pi * lamda * dz * k_squared[:,:]) # P in Kirkland2010 Eq 6.65 is exp(-i...), so to divide by P, we can use Pp, exp(+j...)

    # t = ℱ⁻¹[ ℱ[ ψ₂ ]/P ]/ψ₁
    t = xp.fft.ifft2(xp.fft.fft2(exitwave)/P)/psi1
    # t = exp(i σ O) --> O = log(t)/i/σ
    O = xp.log(t)/1j/sigma
    # WHAT IS exp(i σ O) REALLY DOING? exp(iϕ) is a sinusoid. an only-real object is applying a phase shift? can we do calculate an angle instead?
    O = xp.angle(t)/sigma
    # consider:
    # instead of division, should i multiply by complex conjugate? (not technically the same, but it will deal with near-zeros)
    # instead of log, should i use: https://en.wikipedia.org/wiki/Complex_logarithm#Calculating_the_principal_value
    # should I apply a probe amplitude masking function?

    #fig, axs = plt.subplots(1,3)
    #axs[0].imshow(to_cpu(xp.absolute(exitwave)), cmap="inferno") ; axs[0].set_title("exit wave")
    #axs[1].imshow(to_cpu(xp.absolute(probe._array[0,0,:,:])), cmap="inferno") ; axs[1].set_title("entrance wave")
    #axs[2].imshow(to_cpu(xp.absolute(O)**.1), cmap="inferno") ; axs[2].set_title("object")
    #plt.show()

    delta=(O-asarray(guessedObject))

    # probe amplitude masking function: zeros-out points where probe intensity is zero, without giving probe features as features in your potential
    delta*=xp.absolute(psi1)/(xp.absolute(psi1)+damping*xp.amax(xp.absolute(psi1)))

    return delta*weighting
