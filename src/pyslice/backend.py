"""Lightweight NumPy/Torch compatibility helpers used throughout PySlice.

The module exposes a small ``xp``-style facade so higher-level multislice code
can run with NumPy arrays, PyTorch tensors, or NumPy memmaps without repeating
backend dispatch logic in every kernel.  It is intentionally narrow: functions
only cover operations PySlice currently needs.
"""
import numpy as np
import os
#import torch


def device_and_precision(device_spec=None):
    """Return the active device and default real/complex dtypes.

    MPS uses single precision because Apple Silicon does not support float64
    tensors for the operations PySlice relies on.  CPU and CUDA default to
    float64/complex128.
    """
    
    # We always choose PyTorch if available
    if xp != np:
        if device_spec is None:
            device = DEFAULT_DEVICE
        else: 
            device = xp.device(device_spec)
    else:
        device = None
    
    if device is not None and device.type == 'mps': # Use float32 for MPS (doesn't support float64), float64 for CPU/CUDA
        complex_dtype = xp.complex64
        float_dtype = xp.float32
    else:
        complex_dtype = xp.complex128
        float_dtype = xp.float64
    
    return device, float_dtype, complex_dtype 


try:
    import torch
    xp = torch
    if torch.cuda.is_available():
        config = device_and_precision('cuda')
    elif torch.backends.mps.is_available():
        config = device_and_precision('mps')
    else:
        config = device_and_precision('cpu')
    TORCH_AVAILABLE = True

except ImportError:
    xp = np
    config = device_and_precision()
    TORCH_AVAILABLE = False

DEFAULT_DEVICE, DEFAULT_FLOAT_DTYPE, DEFAULT_COMPLEX_DTYPE = config
del config

# Aliases for convenience
float_dtype = DEFAULT_FLOAT_DTYPE
complex_dtype = DEFAULT_COMPLEX_DTYPE


def configure_backend(device_spec=None, backend_spec=None):
    """
    Configure and return backend settings.

    Args:
        device_spec: Device specification ('cpu', 'cuda', 'mps', or None for auto)
        backend_spec: Backend specification ('numpy', 'torch', or None for auto)

    Returns:
        Tuple of (backend, device, float_dtype, complex_dtype)
    """
    global xp, DEFAULT_DEVICE, DEFAULT_FLOAT_DTYPE, DEFAULT_COMPLEX_DTYPE
    global float_dtype, complex_dtype

    # Determine backend
    if backend_spec == 'numpy':
        backend = np
        device = None
        fdtype = np.float64
        cdtype = np.complex128
    else:
        # Use torch
        backend = torch
        if device_spec is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_spec)

        # Set precision based on device
        if device.type == 'mps':
            fdtype = torch.float32
            cdtype = torch.complex64
        else:
            fdtype = torch.float64
            cdtype = torch.complex128

    return (backend, device, fdtype, cdtype)


def asarray(arraylike, dtype=None, device=None):
    """Convert input data to the active backend array type."""
    if dtype is None:
        dtype = DEFAULT_FLOAT_DTYPE
    if device is None:
        device = DEFAULT_DEVICE
    if xp != np:
#        if dtype == bool:
#            dtype = xp.bool
        array = xp.tensor(arraylike, dtype=dtype, device=device)
    else:
        array = xp.asarray(arraylike, dtype=dtype)
    return array

def astype(arraylike,dtype):
    """Cast a NumPy array or Torch tensor to ``dtype``."""
    if hasattr(arraylike,"to"): # torch
        return arraylike.to(dtype)
    return arraylike.astype(dtype) # numpy

def zeros(dims, dtype=None, device=None, type_match=None):
    """Allocate zeros on the active backend, optionally matching another array."""
    if type_match is not None: # pass an array, and we either infer dtype from the first element, or you also specified a dtype
        if dtype is None:
            dtype = type_match.dtype
        if device is None and hasattr(type_match,"device"):
            device = type_match.device
        if type(type_match) in [ np.memmap, np.ndarray ]:
            return np.zeros(dims,dtype=dtype)
    # default in dtype and device (None in function declaration allows inferring whether it was passed for type_match)
    if dtype is None:
        dtype=DEFAULT_FLOAT_DTYPE
    if device is None:
        device=DEFAULT_DEVICE
    # string handling for dtype, "float" --> float
    if isinstance(dtype,str):
        dtype={"float":DEFAULT_FLOAT_DTYPE,"complex":DEFAULT_COMPLEX_DTYPE,"int":int}[dtype]
    # infer if we're using torch or numpy (numpy does not take device arg)
    if xp != np:
        array = xp.zeros(dims, dtype=dtype, device=device)
    else:
        array = xp.zeros(dims, dtype=dtype)
    return array

def memmap(dims,dtype=DEFAULT_FLOAT_DTYPE,filename=None):
    """Create a NumPy ``.npy`` memmap with backend dtype normalization."""
    from numpy.lib.format import open_memmap
    if filename is None:
        print("WARNING: memmap attempted without filename, falling back to zeros")
        return zeros(dims,dtype)
    # cast to numpy dtypes so we can use numpy memmaps
    if xp != np and dtype in [ xp.complex128, xp.complex64, xp.float64, xp.float32 ]:
        dtype = { xp.complex128:np.complex128, xp.complex64:np.complex64,
                 xp.float64:np.float64, xp.float32:np.float32 }[ dtype ]
    mode = 'w+' #'r+' if os.path.exists(filename) else 'w+'
    #print("creating memmap",mode,dtype,dims,"->",filename)
    return open_memmap(filename, dtype=dtype, mode=mode, shape=dims)

def absolute(array):
    """Return elementwise absolute value for backend arrays or NumPy memmaps."""
    if xp != np and type(array) in [ np.memmap, np.ndarray ]:
        return np.absolute(array)
    return xp.absolute(array)

def reshape(array,shape):
    """Reshape an array, preserving NumPy handling for memmaps."""
    if xp != np and type(array) == np.memmap:
        return np.reshape(array,shape)
    return xp.reshape(array,shape)

def ones(dims, dtype=DEFAULT_FLOAT_DTYPE, device=DEFAULT_DEVICE):
    """Allocate ones on the active backend."""
    if xp != np:
        return xp.ones(dims, dtype=dtype, device=device)
    else:
        return xp.ones(dims, dtype=dtype)

def fftfreq(n, d, dtype=DEFAULT_FLOAT_DTYPE, device=DEFAULT_DEVICE):
    """Return FFT sample frequencies on the active backend."""
    if xp != np:
        return xp.fft.fftfreq(n, d, dtype=dtype, device=device)
    else:
        return xp.fft.fftfreq(n, d, dtype=dtype)

def expand_dims(ary,d):
    """Insert a singleton dimension using backend-native syntax."""
    if xp != np:
        return xp.unsqueeze(ary,dim=d)
    else:
        return np.expand_dims(ary,d)

def exp(x):
    """Return elementwise exponential on the active backend."""
    return xp.exp(x)

def fft(k,**kwargs):
    """One-dimensional FFT with NumPy/Torch keyword normalization."""
    use_torch = TORCH_AVAILABLE
    if type(k) in [ np.memmap, np.ndarray ]:
        use_torch = False
    if use_torch and "axis" in kwargs.keys():
        kwargs["dim"]=kwargs["axis"] ; del kwargs["axis"]
    if not use_torch and "dim" in kwargs.keys():
        kwargs["axis"]=kwargs["dim"] ; del kwargs["dim"]
    if use_torch:
        return xp.fft.fft(k,**kwargs)
    return np.fft.fft(k,**kwargs)

def fftshift(k,**kwargs):
    """Shift zero frequency to the center with NumPy/Torch keyword normalization."""
    use_torch = TORCH_AVAILABLE
    if type(k) in [ np.memmap, np.ndarray ]:
        use_torch = False
    if use_torch and "axes" in kwargs.keys():
        kwargs["dim"]=kwargs["axes"] ; del kwargs["axes"]
    if not use_torch and "dim" in kwargs.keys():
        kwargs["axes"]=kwargs["dim"] ; del kwargs["dim"]
    if use_torch:
        return xp.fft.fftshift(k,**kwargs)
    return np.fft.fftshift(k,**kwargs)

def mean(k,**kwargs):
    """Mean reduction with NumPy/Torch keyword normalization."""
    use_torch = TORCH_AVAILABLE
    if type(k) in [ np.memmap, np.ndarray ]:
        use_torch = False
    if use_torch and "keepdims" in kwargs.keys():
        kwargs["keepdim"]=kwargs["keepdims"] ; del kwargs["keepdims"]
    if not use_torch and "keepdim" in kwargs.keys():
        kwargs["keepdims"]=kwargs["keepdim"] ; del kwargs["keepdim"]
    if use_torch and "axis" in kwargs.keys():
        kwargs["dim"]=kwargs["axis"] ; del kwargs["axis"]
    if not use_torch and "dim" in kwargs.keys():
        kwargs["axis"]=kwargs["dim"] ; del kwargs["dim"]
    if not use_torch:
        return np.mean(k,**kwargs)
    return xp.mean(k,**kwargs)

def ifft2(k):
    """Two-dimensional inverse FFT on the active backend."""
    return xp.fft.ifft2(k)

def real(x):
    """Return the real component of an array."""
    return xp.real(x)

def amax(x):
    """Return the maximum value on the active backend."""
    return xp.amax(x)

def amin(x):
    """Return the minimum value on the active backend."""
    return xp.amin(x)

def sum(x, axis=None, **kwargs):
    """Sum along ``axis`` while translating NumPy/Torch axis keywords."""
    if xp != np and type(x) not in [ np.memmap, np.ndarray ]:
        return xp.sum(x, dim=axis, **kwargs)
    else:
        return np.sum(x, axis=axis, **kwargs)

def any(x):
    """Return whether any elements evaluate true."""
    return xp.any(x)

def einsum(subscripts, *operands, **kwargs):
    """Einstein summation with fallback to NumPy when any operand is NumPy-like."""
    #print([ (type(o),o.dtype) for o in operands])
    numpytypes = [ type(o) in [np.ndarray, np.memmap] for o in operands ]
    if xp != np and True not in numpytypes:
        return xp.einsum(subscripts, *operands, **kwargs)
    else:
        operands = [ to_cpu(o) for o in operands ]
        return np.einsum(subscripts, *operands, optimize=True, **kwargs)

def to_cpu(array):
    """Return a NumPy view/copy of a backend array."""
    if type(array) in [ np.ndarray, np.memmap ]:
        return array
    else:
        return array.cpu().numpy()

def isnan(x):
    """Return elementwise NaN mask."""
    return xp.isnan(x)

def midcrop(a,n): # e.g. unshifted ks: 0,1,2,3,4.....-4,-3,-,2-1, crop out 3 through -3, the inverse of a[n:-n]
    """Crop the middle of an unshifted reciprocal-space vector."""
    return xp.roll(xp.roll(a,len(a)//2)[n:-n],len(a)//2-n)

def ceil(v):
    """Return ``ceil(v)`` as a Python ``int``."""
    if xp != np and type(v)==torch.Tensor:
        return int(xp.ceil(v))
    return int(np.ceil(v))

def cumsum(a):
    """Cumulative sum along the first axis."""
    if xp != np and type(a)==torch.Tensor:
        return xp.cumsum(a,dim=0)
    return xp.cumsum(a)

def histogram(a,bins):
    """Histogram values against explicit bin edges using backend operations."""
    #print(bins)
    #print(bins[1:]-bins[:-1])
    #return np.histogram(to_cpu(a),bins=to_cpu(bins))
    #if xp!=np and type(a)==torch.Tensor: # WHY ARE WE DOING THIS OURSELVES? not-implemented error for torch cuda
    #    #hist = zeros(len(bins)-1,type_match=bins)
    #    #mask = zeros(len(a),type_match=bins)
    #    #for i,(b1,b2) in enumerate(zip(bins[:-1],bins[1:])):
    #    #    mask *= 0
    #    #    mask[a>=b1] = 1 ; mask[a>=b2] = 0
    #    #    hist[i] = xp.sum(mask)
    #    #return hist
    hist = zeros(len(bins)-1,type_match=a)
    for chunk in chunkIDs(len(bins)-1,1000):
        db = bins[chunk+1]-bins[chunk]
        diff = a[None,:]-bins[chunk,None]
        diff[diff>db[:,None]]=-1
        diff[diff<0]=-1
        diff[diff!=-1]=1
        diff[diff==-1]=0
        hist[chunk]=xp.sum(diff,axis=1)
    return hist
    #return np.histogram(to_cpu(a),bins=to_cpu(bins))[0]
    #return np.histogram(a,bins=bins)[0]

def randfloats(N):
    """Return ``N`` random floats from the active backend."""
    N=int(N)
    if xp != np:
        return xp.rand(N)
    return np.random.random(N)

def clone(a):
    """Clone or copy an array-like object when supported."""
    if hasattr(a,"clone"):
        return a.clone()
    try:
        if hasattr(a,"copy"):
            return a.copy()
    except:
        pass
    return a

def chunkIDs(N,chunksize=1000):
    """Return backend index chunks covering ``range(N)``."""
    chunks = [] ; i=0
    while True:
        chunk = xp.arange(i*chunksize,min((i+1)*chunksize,N))
        chunks.append( chunk )
        i += 1
        if i*chunksize >= N:
            break
    return chunks
