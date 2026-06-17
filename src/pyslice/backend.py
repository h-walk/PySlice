# backend.py - Backend abstraction layer for NumPy/PyTorch support
from __future__ import annotations

import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Optional torch import
# ---------------------------------------------------------------------------

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Standalone conversion utilities
# These are intentionally module-level: they convert *away* from a backend
# and therefore don't belong to either backend class.
# ---------------------------------------------------------------------------

def to_cpu(x: Any) -> Any:
    """Move a tensor/array to CPU, preserving its array library when possible."""
    if isinstance(x, (np.ndarray, np.memmap)):
        return x
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, (int, float, complex)):
        return x
    return np.asarray(x)


def to_numpy(x: Any) -> np.ndarray:
    """Convert any array-like to a CPU NumPy array."""
    x = to_cpu(x)
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class Backend(ABC):
    """
    Abstract base class defining the array-operation API.

    Subclasses must set:
        xp            - the underlying array library (np or torch)
        float_dtype   - default floating-point dtype
        complex_dtype - default complex dtype
        device        - device identifier (string 'cpu' for NumPy,
                        torch.device for PyTorch)
    """

    xp: Any
    float_dtype: Any
    complex_dtype: Any
    device: Any

    # ------------------------------------------------------------------
    # Private kwarg-translation helpers
    # Written once here; eliminates per-function if/else throughout the module.
    # ------------------------------------------------------------------

    def _axes_kwarg(self, axes: Optional[Any]) -> dict:
        """Translate an axes/dim argument for fft-family functions."""
        if axes is None:
            return {}
        return {'dim': axes} if self.xp is torch else {'axes': axes}

    def _axis_kwarg(self, axis: Optional[int]) -> dict:
        """Translate an axis/dim argument for reduction functions."""
        if axis is None:
            return {}
        return {'dim': axis} if self.xp is torch else {'axis': axis}

    def _keepdim_kwarg(self, keepdims: Optional[bool]) -> dict:
        """Translate a keepdims/keepdim argument."""
        if keepdims is None:
            return {}
        return {'keepdim': keepdims} if self.xp is torch else {'keepdims': keepdims}

    def _is_numpy_array(self, x: Any) -> bool:
        """Return True if x is a plain NumPy array (not a torch tensor)."""
        return isinstance(x, (np.ndarray, np.memmap))

    def _dtype_in(self, dtype: Any, candidates: set) -> bool:
        """Return whether dtype matches a candidate set, accepting np.dtype."""
        if isinstance(dtype, np.dtype):
            dtype = dtype.type
        return dtype in candidates

    # ------------------------------------------------------------------
    # Array creation — depend on device/dtype, so fully abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def asarray(self, arraylike: Any, dtype=None, device=None) -> Any:
        """Convert array-like to a backend array."""

    @abstractmethod
    def zeros(self, dims, dtype=None, device=None, type_match=None) -> Any:
        """Return a zero-filled array."""

    @abstractmethod
    def ones(self, dims, dtype=None, device=None) -> Any:
        """Return a one-filled array."""

    @abstractmethod
    def fftfreq(self, n: int, d: float = 1.0, dtype=None, device=None) -> Any:
        """Return FFT sample frequencies."""

    @abstractmethod
    def randfloats(self, N: int, device=None, dtype=None) -> Any:
        """Return N uniform random floats on [0, 1)."""

    @abstractmethod
    def memmap(self, dims, dtype=None, filename: Optional[str] = None) -> Any:
        """Return a memory-mapped array."""

    # ------------------------------------------------------------------
    # Type utilities
    # ------------------------------------------------------------------

    def astype(self, arraylike: Any, dtype: Any) -> Any:
        """Cast array to dtype."""
        if hasattr(arraylike, 'to'):   # torch tensor
            dtype = self._normalize_dtype(dtype)
            return arraylike.to(dtype)
        return arraylike.astype(dtype) # numpy array

    def _normalize_dtype(self, dtype: Any) -> Any:
        """Translate Python scalar dtypes into backend-native dtypes."""
        if dtype is int:
            return torch.int64 if self.xp is torch else np.int64
        if dtype is float:
            return self.float_dtype
        if dtype is complex:
            return self.complex_dtype
        if dtype is bool:
            return torch.bool if self.xp is torch else np.bool_
        if self.xp is torch:
            numpy_to_torch = {
                np.float32: torch.float32,
                np.dtype("float32"): torch.float32,
                np.float64: torch.float64,
                np.dtype("float64"): torch.float64,
                np.complex64: torch.complex64,
                np.dtype("complex64"): torch.complex64,
                np.complex128: torch.complex128,
                np.dtype("complex128"): torch.complex128,
                np.int32: torch.int32,
                np.dtype("int32"): torch.int32,
                np.int64: torch.int64,
                np.dtype("int64"): torch.int64,
                np.bool_: torch.bool,
                np.dtype("bool"): torch.bool,
            }
            if dtype in numpy_to_torch:
                return numpy_to_torch[dtype]
        return dtype

    def ones_like(self, x: Any) -> Any:
        """Return a one-filled array with the same shape and dtype as x."""
        return self.xp.ones_like(x)

    def zeros_like(self, x: Any) -> Any:
        """Return a zero-filled array with the same shape and dtype as x."""
        return self.xp.zeros_like(x)

    def clone(self, a: Any) -> Any:
        """Return a copy of a."""
        if hasattr(a, 'clone'):
            return a.clone()
        if hasattr(a, 'copy'):
            return a.copy()
        return a

    # ------------------------------------------------------------------
    # FFT family — dispatch via _axes_kwarg
    # ------------------------------------------------------------------

    def fft(self, x: Any, axes=None) -> Any:
        """1-D FFT."""
        if self._is_numpy_array(x):
            kwargs = {} if axes is None else {"axis": axes}
            return np.fft.fft(x, **kwargs)
        return self.xp.fft.fft(x, **self._axis_kwarg(axes))

    def ifft(self, x: Any, axes=None) -> Any:
        """1-D inverse FFT."""
        if self._is_numpy_array(x):
            kwargs = {} if axes is None else {"axis": axes}
            return np.fft.ifft(x, **kwargs)
        return self.xp.fft.ifft(x, **self._axis_kwarg(axes))

    def fft2(self, x: Any, axes=None) -> Any:
        """2-D FFT."""
        if self._is_numpy_array(x):
            kwargs = {} if axes is None else {"axes": axes}
            return np.fft.fft2(x, **kwargs)
        return self.xp.fft.fft2(x, **self._axes_kwarg(axes))

    def ifft2(self, x: Any, axes=None) -> Any:
        """2-D inverse FFT."""
        if self._is_numpy_array(x):
            kwargs = {} if axes is None else {"axes": axes}
            return np.fft.ifft2(x, **kwargs)
        return self.xp.fft.ifft2(x, **self._axes_kwarg(axes))

    def fftshift(self, x: Any, axes=None) -> Any:
        """Shift zero-frequency component to centre."""
        if self._is_numpy_array(x):
            kwargs = {} if axes is None else {"axes": axes}
            return np.fft.fftshift(x, **kwargs)
        return self.xp.fft.fftshift(x, **self._axes_kwarg(axes))

    def ifftshift(self, x: Any, axes=None) -> Any:
        """Inverse of fftshift."""
        if self._is_numpy_array(x):
            kwargs = {} if axes is None else {"axes": axes}
            return np.fft.ifftshift(x, **kwargs)
        return self.xp.fft.ifftshift(x, **self._axes_kwarg(axes))

    # ------------------------------------------------------------------
    # Reductions — dispatch via _axis_kwarg / _keepdim_kwarg
    # ------------------------------------------------------------------

    def sum(self, x: Any, axis: Optional[int] = None,
                   keepdims: Optional[bool] = None) -> Any:
        """Sum elements, optionally along an axis."""
        if self._is_numpy_array(x):
            # Always use numpy for plain numpy arrays even in torch mode
            kw: dict = {} if axis is None else {'axis': axis}
            if keepdims is not None:
                kw['keepdims'] = keepdims
            return np.sum(x, **kw)
        return self.xp.sum(x, **self._axis_kwarg(axis),
                           **self._keepdim_kwarg(keepdims))

    def mean(self, x: Any, axis: Optional[int] = None,
             keepdims: Optional[bool] = None) -> Any:
        """Mean of elements, optionally along an axis."""
        if self._is_numpy_array(x):
            kw = {} if axis is None else {'axis': axis}
            if keepdims is not None:
                kw['keepdims'] = keepdims
            return np.mean(x, **kw)
        return self.xp.mean(x, **self._axis_kwarg(axis),
                            **self._keepdim_kwarg(keepdims))

    def cumsum(self, a: Any, axis: int = 0) -> Any:
        """Cumulative sum along axis."""
        return self.xp.cumsum(a, **self._axis_kwarg(axis))

    def any(self, x: Any) -> Any:
        """Test whether any element is True."""
        return self.xp.any(x)

    # ------------------------------------------------------------------
    # Shape manipulation
    # ------------------------------------------------------------------

    def reshape(self, x: Any, shape) -> Any:
        """Reshape array."""
        if self._is_numpy_array(x):
            return np.reshape(x, shape)
        return self.xp.reshape(x, shape)

    def expand_dims(self, x: Any, axis: int) -> Any:
        """Insert a new axis."""
        if self.xp is torch:
            return self.xp.unsqueeze(x, dim=axis)
        return np.expand_dims(x, axis)

    def stack(self, arrays, axis: int = 0) -> Any:
        """Stack arrays along a new axis."""
        return self.xp.stack(arrays, **self._axis_kwarg(axis))

    def roll(self, x: Any, shift: int, axis: int) -> Any:
        """Roll array elements along an axis."""
        if self.xp is torch:
            return self.xp.roll(x, shifts=shift, dims=axis)
        return np.roll(x, shift, axis=axis)

    def midcrop(self, a: Any, n: int) -> Any:
        """Crop the centre of an unshifted frequency array."""
        return self.xp.roll(self.xp.roll(a, len(a) // 2)[n:-n], len(a) // 2 - n)

    # ------------------------------------------------------------------
    # Elementwise math
    # ------------------------------------------------------------------

    def absolute(self, x: Any) -> Any:
        if self._is_numpy_array(x):
            return np.absolute(x)
        return self.xp.absolute(x)

    def sqrt(self, x: Any) -> Any:
        return self.xp.sqrt(x)

    def exp(self, x: Any) -> Any:
        return self.xp.exp(x)

    def log(self, x: Any) -> Any:
        return self.xp.log(x)

    def real(self, x: Any) -> Any:
        return self.xp.real(x)

    def cos(self, x: Any) -> Any:
        return self.xp.cos(x)

    def arctan2(self, y: Any, x: Any) -> Any:
        return self.xp.arctan2(y, x)

    def angle(self, x: Any) -> Any:
        return self.xp.angle(x)

    def isnan(self, x: Any) -> Any:
        return self.xp.isnan(x)

    def ceil(self, v: Any) -> int:
        if self.xp is torch and TORCH_AVAILABLE and isinstance(v, torch.Tensor):
            return int(self.xp.ceil(v))
        return int(np.ceil(v))

    # ------------------------------------------------------------------
    # Array construction helpers (library-agnostic once xp is known)
    # ------------------------------------------------------------------

    def arange(self, *args, **kwargs) -> Any:
        return self.xp.arange(*args, **kwargs)

    def linspace(self, start, stop, num: int = 50, **kwargs) -> Any:
        return self.xp.linspace(start, stop, num, **kwargs)

    def meshgrid(self, *args, **kwargs) -> Any:
        return self.xp.meshgrid(*args, **kwargs)

    def amin(self, x: Any) -> Any:
        return self.xp.amin(x)

    def amax(self, x: Any) -> Any:
        return self.xp.amax(x)

    def argwhere(self, condition: Any) -> Any:
        return self.xp.argwhere(condition)

    # ------------------------------------------------------------------
    # Einstein summation — falls back to numpy for mixed operands
    # ------------------------------------------------------------------

    def einsum(self, subscripts: str, *operands, **kwargs) -> Any:
        has_numpy = any(self._is_numpy_array(o) for o in operands)
        if self.xp is torch and not has_numpy:
            return self.xp.einsum(subscripts, *operands, **kwargs)
        return np.einsum(subscripts, *[to_numpy(o) for o in operands],
                         optimize=True, **kwargs)

    # ------------------------------------------------------------------
    # Histogram — torch.histogram has CUDA limitations; always use numpy
    # ------------------------------------------------------------------

    def histogram(self, a: Any, bins: Any):
        """Compute histogram. Always falls back to NumPy for device compatibility."""
        return np.histogram(to_numpy(a), bins=to_numpy(bins))

    # ------------------------------------------------------------------
    # Chunked index utility
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_ids(N: int, chunksize: int = 1000):
        """Return a list of numpy index arrays covering [0, N) in chunks."""
        return [np.arange(i, min(i + chunksize, N))
                for i in range(0, N, chunksize)]

    # ------------------------------------------------------------------
    # Convenience properties mirroring the old module-level aliases
    # ------------------------------------------------------------------

    @property
    def pi(self) -> float:
        return self.xp.pi


# ---------------------------------------------------------------------------
# NumpyBackend
# ---------------------------------------------------------------------------

class NumpyBackend(Backend):
    """Pure NumPy backend. No device concept; always CPU float64."""

    xp = np
    float_dtype = np.float64
    complex_dtype = np.complex128
    device = 'cpu'

    _NUMPY_FLOAT_DTYPES = {np.float32, np.float64}
    _NUMPY_COMPLEX_DTYPES = {np.complex64, np.complex128}
    FLOAT_DTYPES = _NUMPY_FLOAT_DTYPES
    COMPLEX_DTYPES = _NUMPY_COMPLEX_DTYPES

    _rng = np.random.default_rng()

    def asarray(self, arraylike: Any, dtype=None, device=None) -> np.ndarray:
        if dtype is None:
            dtype = self.float_dtype
        input_dtype = getattr(arraylike, 'dtype', None)
        if self._dtype_in(dtype, self.FLOAT_DTYPES) and self._dtype_in(input_dtype, self.COMPLEX_DTYPES):
            arraylike = arraylike.real
        return np.asarray(arraylike, dtype=dtype)

    def zeros(self, dims, dtype=None, device=None, type_match=None) -> np.ndarray:
        dtype = self._resolve_dtype(dtype, type_match)
        return np.zeros(dims, dtype=dtype)

    def ones(self, dims, dtype=None, device=None) -> np.ndarray:
        if dtype is None:
            dtype = self.float_dtype
        return np.ones(dims, dtype=dtype)

    def fftfreq(self, n: int, d: float = 1.0, dtype=None, device=None) -> np.ndarray:
        if dtype is None:
            dtype = self.float_dtype
        return np.fft.fftfreq(n, d).astype(dtype)

    def randfloats(self, N: int, device=None, dtype=None) -> np.ndarray:
        if dtype is None:
            dtype = self.float_dtype
        return self._rng.random(int(N)).astype(dtype)

    def memmap(self, dims, dtype=None, filename: Optional[str] = None) -> np.memmap:
        from numpy.lib.format import open_memmap
        if dtype is None:
            dtype = self.float_dtype
        if filename is None:
            logger.warning("memmap called without filename, falling back to zeros")
            return self.zeros(dims, dtype=dtype)
        logger.debug("creating memmap dtype=%s dims=%s filename=%s", dtype, dims, filename)
        return open_memmap(filename, dtype=dtype, mode='w+', shape=dims)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_dtype(self, dtype, type_match):
        if type_match is not None and dtype is None:
            dtype = type_match.dtype
        if dtype is None:
            dtype = self.float_dtype
        if isinstance(dtype, str):
            dtype = {'float': self.float_dtype,
                     'complex': self.complex_dtype,
                     'int': np.int64}[dtype]
        return dtype


# ---------------------------------------------------------------------------
# TorchBackend
# ---------------------------------------------------------------------------

class TorchBackend(Backend):
    """PyTorch backend with automatic device detection."""

    _NUMPY_FLOAT_DTYPES = {np.float32, np.float64}
    _NUMPY_COMPLEX_DTYPES = {np.complex64, np.complex128}

    _TORCH_TO_NUMPY_DTYPE: dict = {
        torch.complex128: np.complex128,
        torch.complex64:  np.complex64,
        torch.float64:    np.float64,
        torch.float32:    np.float32,
    } if TORCH_AVAILABLE else {}

    def __init__(self, device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        self.xp = torch
        self.device, self.float_dtype, self.complex_dtype = \
            self._detect_device_and_precision(device)

        self.FLOAT_DTYPES = (self._NUMPY_FLOAT_DTYPES |
                             {torch.float32, torch.float64})
        self.COMPLEX_DTYPES = (self._NUMPY_COMPLEX_DTYPES |
                               {torch.complex64, torch.complex128})

        logger.debug("TorchBackend initialised: device=%s float_dtype=%s",
                     self.device, self.float_dtype)

    # ------------------------------------------------------------------
    # Device / precision detection (instance-level, not module-level)
    # ------------------------------------------------------------------

    def _detect_device_and_precision(self, device_spec: Optional[str]):
        env_device = os.environ.get('PYSLICE_DEVICE', '').lower()
        if env_device:
            device_spec = env_device

        if device_spec is None:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
        else:
            device = torch.device(device_spec)

        # MPS does not support float64
        if device.type == 'mps':
            float_dtype = torch.float32
            complex_dtype = torch.complex64
        else:
            float_dtype = torch.float64
            complex_dtype = torch.complex128

        return device, float_dtype, complex_dtype

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def asarray(self, arraylike: Any, dtype=None, device=None) -> Any:
        dtype = self._normalize_dtype(dtype)
        if dtype is None:
            dtype = self.float_dtype
        if device is None:
            device = self.device
        input_dtype = getattr(arraylike, 'dtype', None)
        if self._dtype_in(dtype, self.FLOAT_DTYPES) and self._dtype_in(input_dtype, self.COMPLEX_DTYPES):
            arraylike = arraylike.real
        if hasattr(arraylike, 'detach'):  # already a tensor
            return arraylike.detach().clone().to(dtype=dtype, device=device)
        return torch.tensor(arraylike, dtype=dtype, device=device)

    def zeros(self, dims, dtype=None, device=None, type_match=None) -> Any:
        dtype, device = self._resolve_dtype_device(dtype, device, type_match)
        if type_match is not None and self._is_numpy_array(type_match):
            return np.zeros(dims, dtype=self._TORCH_TO_NUMPY_DTYPE.get(dtype, dtype))
        return torch.zeros(dims, dtype=dtype, device=device)

    def ones(self, dims, dtype=None, device=None) -> Any:
        dtype = self._normalize_dtype(dtype)
        if dtype is None:
            dtype = self.float_dtype
        if device is None:
            device = self.device
        return torch.ones(dims, dtype=dtype, device=device)

    def fftfreq(self, n: int, d: float = 1.0, dtype=None, device=None) -> Any:
        dtype = self._normalize_dtype(dtype)
        if dtype is None:
            dtype = self.float_dtype
        if device is None:
            device = self.device
        logger.debug("fftfreq device=%s dtype=%s", device, dtype)
        return torch.fft.fftfreq(n, d, dtype=dtype, device=device)

    def randfloats(self, N: int, device=None, dtype=None) -> Any:
        if device is None:
            device = self.device
        dtype = self._normalize_dtype(dtype)
        if dtype is None:
            dtype = self.float_dtype
        return torch.rand(int(N), device=device, dtype=dtype)

    def arange(self, *args, **kwargs) -> Any:
        kwargs.setdefault("device", self.device)
        if "dtype" in kwargs:
            kwargs["dtype"] = self._normalize_dtype(kwargs["dtype"])
        return torch.arange(*args, **kwargs)

    def linspace(self, start, stop, num: int = 50, **kwargs) -> Any:
        kwargs.setdefault("device", self.device)
        if "dtype" in kwargs:
            kwargs["dtype"] = self._normalize_dtype(kwargs["dtype"])
        return torch.linspace(start, stop, num, **kwargs)

    def memmap(self, dims, dtype=None, filename: Optional[str] = None) -> np.memmap:
        from numpy.lib.format import open_memmap
        if dtype is None:
            dtype = self.float_dtype
        if filename is None:
            logger.warning("memmap called without filename, falling back to zeros")
            return self.zeros(dims, dtype=dtype)
        # memmaps are always numpy; convert torch dtype if needed
        np_dtype = self._TORCH_TO_NUMPY_DTYPE.get(dtype, dtype)
        logger.debug("creating memmap dtype=%s dims=%s filename=%s",
                     np_dtype, dims, filename)
        return open_memmap(filename, dtype=np_dtype, mode='w+', shape=dims)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_dtype_device(self, dtype, device, type_match):
        if type_match is not None:
            if dtype is None:
                dtype = type_match.dtype
            if device is None and hasattr(type_match, 'device'):
                device = type_match.device
        if dtype is None:
            dtype = self.float_dtype
        if device is None:
            device = self.device
        if isinstance(dtype, str):
            dtype = {'float': self.float_dtype,
                     'bool': torch.bool,
                     'complex': self.complex_dtype,
                     'int': torch.int64}[dtype]
        dtype = self._normalize_dtype(dtype)
        return dtype, device


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def make_backend(device: Optional[str] = None) -> Backend:
    """
    Return the appropriate backend based on environment and availability.

    The PYSLICE_BACKEND environment variable can be set to 'numpy' to force
    the NumPy backend regardless of torch availability.

    Args:
        device: Optional device string ('cpu', 'cuda', 'mps').
                Ignored when using the NumPy backend.
                Can also be set via the PYSLICE_DEVICE environment variable.

    Returns:
        A Backend instance (NumpyBackend or TorchBackend).
    """
    backend_override = os.environ.get('PYSLICE_BACKEND', '').lower()
    if backend_override == 'numpy' or not TORCH_AVAILABLE:
        if backend_override == 'numpy' and device is not None:
            logger.warning("device argument ignored for NumpyBackend")
        return NumpyBackend()
    return TorchBackend(device=device)
