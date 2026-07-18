import numpy as np
import pytest

from pyslice.backend import Backend, NumpyBackend, TORCH_AVAILABLE, make_backend, to_cpu, to_numpy

if TORCH_AVAILABLE:
    import torch
    from pyslice.backend import TorchBackend


@pytest.fixture(
    params=[
        "numpy",
        pytest.param(
            "torch",
            marks=pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available"),
        ),
    ]
)
def backend(request):
    if request.param == "numpy":
        return NumpyBackend()
    return TorchBackend(device="cpu")


def as_numpy(value):
    return to_numpy(value)


def test_to_cpu_preserves_library_and_to_numpy_materializes_numpy():
    array = np.array([1.0, 2.0])
    assert to_cpu(array) is array
    assert to_cpu(3.14) == pytest.approx(3.14)
    np.testing.assert_array_equal(to_numpy([1, 2, 3]), np.array([1, 2, 3]))

    if TORCH_AVAILABLE:
        tensor = torch.tensor([1.0, 2.0], dtype=torch.float64)
        cpu_tensor = to_cpu(tensor)
        assert isinstance(cpu_tensor, torch.Tensor)
        assert cpu_tensor.device.type == "cpu"

        converted = to_numpy(tensor)
        assert isinstance(converted, np.ndarray)
        np.testing.assert_allclose(converted, np.array([1.0, 2.0]))


def test_make_backend_returns_backend_instance(monkeypatch):
    monkeypatch.setenv("PYSLICE_BACKEND", "numpy")
    assert isinstance(make_backend(), NumpyBackend)

    monkeypatch.delenv("PYSLICE_BACKEND", raising=False)
    assert isinstance(make_backend(device="cpu"), Backend)


def test_backend_array_creation_and_dtype_normalization(backend):
    floats = backend.asarray([1, 2, 3], dtype=float)
    assert as_numpy(floats).dtype in (np.float32, np.float64)

    explicit_floats = backend.asarray([1, 2, 3], dtype=np.float32)
    assert as_numpy(explicit_floats).dtype == np.float32

    complex_values = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
    real_values = backend.asarray(complex_values, dtype=backend.float_dtype)
    np.testing.assert_allclose(as_numpy(real_values), np.array([1.0, 3.0]))

    ints = backend.asarray([1, 2, 3], dtype=int)
    assert np.issubdtype(as_numpy(ints).dtype, np.integer)

    bools = backend.asarray([True, False], dtype=bool)
    assert as_numpy(bools).dtype == np.bool_

    complex_array = backend.zeros((2, 3), dtype="complex")
    assert as_numpy(complex_array).shape == (2, 3)
    assert np.issubdtype(as_numpy(complex_array).dtype, np.complexfloating)

    int_array = backend.zeros((2,), dtype="int")
    assert np.issubdtype(as_numpy(int_array).dtype, np.integer)

    matched = backend.zeros((2,), type_match=complex_array)
    assert np.issubdtype(as_numpy(matched).dtype, np.complexfloating)

    ones = backend.ones((2, 3))
    np.testing.assert_allclose(as_numpy(ones), np.ones((2, 3)))

    freqs = as_numpy(backend.fftfreq(8, d=1.0))
    assert freqs.shape == (8,)
    assert freqs[0] == pytest.approx(0.0)

    random_values = as_numpy(backend.randfloats(20))
    assert random_values.shape == (20,)
    assert np.all((random_values >= 0.0) & (random_values < 1.0))


def test_backend_memmap_creation(backend, tmp_path):
    fallback = backend.memmap((2, 2))
    assert as_numpy(fallback).shape == (2, 2)
    np.testing.assert_allclose(as_numpy(fallback), np.zeros((2, 2)))

    filename = tmp_path / "backend_memmap.npy"
    memmap = backend.memmap((3, 3), dtype=backend.float_dtype, filename=filename)
    assert memmap.shape == (3, 3)
    assert filename.exists()


def test_backend_fft_axes_roundtrip(backend):
    data = backend.asarray(np.random.default_rng(0).random((4, 8, 8)))
    roundtrip = backend.ifft2(backend.fft2(data, axes=(-2, -1)), axes=(-2, -1))
    np.testing.assert_allclose(as_numpy(roundtrip).real, as_numpy(data), atol=1e-10)

    one_d = backend.asarray(np.random.default_rng(1).random((4, 8)))
    one_d_roundtrip = backend.ifft(backend.fft(one_d, axes=1), axes=1)
    np.testing.assert_allclose(as_numpy(one_d_roundtrip).real, as_numpy(one_d), atol=1e-10)


def test_backend_reductions_and_shape_helpers(backend):
    array = backend.asarray(np.arange(6, dtype=float).reshape(2, 3))
    assert float(as_numpy(backend.sum(array))) == pytest.approx(15.0)
    np.testing.assert_allclose(as_numpy(backend.sum(array, axis=0)), np.array([3.0, 5.0, 7.0]))
    assert as_numpy(backend.sum(array, axis=1, keepdims=True)).shape == (2, 1)
    np.testing.assert_allclose(as_numpy(backend.mean(array, axis=1)), np.array([1.0, 4.0]))

    expanded = backend.expand_dims(array, 0)
    assert as_numpy(expanded).shape == (1, 2, 3)

    stacked = backend.stack([array, array], axis=0)
    assert as_numpy(stacked).shape == (2, 2, 3)

    reshaped = backend.reshape(array, (3, 2))
    assert as_numpy(reshaped).shape == (3, 2)

    rolled = backend.roll(backend.asarray([1.0, 2.0, 3.0, 4.0]), 1, axis=0)
    np.testing.assert_allclose(as_numpy(rolled), np.array([4.0, 1.0, 2.0, 3.0]))

    cumsum = backend.cumsum(backend.asarray([1.0, 2.0, 3.0]), axis=0)
    np.testing.assert_allclose(as_numpy(cumsum), np.array([1.0, 3.0, 6.0]))

    truthy = backend.asarray([0.0, 1.0, 0.0])
    falsy = backend.asarray([0.0, 0.0, 0.0])
    assert bool(backend.any(truthy > 0.5))
    assert not bool(backend.any(falsy > 0.5))


def test_backend_copy_and_type_helpers(backend):
    array = backend.asarray([1.0, 2.0, 3.0])
    cast = backend.astype(array, backend.complex_dtype)
    assert np.issubdtype(as_numpy(cast).dtype, np.complexfloating)

    ones = backend.ones_like(array)
    zeros = backend.zeros_like(array)
    np.testing.assert_allclose(as_numpy(ones), np.ones(3))
    np.testing.assert_allclose(as_numpy(zeros), np.zeros(3))

    cloned = backend.clone(array)
    array[0] = 99.0
    assert as_numpy(cloned)[0] == pytest.approx(1.0)


def test_backend_elementwise_math_and_indices(backend):
    values = backend.asarray([-1.0, 2.0, -3.0])
    np.testing.assert_allclose(as_numpy(backend.absolute(values)), np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(as_numpy(backend.sqrt(backend.asarray([4.0, 9.0]))), np.array([2.0, 3.0]))
    np.testing.assert_allclose(as_numpy(backend.exp(backend.asarray([0.0]))), np.array([1.0]))
    np.testing.assert_allclose(as_numpy(backend.log(backend.asarray([1.0, np.e]))), np.array([0.0, 1.0]))
    np.testing.assert_allclose(as_numpy(backend.cos(backend.asarray([0.0]))), np.array([1.0]))

    complex_values = backend.asarray([1 + 2j], dtype=backend.complex_dtype)
    np.testing.assert_allclose(as_numpy(backend.real(complex_values)), np.array([1.0]))
    np.testing.assert_allclose(as_numpy(backend.angle(backend.asarray([1 + 0j], dtype=backend.complex_dtype))), np.array([0.0]))

    nan_mask = as_numpy(backend.isnan(backend.asarray([np.nan, 1.0])))
    assert bool(nan_mask[0])
    assert not bool(nan_mask[1])

    assert backend.ceil(2.3) == 3
    np.testing.assert_array_equal(as_numpy(backend.arange(5)), np.arange(5))
    np.testing.assert_allclose(as_numpy(backend.linspace(0.0, 1.0, num=5)), np.linspace(0.0, 1.0, 5))

    extrema = backend.asarray([3.0, 1.0, 4.0])
    assert float(as_numpy(backend.amin(extrema))) == pytest.approx(1.0)
    assert float(as_numpy(backend.amax(extrema))) == pytest.approx(4.0)
    np.testing.assert_array_equal(as_numpy(backend.argwhere(extrema > 2.0)).ravel(), np.array([0, 2]))

    x = backend.asarray([0.0, 1.0])
    y = backend.asarray([1.0, 0.0])
    np.testing.assert_allclose(as_numpy(backend.arctan2(y, x)), np.array([np.pi / 2, 0.0]))
    assert float(as_numpy(backend.pi)) == pytest.approx(np.pi)


def test_backend_einsum(backend):
    matrix = backend.asarray(np.eye(3))
    vector = backend.asarray([1.0, 2.0, 3.0])
    np.testing.assert_allclose(as_numpy(backend.einsum("ij,j->i", matrix, vector)), np.array([1.0, 2.0, 3.0]))


def test_backend_histogram_is_numpy_backed(backend):
    counts, edges = backend.histogram(
        backend.asarray([0.5, 1.5, 2.5, 3.5]),
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    )
    assert isinstance(counts, np.ndarray)
    assert isinstance(edges, np.ndarray)
    np.testing.assert_array_equal(counts, np.array([1, 1, 1, 1]))


def test_chunk_ids_cover_range():
    chunks = Backend.chunk_ids(10, chunksize=4)
    assert [chunk.tolist() for chunk in chunks] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_torch_backend_respects_cpu_device():
    backend = TorchBackend(device="cpu")
    array = backend.asarray([1.0, 2.0])
    assert isinstance(array, torch.Tensor)
    assert array.device.type == "cpu"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_torch_backend_constructors_use_backend_device():
    backend = TorchBackend(device="cpu")
    assert backend.arange(3).device.type == "cpu"
    assert backend.linspace(0.0, 1.0, 3).device.type == "cpu"
