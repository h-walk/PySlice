"""Numerical and allocation contracts for low-memory multislice optimizations."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyslice.backend import NumpyBackend, TORCH_AVAILABLE, to_numpy
from pyslice.multislice import calculators as cm
from pyslice.multislice import multislice as mm
from pyslice.multislice import potentials as pm
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.multislice.trajectory import Trajectory


@pytest.fixture(params=['numpy', 'torch', 'torch32', 'mps', 'cuda'])
def backend(request, monkeypatch):
    """Use the same backend and precision for reference and optimized paths."""
    monkeypatch.delenv('PYSLICE_DEVICE', raising=False)
    if request.param == 'numpy':
        return NumpyBackend()
    if not TORCH_AVAILABLE:
        pytest.skip('Torch unavailable')
    import torch
    from pyslice.backend import TorchBackend
    device = request.param if request.param in ('mps', 'cuda') else 'cpu'
    if device == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('Metal GPU unavailable')
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA GPU unavailable')
    result = TorchBackend(device=device)
    if request.param == 'torch32':
        result.float_dtype, result.complex_dtype = torch.float32, torch.complex64
    assert result.zeros(1).device.type == device
    return result


def test_kirkland_coefficients_match_original_parser():
    """Single-read parsing retains all 103 original coefficient tables."""
    expected = []
    for i in range(103):
        values = np.loadtxt(pm.kirkland_file, skiprows=i * 4 + 1, max_rows=3).ravel()
        expected.append([[values[2*j], values[2*j+1], values[6+2*j], values[7+2*j]]
                         for j in range(3)])
    np.testing.assert_array_equal(pm.load_kirkland(NumpyBackend()), expected)


def test_kirkland_resource_read_once_and_returned_arrays_are_independent(backend, monkeypatch):
    """No per-frame text reads or mutable arrays shared across callers/devices."""
    source = pm.kirkland_file.read_text()
    calls = []

    class Resource:
        """Count actual resource reads without modifying the bundled file."""

        def read_text(self):
            """Return the original table text."""
            calls.append(1)
            return source

    monkeypatch.setattr(pm, 'kirkland_file', Resource())
    first = pm.load_kirkland(backend)
    expected = to_numpy(first).copy()
    first[:] = 0
    second = pm.load_kirkland(backend)
    cpu = pm.load_kirkland(NumpyBackend())
    assert len(calls) == 1
    np.testing.assert_array_equal(to_numpy(second), expected)
    np.testing.assert_allclose(cpu, expected)
    assert not pm._read_kirkland_table(pm.kirkland_file).flags.writeable


@pytest.mark.parametrize('symbols', [False, True])
@pytest.mark.parametrize('zs', [np.array([0.]), np.linspace(0., 4., 13, endpoint=False),
                               np.array([0., .3, .59999999, 1.05, 1.3])])
def test_slice_indices_preserve_wrapping_boundaries_and_atom_order(backend, symbols, zs):
    """Indexing preserves half-open bounds, roundoff gaps, and chunk order."""
    rng = np.random.default_rng(12)
    positions = rng.random((100, 3)) * [3., 3., 6.]
    # Exercise first/last slices, out-of-cell atoms, and near-boundary values.
    positions[:10, 2] = [-10., -0.01, 0., .15, .15-1e-8, .15+1e-8,
                         4., 4.01, 1.2, 1.5]
    types = np.where(np.arange(100) % 2, 'Si', 'C') if symbols else np.arange(100) % 2 + 6
    potential = pm.Potential(np.arange(5.) * .6, np.arange(7.) * .5, zs,
                             positions, types, backend=backend, chunk_size=3)
    coords = np.mod(potential._positions[:, 2], potential.slice_period)
    for z in range(potential.n_slices):
        lo, hi = potential._slice_bounds(z)
        for at in potential._unique_types:
            expected = np.flatnonzero((types == at) & (coords >= lo) & (coords < hi))
            np.testing.assert_array_equal(potential._slice_atom_indices[z][at], expected)


def _reference_slice(potential, z):
    """Evaluate the previous atom-masking and chunked summation algorithm.

    Parameters
    ----------
    potential : Potential
        Source geometry, positions, and form factors.
    z : int
        Slice index.

    Returns
    -------
    array_like
        Reference real-space potential on the active backend.
    """
    b = potential._backend
    reciprocal = b.zeros((potential.nx, potential.ny), dtype=b.complex_dtype)
    lo, hi = potential._slice_bounds(z)
    for at in potential._unique_types:
        mask = (np.asarray([t == at for t in potential._atom_types])
                if isinstance(at, str) else potential._atom_z_np == int(at))
        positions = potential._positions[mask]
        coords = np.mod(positions[:, 2], potential.slice_period)
        positions = positions[(coords >= lo) & (coords < hi)]
        if not len(positions):
            continue
        x, y = b.asarray(positions[:, 0]), b.asarray(positions[:, 1])
        shape = b.zeros((potential.nx, potential.ny), dtype=b.complex_dtype)
        for start in range(0, len(x), potential._chunk_size):
            end = start + potential._chunk_size
            expx = b.exp(-2j * np.pi * potential.kxs[None, :] * x[start:end, None])
            expy = b.exp(-2j * np.pi * potential.kys[None, :] * y[start:end, None])
            shape += b.einsum('ax,ay->xy', expx, expy)
        reciprocal += shape * potential._form_factors[at]
    return b.real(b.ifft2(reciprocal)) * pm._FE_TO_V / (potential.dx * potential.dy)


@pytest.mark.parametrize('empty', [False, True])
def test_indexed_potentials_match_reference_without_per_slice_host_downloads(backend, monkeypatch, empty):
    """Values stay identical and atom arrays are not downloaded in the slice loop."""
    rng = np.random.default_rng(2)
    positions = rng.random((0 if empty else 40, 3)) * [3., 3., 2.]
    types = np.where(np.arange(len(positions)) % 2, 'Si', 'C')
    potential = pm.Potential(np.arange(5.) * .6, np.arange(7.) * .5,
                             np.arange(5.) * .4, positions, types,
                             backend=backend, chunk_size=3)
    expected = [_reference_slice(potential, z) for z in range(5)]

    def no_host_download(*args):
        """Fail if slice evaluation transfers an atom/grid array to the host."""
        raise AssertionError('unexpected per-slice host download')

    monkeypatch.setattr(pm, 'to_numpy', no_host_download)
    for z in range(5):
        np.testing.assert_array_equal(to_numpy(potential._calculate_slice(z)), to_numpy(expected[z]))
    assert potential.array is None  # no eager full-volume storage


def _reference_propagate(probe, potential, layers):
    """Evaluate the old flattened, per-scan-position operator representation.

    Parameters
    ----------
    probe : object
        Probe with two independent energy copies and arbitrary scan positions.
    potential : Potential
        Fixed potential array.
    layers : list[int] | None
        Stored slice indices, or only the final exit wave when None.

    Returns
    -------
    array_like
        Flattened copy-major exit wave or layer stack.
    """
    b = potential._backend
    nc, npt, nx, ny = probe._array.shape
    array = b.reshape(probe._array, (nc*npt, nx, ny))
    wavelengths = b.reshape(probe.wavelengths[:, None] * b.ones(npt)[None, :], (-1,))
    eVs = b.reshape(probe.eVs[:, None] * b.ones(npt)[None, :], (-1,))
    e0 = mm.m_electron * mm.c_light**2 / mm.q_electron
    sigma = 2*b.pi / (wavelengths * eVs) * (e0 + eVs) / (2*e0 + eVs)
    kx, ky = potential.kxs, potential.kys
    if probe.cropping:
        kx, ky = b.fftfreq(nx, probe.dx), b.fftfreq(ny, probe.dy)
    ksq = kx[:, None]**2 + ky[None, :]**2
    dz = float(to_numpy(potential.zs[1] - potential.zs[0])) if len(potential.zs) > 1 else .5
    propagator = b.exp(-1j*b.pi*wavelengths[:, None, None]*dz*ksq) * mm.antialias_aperture(kx, ky, b)
    saved = []
    for z in range(potential.n_slices):
        plane = potential.array[:, :, z]
        if probe.cropping:
            windows = []
            for i in range(nc*npt):
                ox, oy = to_numpy(probe.offsets)[i % npt]
                xi = b.asarray(np.roll(np.arange(potential.nx), -int(ox))[:nx], dtype=int)
                yi = b.asarray(np.roll(np.arange(potential.ny), -int(oy))[:ny], dtype=int)
                windows.append(plane[xi[:, None], yi[None, :]])
            transmission = b.exp(1j*sigma[:, None, None]*b.stack(windows, axis=0))
        else:
            transmission = b.exp(1j*sigma[:, None, None]*plane[None])
        array = transmission * array
        if layers is not None and z in layers:
            saved.append(b.clone(array))
        if z < potential.n_slices - 1:
            array = b.ifft2(propagator * b.fft2(array, axes=(-2, -1)), axes=(-2, -1))
    return array if layers is None else b.stack(saved, axis=0)


@pytest.mark.parametrize('cropped', [False, True])
@pytest.mark.parametrize('layers', [None, [0, 3], [0, 1, 2, 3]])
def test_shared_operators_and_crop_indices_match_flat_reference(backend, monkeypatch, cropped, layers):
    """Sharing preserves copy order, every requested layer, and periodic windows."""
    b = backend
    rng = np.random.default_rng(31)
    potential = pm.Potential(np.arange(9.) * .3, np.arange(7.) * .4,
                             np.arange(4.) * .5, backend=b,
                             array=rng.normal(size=(9, 7, 4)))
    nx, ny = (5, 5) if cropped else (9, 7)
    probe = SimpleNamespace(
        _array=b.asarray(rng.normal(size=(2, 3, nx, ny)) + 1j*rng.normal(size=(2, 3, nx, ny)),
                         dtype=b.complex_dtype),
        wavelengths=b.asarray([.037, .045]), eVs=b.asarray([100e3, 80e3]),
        offsets=b.asarray([[2, 1], [-1, -3], [23, 17]], dtype=int),
        cropping=5 if cropped else False, dx=potential.dx, dy=potential.dy,
    )
    original = to_numpy(probe._array).copy()
    expected = _reference_propagate(probe, potential, layers)
    exp_shapes = []
    exp = b.exp

    def record_exp(array):
        """Record operator allocation shapes to catch accidental broadcasting copies."""
        exp_shapes.append(tuple(array.shape))
        return exp(array)

    def no_roll(*args, **kwargs):
        """Crop geometry must not roll full grid arrays at each slice."""
        raise AssertionError('unexpected grid roll')

    monkeypatch.setattr(b, 'exp', record_exp)
    monkeypatch.setattr(b, 'roll', no_roll)
    actual = mm.Propagate(probe, potential, b, store_all_slices=layers is not None,
                          stored_slice_indices=layers)
    tolerance = 20 * np.finfo(to_numpy(actual).real.dtype).eps
    np.testing.assert_allclose(to_numpy(actual), to_numpy(expected), rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(to_numpy(probe._array), original)
    assert exp_shapes[0] == (2, nx, ny)  # Fresnel planes never repeat per position
    assert exp_shapes[1:] == ([(2, 3, nx, ny)] if cropped else [(2, nx, ny)]) * 4


@pytest.mark.parametrize('limit,disk', [(0, False), (1, False), (16*1024**2, False),
                                       (3240, False), (3239, False),
                                       (0, True), (16*1024**2, True)])
def test_probe_batch_potential_reuse_is_bounded_and_numerically_identical(backend, tmp_path, monkeypatch, limit, disk):
    """Only fitting volumes are retained; disk caching remains independently opt-in."""
    monkeypatch.setattr(cm, 'make_backend', lambda device: backend)
    itemsize = to_numpy(backend.zeros(0)).dtype.itemsize
    # Keep the exact-fit and one-byte-too-small cases in every precision.
    if limit in (3240, 3239):
        limit = 405 * itemsize - (limit == 3239)
    positions = np.array([[[1., 1., .3], [2., 2., 1.3]],
                          [[1.1, 1., .3], [2., 2.1, 1.3]]])
    trajectory = Trajectory(np.array([6, 14]), positions, np.zeros_like(positions),
                            np.diag([4., 4., 2.]), .01)

    def run(budget, cache_disk, name):
        """Run a small looped-probe calculation in isolated scratch storage."""
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(trajectory, aperture=10., sampling=.5, slice_thickness=.5,
                   probe_xs=[1., 2., 3.], probe_ys=[2.], loop_probes=1,
                   save_path=tmp_path / name, cache_wavefunctions=False,
                   cache_potentials=cache_disk, potential_cache_max_bytes=budget)
        return calc, calc.run()

    _, baseline = run(0, False, 'baseline')
    evaluated = []
    build_calls = []
    original_slice, original_build = pm.Potential._calculate_slice, pm.Potential.build

    def count_slice(self, z):
        """Count actual generation, excluding RAM and disk cache hits."""
        cache_file = self._cache_path(z)
        if self.array is None and not (cache_file is not None and cache_file.exists()):
            evaluated.append((self._frame_idx, z))
        return original_slice(self, z)

    def count_build(self, progress=False):
        """Record retained full-volume allocations."""
        build_calls.append(self.nx * self.ny * self.n_slices * itemsize)
        return original_build(self, progress)

    monkeypatch.setattr(pm.Potential, '_calculate_slice', count_slice)
    monkeypatch.setattr(pm.Potential, 'build', count_build)
    calc, result = run(limit, disk, 'optimized')
    tolerance = 20 * np.finfo(to_numpy(backend.zeros(0)).dtype).eps
    np.testing.assert_allclose(result.data, baseline.data, rtol=tolerance, atol=tolerance)
    assert all(size <= limit for size in build_calls)
    fits = len(calc.xs) * len(calc.ys) * calc.nz * itemsize <= limit
    assert bool(build_calls) == fits
    expected_passes = 1 if fits or disk else 3
    assert len(evaluated) == trajectory.n_frames * calc.nz * expected_passes
    assert bool(list(calc.output_dir.glob('potential_*.npy'))) == disk


@pytest.mark.parametrize('limit', [-1, 1.5, True, np.inf])
def test_potential_budget_rejects_invalid_values(limit):
    """The memory limit is explicit, finite, integer-valued, and nonnegative."""
    trajectory = Trajectory(np.array([6]), np.zeros((1, 1, 3)), np.zeros((1, 1, 3)),
                            np.eye(3) * 3, .01)
    with pytest.raises(ValueError, match='potential_cache_max_bytes'):
        MultisliceCalculator(force_cpu=True).setup(trajectory, potential_cache_max_bytes=limit)


@pytest.mark.parametrize('cropped', [False, True])
@pytest.mark.parametrize('reciprocal', [False, True])
def test_batched_probe_shifts_match_position_loop(backend, monkeypatch, cropped, reciprocal):
    """Template FFT reuse preserves offsets, energy copies, and centre no-ops."""
    b = backend
    rng = np.random.default_rng(53)
    grid = np.arange(12.) * .4
    positions = np.array([[2.4, 2.4], [.1, 1.3], [-.7, 8.9], [3.1, 2.3]])
    probe = mm.Probe(grid, grid, 10., 100e3, backend=b, defer_shifts=True,
                     array=rng.normal(size=(2, 1, 12, 12)) + 1j*rng.normal(size=(2, 1, 12, 12)),
                     probe_positions=positions, cropping=7 if cropped else False,
                     stay_reciprocal=reciprocal)
    positions[0] = [probe.lx / 2, probe.ly / 2]
    template = probe._array[:, 0]
    if cropped:
        template = template[:, 3:10, 3:10]
    expected, offsets = [], []
    for x, y in positions:
        if x == probe.lx/2 and y == probe.ly/2:
            value, offset = template, (3, 3) if cropped else (0, 0)
        else:
            value, offset = probe.placeProbe(template, x, y)
        expected.append(to_numpy(value))
        offsets.append(offset)
    calls = []
    original = b.fft2

    def record_fft(value, *args, **kwargs):
        """Count template transforms, not transforms used by the reference."""
        calls.append(tuple(value.shape))
        return original(value, *args, **kwargs)

    monkeypatch.setattr(b, 'fft2', record_fft)
    probe.applyShifts()
    tolerance = 30 * np.finfo(to_numpy(probe._array).real.dtype).eps
    np.testing.assert_allclose(to_numpy(probe._array), np.stack(expected, axis=1),
                               atol=tolerance, rtol=tolerance)
    np.testing.assert_array_equal(probe.offsets, np.asarray(offsets, dtype=int))
    assert len(calls) == (0 if reciprocal else 1)
    saved = to_numpy(probe._array).copy()
    probe.applyShifts()
    np.testing.assert_array_equal(to_numpy(probe._array), saved)


def test_static_potential_state_excludes_frame_data(backend):
    """Invariant sharing cannot pin a previous frame's positions or volume."""
    grid = np.arange(6.) * .5
    positions = np.array([[.4, 1., .3], [1.4, .6, .9]])
    first = pm.Potential(grid, grid, grid, positions, ['C', 'Si'], backend=backend)
    first.build()
    state = first._static_state()
    assert not {'array', '_positions', '_slice_atom_indices', '_frame_idx'} & state.keys()
    second = pm.Potential._from_static(state, positions + .1, 1)
    reference = pm.Potential(grid, grid, grid, positions + .1, ['C', 'Si'], backend=backend)
    second.build()
    reference.build()
    assert second._form_factors is first._form_factors
    assert second.kxs is first.kxs
    assert second.array is not first.array
    np.testing.assert_array_equal(to_numpy(second.array), to_numpy(reference.array))


@pytest.mark.parametrize('memmap', [False, True])
@pytest.mark.parametrize('disk_cache', [False, True])
def test_run_reuses_operators_and_streams_frame_output(backend, tmp_path, monkeypatch, memmap, disk_cache):
    """No duplicate output frame, no stale operators, and bounded cache uploads."""
    b = backend
    monkeypatch.setattr(cm, 'make_backend', lambda device: b)
    positions = np.array([[[1., 1., .3]], [[1.1, 1., .4]]])
    trajectory = Trajectory(np.array([6]), positions, np.zeros_like(positions), np.diag([4., 4., 2.]), .01)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(trajectory, sampling=.5, slice_thickness=.5, aperture=10.,
               probe_xs=[1., 2., 3.], probe_ys=[2.], loop_probes=2,
               return_layers=[1., 2.], use_memmap=memmap,
               cache_wavefunctions=disk_cache, save_path=tmp_path)
    builds, forms, zeros, uploads = [], [], [], []
    operators = cm._propagation_operators
    form_factor = pm.kirkland_form_factor
    original_zeros, original_asarray = b.zeros, b.asarray

    def record_operators(*args):
        """Count shared operator preparation across all batches and frames."""
        builds.append(1)
        return operators(*args)

    def record_form(*args):
        """Count element grids, which must be shared across frames."""
        forms.append(1)
        return form_factor(*args)

    def record_zeros(shape, *args, **kwargs):
        """Record allocation shapes without retaining arrays."""
        zeros.append(shape)
        return original_zeros(shape, *args, **kwargs)

    def record_upload(value, *args, **kwargs):
        """Track wave arrays uploaded during cache replay."""
        if isinstance(value, np.ndarray) and value.ndim == 5:
            uploads.append(value.shape)
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(cm, '_propagation_operators', record_operators)
    monkeypatch.setattr(pm, 'kirkland_form_factor', record_form)
    monkeypatch.setattr(b, 'zeros', record_zeros)
    monkeypatch.setattr(b, 'asarray', record_upload)
    first = calc.run().data.copy()
    assert len(builds) == len(forms) == 1
    assert (3, calc.nx, calc.ny, 2, 1) not in zeros
    assert bool(list(calc.output_dir.glob('frame_*.npy'))) == disk_cache
    second = calc.run().data.copy()
    np.testing.assert_array_equal(second, first)
    assert len(builds) == (1 if disk_cache else 2)
    assert all(shape[0] <= 2 for shape in uploads)
    # A fresh run must rebuild operators after energy/copy changes.
    calc.base_probe.addTemporalDecoherence(2., 2)
    calc.run(force_rerun=True)
    assert len(builds) == (2 if disk_cache else 3)


@pytest.mark.parametrize('keep_complex,copies', [(False, 1), (False, 2), (True, 1)])
@pytest.mark.parametrize('memmap', [False, True])
def test_tacaw_spatial_batches_preserve_spectra_and_cache(backend, tmp_path, keep_complex, copies, memmap):
    """Spatial batching changes neither the time window nor incoherent folding."""
    from pyslice.postprocessing.wf_data import WFData
    from pyslice.postprocessing.tacaw_data import TACAWData
    b = backend
    rng = np.random.default_rng(16)
    raw = rng.normal(size=(2*copies, 12, 7, 3, 1)) + 1j*rng.normal(size=(2*copies, 12, 7, 3, 1))
    array = b.asarray(raw, dtype=b.complex_dtype)
    if memmap:
        source = b.memmap(raw.shape, dtype=b.complex_dtype, filename=tmp_path/'waves.npy')
        source[:] = to_numpy(array)
        array = source
    probe = SimpleNamespace(eV=100e3, wavelength=.037, mrad=10.)
    wf = WFData(probe_positions=[(0., 0.), (1., 0.)], probe_xs=[0., 1.], probe_ys=[0.],
                time=np.arange(12)*.1, kxs=np.fft.fftshift(np.fft.fftfreq(7)),
                kys=np.fft.fftshift(np.fft.fftfreq(3)),
                xs=np.arange(7.), ys=np.arange(3.), layer=[1.],
                array=array, probe=probe, backend=b, cache_dir=tmp_path)
    reference = TACAWData(wf, keep_complex=keep_complex, chunk_size_time=6, force_rerun=True)
    expected = reference.array.copy()
    itemsize = to_numpy(b.zeros(0, dtype=b.complex_dtype)).dtype.itemsize
    # Exactly two kx columns per batch, plus an incomplete final column.
    budget = 4 * 2*copies * 6 * 3 * itemsize * 2
    result = TACAWData(wf, keep_complex=keep_complex, chunk_size_time=6,
                       fft_batch_max_bytes=budget, force_rerun=True)
    assert result.fft_batch_kx == 2
    assert result.array.shape == expected.shape
    np.testing.assert_array_equal(result.frequencies, reference.frequencies)
    tolerance = 30 * max(np.finfo(expected.real.dtype).eps,
                         np.finfo(to_numpy(b.zeros(0)).dtype).eps)
    np.testing.assert_allclose(result.array, expected, rtol=tolerance, atol=tolerance)
    reloaded = TACAWData(wf, keep_complex=keep_complex, chunk_size_time=6,
                         fft_batch_max_bytes=budget)
    np.testing.assert_array_equal(reloaded.array, result.array)
    if memmap:
        assert isinstance(reloaded._array, np.memmap)
    if not keep_complex:
        reference.apply_bose_correction(300., fold=True)
        reloaded.apply_bose_correction(300., fold=True)
        np.testing.assert_allclose(reloaded.array, reference.array, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize('budget', [0, -1, 1.5, True, np.inf])
def test_tacaw_rejects_invalid_spatial_budget(budget):
    """Reject ambiguous or nonpositive workspace targets before computation."""
    from pyslice.postprocessing.tacaw_data import TACAWData
    wf = SimpleNamespace(_backend=NumpyBackend(), probe_positions=[], _time=[], _kxs=[],
                          _kys=[], _xs=[], _ys=[], _layer=[], probe=None, cache_dir=None)
    with pytest.raises(ValueError, match='fft_batch_max_bytes'):
        TACAWData(wf, fft_batch_max_bytes=budget)


def test_single_precision_nyquist_validation_preserves_closed_grid_requirement():
    """Accept float32 FFT roundoff without admitting physically unpaired bins."""
    from pyslice.postprocessing.tacaw_data import _inversion_indices
    frequencies = np.fft.fftshift(np.fft.fftfreq(6, .1)).astype(np.float32)
    np.testing.assert_array_equal(_inversion_indices(frequencies, 'frequency'), [0, 5, 4, 3, 2, 1])
    frequencies[-1] += .01
    with pytest.raises(ValueError, match='not closed under inversion'):
        _inversion_indices(frequencies, 'frequency')


@pytest.mark.parametrize('memmap', [False, True])
def test_batched_cache_replay_preserves_adf_and_energy_copies(backend, tmp_path, monkeypatch, memmap):
    """Mapped cache consumption preserves ADF layers and incoherent copy order."""
    monkeypatch.setattr(cm, 'make_backend', lambda device: backend)
    positions = np.array([[[1., 1., .3]], [[1.1, 1., .4]]])
    trajectory = Trajectory(np.array([6]), positions, np.zeros_like(positions), np.diag([4., 4., 2.]), .01)

    def run():
        """Create a fresh calculator so detector state is identical on replay."""
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(trajectory, sampling=.5, slice_thickness=.5, aperture=10.,
                   probe_xs=[1., 2., 3.], probe_ys=[2.], loop_probes=2,
                   return_layers=[1., 2.], use_memmap=memmap, ADF=(0., 10.),
                   cache_wavefunctions=True, save_path=tmp_path)
        calc.base_probe.addTemporalDecoherence(2., 2)
        return calc.run()

    waves, adf = run()
    expected_waves, expected_adf = waves.data.copy(), to_numpy(adf._array).copy()

    def unexpected_propagation(*args, **kwargs):
        """Prove the second run used the disk cache."""
        raise AssertionError('cache replay propagated waves')

    monkeypatch.setattr(cm, 'Propagate', unexpected_propagation)
    cached_waves, cached_adf = run()
    np.testing.assert_array_equal(cached_waves.data, expected_waves)
    np.testing.assert_allclose(to_numpy(cached_adf._array), expected_adf,
                               rtol=20*np.finfo(expected_adf.real.dtype).eps, atol=1e-12)
