"""Regression coverage for the five public-main review findings."""
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.verlet import VelocityVerlet

from pyslice import Loader, TACAWData, WFData
from pyslice.backend import NumpyBackend, TORCH_AVAILABLE, to_numpy
from pyslice.data.seashell import sea_available


@pytest.fixture(params=['numpy', 'torch'])
def backend(request):
    """Exercise array operations on NumPy and optional Torch CPU."""
    if request.param == 'numpy':
        return NumpyBackend()
    if not TORCH_AVAILABLE:
        pytest.skip('Torch is not installed')
    from pyslice.backend import TorchBackend
    return TorchBackend(device='cpu')


def _wave(backend, n_copies=2):
    """Build distinct scan positions and wavelengths on an odd-sized grid.

    Parameters
    ----------
    backend : pyslice.backend.Backend
        Array implementation to exercise.
    n_copies : int
        Number of initial incoherent copies, each with two scan positions.

    Returns
    -------
    WFData
        Four time frames and two stored specimen layers.
    """
    array = np.ones((n_copies, 2, 4, 5, 5, 2), dtype=np.complex128)
    array[:, 1] *= 3
    array *= np.arange(1, n_copies + 1)[:, None, None, None, None, None]
    array *= np.exp(2j * np.pi * np.arange(4) / 4)[None, None, :, None, None, None]
    probe = SimpleNamespace(
        _array=backend.asarray(np.ones((n_copies, 2, 5, 5))),
        eV=100e3, wavelength=0.037, mrad=0.0,
        wavelengths=backend.asarray(np.linspace(0.03, 0.06, n_copies)),
    )
    return WFData(
        probe_positions=[(0., 0.), (1., 0.)], probe_xs=[0., 1.], probe_ys=[0.],
        time=backend.asarray(np.arange(4) * 0.01),
        kxs=backend.asarray(np.fft.fftshift(np.fft.fftfreq(5))),
        kys=backend.asarray(np.fft.fftshift(np.fft.fftfreq(5))),
        xs=backend.asarray(np.arange(5.)), ys=backend.asarray(np.arange(5.)),
        layer=backend.asarray(np.array([0, 1])),
        array=backend.asarray(array.reshape(-1, 4, 5, 5, 2), dtype=backend.complex_dtype),
        probe=probe, backend=backend,
    )


@pytest.mark.parametrize('samples', [1, 3, 6])
def test_defocus_preserves_each_scan_spectrum_and_copy_order(backend, samples):
    """Defocus is phase-only, with normalized, copy-major ensemble weights."""
    wf = _wave(backend)
    before = to_numpy(wf.array).copy()
    probe_before = to_numpy(wf.probe._array).copy()
    wavelengths_before = to_numpy(wf.probe.wavelengths).copy()
    spectrum_before = [TACAWData(wf).spectrum(probe_index=i) for i in range(2)]
    wf.addSpatialDecoherence(10., samples)

    reshaped = to_numpy(wf.reshaped())
    assert reshaped.shape == (2 * samples, 2, 1, 4, 5, 5, 2)
    np.testing.assert_allclose(
        np.sum(np.abs(reshaped[:, :, 0]) ** 2, axis=0),
        np.sum(np.abs(before.reshape(2, 2, 4, 5, 5, 2)) ** 2, axis=0),
        rtol=2e-6,
    )
    for i in range(2):
        np.testing.assert_allclose(TACAWData(wf).spectrum(probe_index=i),
                                   spectrum_before[i], rtol=2e-6, atol=1e-4)
    np.testing.assert_array_equal(to_numpy(wf.probe._array), probe_before)
    np.testing.assert_array_equal(to_numpy(wf.probe.wavelengths), wavelengths_before)
    if samples == 1:
        np.testing.assert_array_equal(to_numpy(wf.array), before)


def test_defocus_retains_wavelengths_for_repeated_calls_and_downstream_optics(backend):
    """Each expanded row uses its own wavelength through later operations."""
    wf = _wave(backend)
    before = to_numpy(wf.array).copy()
    row_wavelengths = np.repeat(to_numpy(wf.probe.wavelengths), 2)
    k_sq = wf.kxs[:, None] ** 2 + wf.kys[None, :] ** 2
    offsets = np.linspace(-20., 20., 3)
    weights = np.exp(-offsets ** 2 / 100.)
    weights /= np.linalg.norm(weights)
    expected = np.concatenate([
        before * weight * np.exp(-1j * np.pi * row_wavelengths[:, None, None]
                                 * dz * k_sq)[..., None][:, None]
        for dz, weight in zip(offsets, weights)
    ])
    wf.addSpatialDecoherence(10., 3)
    np.testing.assert_allclose(to_numpy(wf.array), expected, rtol=2e-6, atol=1e-6)
    wf.addSpatialDecoherence(2., 1)
    np.testing.assert_allclose(to_numpy(wf.array), expected, rtol=2e-6, atol=1e-6)
    expanded_wavelengths = np.tile(row_wavelengths, 3)
    expected = np.concatenate([
        expected / np.sqrt(2.) * np.exp(
            -1j * np.pi * expanded_wavelengths[:, None, None] * dz * k_sq
        )[:, None, :, :, None] for dz in (-4., 4.)
    ])
    wf.addSpatialDecoherence(2., 2)
    expanded_wavelengths = np.tile(expanded_wavelengths, 2)
    np.testing.assert_allclose(to_numpy(wf.array), expected, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(to_numpy(wf._row_wavelengths()), expanded_wavelengths)
    wf.propagate_free_space(12.)
    expected[..., -1] *= np.exp(
        -1j * np.pi * expanded_wavelengths[:, None, None] * 12. * k_sq
    )[:, None]
    np.testing.assert_allclose(to_numpy(wf.array), expected, rtol=2e-6, atol=1e-6)
    wf.propagate_through_lens(100.)
    assert np.all(np.isfinite(to_numpy(wf.array)))


def test_defocus_supports_scalar_wavelength_and_real_valued_initial_wave(backend):
    """Legacy scalar-only probes retain phase and normalized intensity."""
    wf = _wave(backend, n_copies=1)
    del wf.probe.wavelengths
    del wf._copy_wavelengths
    wf.array = wf.array.real
    before = np.sum(np.abs(to_numpy(wf.array)) ** 2)
    wf.addSpatialDecoherence(10., 2)
    assert np.iscomplexobj(to_numpy(wf.array))
    assert np.any(to_numpy(wf.array).imag != 0)
    np.testing.assert_allclose(np.sum(np.abs(to_numpy(wf.array)) ** 2), before, rtol=2e-6)
    np.testing.assert_allclose(to_numpy(wf._row_wavelengths()), wf.probe.wavelength)


@pytest.mark.parametrize('sigma,samples', [(0., 3), (-1., 3), (np.nan, 3),
                                          (np.inf, 3), (1., 0), (1., 2.5), (1., True)])
def test_defocus_rejects_invalid_parameters_without_mutating(sigma, samples):
    """Invalid ensemble parameters fail before changing the result."""
    wf = _wave(NumpyBackend())
    before = wf.array.copy()
    with pytest.raises(ValueError):
        wf.addSpatialDecoherence(sigma, samples)
    np.testing.assert_array_equal(wf.array, before)


@pytest.mark.skipif(not sea_available, reason='compatible sea-eco unavailable')
@pytest.mark.parametrize('operation', ['pad', 'crop', 'defocus'])
def test_shape_changes_keep_signal_calibrated_through_sea_roundtrip(backend, tmp_path, operation):
    """Coordinates, sizes, and roles follow arrays through both export routes."""
    from pySEA.sea_eco.io import load as sea_load
    from pyslice.mcp.service import PySliceService

    wf = _wave(backend)
    analysis, provenance = wf.Analysis, wf.Provenance
    wf.metadata.General.title = 'Custom title'
    if operation == 'pad':
        wf.pad_real_space(1., 2.)
    elif operation == 'crop':
        wf.crop(kx_range=(-0.2, 0.2), ky_range=(0., 0.4))
    else:
        wf.addSpatialDecoherence(10., 3)
    assert wf.Analysis is analysis
    assert wf.Provenance is provenance
    assert wf.metadata.General.title == 'Custom title'

    plain = PySliceService._as_plain_signal(wf)
    plain.to_sea(str(tmp_path / 'plain.sea'))
    loaded_plain = sea_load(str(tmp_path / 'plain.sea'))
    wf.to_sea(str(tmp_path / 'wave.sea'))
    loaded_wave = WFData.load(str(tmp_path / 'wave.sea'))
    for signal in (wf, plain, loaded_plain, loaded_wave):
        np.testing.assert_allclose(to_numpy(signal.data), to_numpy(wf.array))
        for axis, name in enumerate(('probe', 'time', 'kx', 'ky', 'layer')):
            assert len(signal.dimensions[name].values) == wf.array.shape[axis]
            assert signal.dimensions[name].size == wf.array.shape[axis]
        for name, values in [('kx', wf.kxs), ('ky', wf.kys)]:
            dim = signal.dimensions[name]
            np.testing.assert_allclose(to_numpy(dim.values), values)
            np.testing.assert_allclose(dim.offset, values[0])
            np.testing.assert_allclose(dim.scale, values[1] - values[0])
        assert signal.dimension_signature == ['probe', 'time', 'kx', 'ky', 'layer']
    np.testing.assert_allclose(loaded_wave._copy_wavelengths,
                               to_numpy(wf._copy_wavelengths))
    # Backend and Probe are intentionally excluded from .sea serialization.
    loaded_wave._backend = NumpyBackend()
    assert loaded_wave.reshaped().shape == wf.reshaped().shape
    np.testing.assert_allclose(loaded_wave._row_wavelengths(), to_numpy(wf._row_wavelengths()))


class _FreeParticle(Calculator):
    """Zero-force ASE calculator for a physical velocity-unit check."""
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Evaluate zero energy and forces for a free particle.

        Parameters
        ----------
        atoms : ase.Atoms, optional
            Current particle configuration.
        properties : list[str], optional
            Requested ASE properties.
        system_changes : list[str], optional
            ASE change flags since the last calculation.
        """
        super().calculate(atoms, properties, system_changes)
        self.results = {'energy': 0., 'forces': np.zeros((len(atoms), 3))}


@pytest.mark.parametrize('multiple_frames', [False, True])
def test_ase_velocity_matches_displacement_and_roundtrips(multiple_frames):
    """Import/export velocity agrees with motion over a known physical time."""
    from pyslice.io.build import trajectory_to_ase

    atoms = Atoms('He', positions=[[1., 1., 1.]], cell=[10., 10., 10.], pbc=True)
    atoms.set_velocities([[1., -2., 0.5]])
    atoms.calc = _FreeParticle()
    first = atoms.copy()
    VelocityVerlet(atoms, timestep=units.fs).run(1)
    physical_velocity = (atoms.positions - first.positions) / 0.001
    trajectory = Loader(atoms=[first, atoms] if multiple_frames else atoms,
                        timestep=0.001).load()
    for i in range(trajectory.n_frames):
        np.testing.assert_allclose(trajectory.velocities[i], physical_velocity, rtol=1e-6)
        np.testing.assert_allclose(trajectory.to_ase(i).get_velocities(),
                                   atoms.get_velocities(), rtol=1e-6)
        np.testing.assert_allclose(trajectory_to_ase(trajectory, i).get_velocities(),
                                   atoms.get_velocities(), rtol=1e-6)
    if sea_available:
        velocity = trajectory.sea['atoms']['velocity']
        expected = trajectory.velocities if multiple_frames else trajectory.velocities[0]
        np.testing.assert_allclose(to_numpy(velocity.data), expected, rtol=1e-6)
        assert velocity.signal_quantities.dimensions[0].units == 'Å/ps'


def test_tmdc_example_uses_concrete_orb_calculator(monkeypatch, tmp_path):
    """Exercise the actual example's MD entry point without downloading a model."""
    from pyslice import ORBMDCalculator

    calls = {}
    result = object()

    def setup(self, **kwargs):
        """Record the example's requested MD settings."""
        calls['calculator'] = self
        calls['settings'] = kwargs

    monkeypatch.setattr(ORBMDCalculator, 'setup', setup)
    monkeypatch.setattr(ORBMDCalculator, 'run', lambda self: result)
    example = Path(__file__).resolve().parents[1] / 'examples/k_space_tmdc_showcase_pub copy.txt'
    namespace = runpy.run_path(str(example), run_name='test_tmdc_example')
    atoms = Atoms('MoS2', positions=np.zeros((3, 3)), cell=[3., 3., 3.])
    assert namespace['run_md'](atoms, tmp_path) is result
    assert isinstance(calls['calculator'], ORBMDCalculator)
    assert calls['calculator'].weights_path == namespace['CACHED_WEIGHTS_PATH']
    assert calls['settings']['atoms'] is atoms
    assert calls['settings']['production_ensemble'] == 'nve'
