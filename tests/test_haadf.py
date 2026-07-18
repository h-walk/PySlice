"""HAADF/ADF regressions: depth-resolved stack, decoherence, layers."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyslice.backend import NumpyBackend, TORCH_AVAILABLE, to_numpy
if TORCH_AVAILABLE:
    from pyslice.backend import TorchBackend
from pyslice.multislice import calculators as calculators_module
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.multislice.multislice import PrismProbe, Probe, Propagate, wavelength
from pyslice.multislice.potentials import Potential
from pyslice.multislice.trajectory import Trajectory
from pyslice.io.loader import Loader
from pyslice.postprocessing.haadf_data import HAADFData
from pyslice.postprocessing.tacaw_data import TACAWData
from pyslice.postprocessing.wf_data import WFData



from _regression_helpers import *  # shared builders (see __all__)


def test_haadf_serialization_excludes_backend():
    assert "_backend" in HAADFData._sea_config["exclude_attrs"]

def test_adf_depth_resolved_stack_not_layer_sum(tmp_path, monkeypatch):
    # Default: a single exit-wave ADF, kept 2D for back-compat.
    _, adf = _adf_run(tmp_path, monkeypatch)
    assert to_numpy(adf.array).shape == (3, 3)
    # return_layers='all': one ADF per thickness (a stack), NOT a sum over layers.
    _, adf_all = _adf_run(tmp_path / "all", monkeypatch, return_layers="all")
    stack = to_numpy(adf_all.array)
    assert stack.ndim == 3 and stack.shape[1:] == (3, 3)
    assert stack.shape[0] == len(adf_all.thicknesses)
    # Stored waves are captured after each slice transmission, so their depth
    # is the far boundary of that slice: first > 0 and exit == specimen depth.
    expected_depths = np.linspace(4.0 / len(stack), 4.0, len(stack))
    np.testing.assert_allclose(adf_all.thicknesses, expected_depths)
    np.testing.assert_allclose(adf.thicknesses, [4.0])
    np.testing.assert_allclose(stack[-1], to_numpy(adf.array), atol=1e-6)  # exit == default
    assert not np.allclose(stack[0], stack[-1])          # genuinely per-thickness
    assert not np.allclose(stack[-1], stack.sum(0))      # not the old layer-sum

def test_adf_depth_stack_exposes_a_layer_dimension(monkeypatch):
    import pyslice.postprocessing.haadf_data as haadf_module

    class FakeDimension:
        def __init__(self, **kwargs):
            self.name = kwargs['name']
            self.values = np.asarray(kwargs['values'])
            self.units = kwargs.get('units')

    class FakeDimensions:
        def __init__(self, dimensions, nav_dimensions, sig_dimensions):
            self.dimensions = dimensions
            self.nav_dimensions = nav_dimensions
            self.sig_dimensions = sig_dimensions

    class FakeMetadata:
        def __init__(self, values):
            self.Simulation = SimpleNamespace(**values['Simulation'])

    monkeypatch.setattr(haadf_module, 'Dimension', FakeDimension)
    monkeypatch.setattr(haadf_module, 'Dimensions', FakeDimensions)
    monkeypatch.setattr(haadf_module, 'Metadata', FakeMetadata)

    wf = _make_layered_wf(None, n_layers=2)
    adf = haadf_module.HAADFData(wf)
    adf.calculateADF(inner_mrad=0, outer_mrad=1e6)
    assert [d.name for d in adf.dimensions.dimensions] == ['layer', 'x', 'y']
    np.testing.assert_array_equal(adf.dimensions.dimensions[0].values, [0, 1])

def test_adf_sums_decoherence_copies(tmp_path, monkeypatch):
    def total(n_copies):
        s = np.array([[[4., 4., 1.0], [3., 5., 2.5]]], dtype=np.float32)
        traj = Trajectory(atom_types=np.array([14, 14]), positions=s,
                          velocities=np.zeros_like(s), box_matrix=np.diag([8., 8., 4.]),
                          timestep=0.1)
        monkeypatch.chdir(tmp_path)
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(traj, aperture=30, voltage_eV=100e3, sampling=0.25, slice_thickness=1.0,
                   probe_xs=[2., 4., 6.], probe_ys=[2., 4., 6.], ADF=(45, 150),
                   cache_wavefunctions=False, return_layers=None)
        if n_copies:
            calc.base_probe.addTemporalDecoherence(2.0, n_copies)
        return float(np.sum(np.abs(to_numpy(calc.run(force_rerun=True)[1].array))))
    baseline = total(None)
    # Correct quadrature is independent of the number of samples. N=3 happened
    # to pass before because its central sample dominated; N=5/7 exposed the
    # missing normalisation as a dose increase.
    for n_copies in (3, 5, 7):
        ratio = total(n_copies) / baseline
        assert 0.9 < ratio < 1.1, (n_copies, ratio)

def test_cached_looped_adf_uses_full_scan_count(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    traj = _make_tiny_trajectory()

    def calculator():
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(
            traj, aperture=20, voltage_eV=60e3, sampling=0.5,
            slice_thickness=1.0, probe_xs=[1.0, 2.0],
            probe_ys=[1.0, 2.0], loop_probes=2, ADF=(5, 40),
            cache_wavefunctions=True)
        return calc

    first = calculator()
    expected = to_numpy(first.run(force_rerun=True)[1].array).copy()
    second = calculator()
    np.testing.assert_allclose(to_numpy(second.run()[1].array), expected)

def test_adf_layers_decoupled_from_return_layers(tmp_path, monkeypatch):
    # A depth-resolved ADF can be produced without returning/storing those
    # wavefunctions (adf_layers is independent of return_layers).
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    s = np.array([[[4., 4., 1.0], [3., 5., 2.5]]], dtype=np.float32)
    traj = Trajectory(atom_types=np.array([14, 14]), positions=s,
                      velocities=np.zeros_like(s), box_matrix=np.diag([8., 8., 4.]),
                      timestep=0.1)

    def run(**kw):
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(traj, aperture=30, voltage_eV=100e3, sampling=0.25, slice_thickness=1.0,
                   probe_xs=[2., 4., 6.], probe_ys=[2., 4., 6.], ADF=(45, 150),
                   cache_wavefunctions=False, **kw)
        _, adf = calc.run(force_rerun=True)
        return calc, adf

    # ADF at every thickness, but NO wavefunctions returned or stored.
    calc, adf = run(return_layers=None, adf_layers="all")
    assert to_numpy(adf.array).shape == (5, 3, 3)
    assert calc.returns_wavefunctions is False
    assert not hasattr(calc, "wavefunction_data")            # no wavefunction RAM

    # Identical to the coupled way (return_layers='all'), which does store them.
    _, adf_coupled = run(return_layers="all")
    np.testing.assert_allclose(to_numpy(adf.array), to_numpy(adf_coupled.array), atol=1e-8)

    # Explicit subset of thicknesses.
    _, adf_sub = run(return_layers=None, adf_layers=[0, 4])
    assert to_numpy(adf_sub.array).shape == (2, 3, 3)
    assert len(adf_sub.thicknesses) == 2
