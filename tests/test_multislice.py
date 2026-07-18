"""Multislice engine regressions: probe shifts, cropping, dose weights."""
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


def test_wavelength_public_helper_accepts_scalar_without_backend():
    lam = wavelength(100e3)

    assert isinstance(lam, float)
    assert lam > 0

def test_apply_shifts_is_idempotent_for_single_offcentre_probe():
    probe = _make_deferred_probe(_OFFCENTRE)  # cell centre is 4.0 -> off-centre

    probe.applyShifts()
    after_first = to_numpy(probe._array).copy()
    probe.applyShifts()  # the historically buggy re-application
    probe.applyShifts()

    np.testing.assert_array_equal(to_numpy(probe._array), after_first)
    # positioned once: peak at the requested position (not centre, not doubled)
    assert _peak_pixel(probe._array[0, 0]) == _ONCE_PIXEL

def test_apply_shifts_reapplied_after_decoherence_rebuild():
    # setup() applies shifts; addTemporalDecoherence then rebuilds the template
    # from scratch and must re-position it (flag reset), then stay idempotent.
    probe = _make_deferred_probe(_OFFCENTRE)
    probe.applyShifts()
    probe.addTemporalDecoherence(sigma_eV=1.0, N=3)

    assert probe._array.shape[0] == 3  # three energy copies
    summed = np.sum(np.abs(to_numpy(probe._array[:, 0])) ** 2, axis=0)
    assert _peak_pixel(summed) == _ONCE_PIXEL  # shifted exactly once, not twice

    frozen = to_numpy(probe._array).copy()
    probe.applyShifts()  # must be a no-op now
    np.testing.assert_array_equal(to_numpy(probe._array), frozen)

def test_cropped_probe_reads_window_centred_on_probe_not_grid_corner():
    C = 16
    for position, decohere in [((4.0, 4.0), False),   # centred
                               ((2.0, 6.0), False),   # off-centre
                               ((4.0, 4.0), True)]:    # centred + decoherence
        full, _ = _propagate(position, 0, decohere)
        crop, probe = _propagate(position, C, decohere)
        assert crop.shape[-2:] == (C, C)
        # every coherent copy must match the uncropped window at the probe's
        # recorded offset (b.roll crash + corner-window + decoherence-row bugs)
        ox, oy = (int(v) for v in probe.offsets[0])
        window = full[:, ox:ox + C, oy:oy + C]
        np.testing.assert_allclose(crop, window, atol=1e-12, rtol=0)
    # and the centred crop must NOT be the grid-corner window (the old bug)
    full, _ = _propagate((4.0, 4.0), 0)
    crop, _ = _propagate((4.0, 4.0), C)
    assert np.max(np.abs(crop[0] - full[0, 0:C, 0:C])) > 1e-3

def test_decoherence_weights_preserve_total_probe_dose():
    baseline_probe = _make_deferred_probe((4.0, 4.0))
    baseline_probe.applyShifts()
    baseline = np.sum(np.abs(to_numpy(baseline_probe._array)) ** 2)

    for n_copies in (3, 5, 7):
        probe = _make_deferred_probe((4.0, 4.0))
        probe.addTemporalDecoherence(2.0, n_copies)
        total = np.sum(np.abs(to_numpy(probe._array)) ** 2)
        np.testing.assert_allclose(total, baseline, rtol=1e-7)

    probe = _make_deferred_probe((4.0, 4.0))
    probe.addSpatialDecoherence(50.0, 7)
    total = np.sum(np.abs(to_numpy(probe._array)) ** 2)
    np.testing.assert_allclose(total, baseline, rtol=1e-7)

    with pytest.raises(ValueError, match="positive"):
        probe.addTemporalDecoherence(0, 3)
    with pytest.raises(ValueError, match="positive integer"):
        probe.addSpatialDecoherence(1, 0)

    # One quadrature point denotes the distribution centre, not its -2 sigma
    # endpoint (numpy.linspace(start, stop, 1) returns start).
    temporal_one = _make_deferred_probe((4.0, 4.0))
    temporal_one.addTemporalDecoherence(2.0, 1)
    np.testing.assert_allclose(to_numpy(temporal_one.eVs), [temporal_one.eV])
    spatial_one = _make_deferred_probe((4.0, 4.0))
    reference = to_numpy(spatial_one._array).copy()
    spatial_one.addSpatialDecoherence(50.0, 1)
    np.testing.assert_allclose(to_numpy(spatial_one._array), reference, atol=1e-12)
