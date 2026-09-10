"""Detector-angle validation against the propagated reciprocal bandwidth."""

import numpy as np
import pytest

from pyslice import HAADFData, MultisliceCalculator, Trajectory, WFData
from pyslice.backend import to_numpy


def _make_haadf_data(
    sampling=0.25, voltage_eV=100e3, max_kx=np.inf, max_ky=np.inf
):
    trajectory = Trajectory(
        atom_types=np.array([14]),
        positions=np.array([[[2.0, 2.0, 1.0]]], dtype=np.float32),
        velocities=np.zeros((1, 1, 3), dtype=np.float32),
        box_matrix=np.diag([4.0, 4.0, 2.0]).astype(np.float32),
        timestep=0.005,
    )
    calculator = MultisliceCalculator(force_cpu=True)
    calculator.setup(
        trajectory,
        aperture=20,
        voltage_eV=voltage_eV,
        sampling=sampling,
        slice_thickness=0.5,
        max_kx=max_kx,
        max_ky=max_ky,
        cache_wavefunctions=False,
    )
    backend = calculator._backend
    kxs = calculator.kxs[calculator.keep_kxs_indices]
    kys = calculator.kys[calculator.keep_kys_indices]
    array = backend.zeros(
        (1, 1, len(kxs), len(kys), 1), dtype=backend.complex_dtype
    )
    wave = WFData(
        probe_positions=calculator.probe_positions,
        probe_xs=calculator.probe_xs,
        probe_ys=calculator.probe_ys,
        time=np.array([0.0]),
        kxs=kxs,
        kys=kys,
        xs=calculator.xs,
        ys=calculator.ys,
        layer=np.array([calculator.nz - 1]),
        array=array,
        probe=calculator.base_probe,
        backend=backend,
        cache_dir=None,
    )
    return HAADFData(wave)


def test_detector_limit_matches_untapered_antialias_bandwidth():
    haadf = _make_haadf_data()
    probe_axis_limit = min(
        np.max(np.abs(to_numpy(haadf.probe.kxs))),
        np.max(np.abs(to_numpy(haadf.probe.kys))),
    )
    expected = (
        (2 / 3 - 0.02)
        * probe_axis_limit
        * float(haadf.probe.wavelength)
        * 1e3
    )
    assert haadf.max_detector_mrad == pytest.approx(expected)


def test_detector_warns_when_outer_angle_enters_tapered_region():
    haadf = _make_haadf_data()
    with pytest.warns(RuntimeWarning, match="cropped or tapered"):
        mask = haadf.getMask(5, haadf.max_detector_mrad + 10)
    assert mask.shape == (len(haadf.kxs), len(haadf.kys))


def test_detector_rejects_empty_or_invalid_annuli():
    haadf = _make_haadf_data()
    with pytest.raises(ValueError, match="outside the usable"):
        haadf.getMask(haadf.max_detector_mrad, haadf.max_detector_mrad + 10)
    with pytest.raises(ValueError, match="0 <= inner_mrad < outer_mrad"):
        haadf.getMask(20, 10)


@pytest.mark.parametrize(
    ("sampling", "outer_mrad", "max_k"),
    [
        (0.05, 200, np.inf),
        (0.075, 150, 4.25),
    ],
)
def test_documented_haadf_grids_fully_represent_their_detectors(
    sampling, outer_mrad, max_k
):
    haadf = _make_haadf_data(sampling=sampling, max_kx=max_k, max_ky=max_k)
    assert haadf.max_detector_mrad > outer_mrad


def test_haadf_plot_uses_ascending_xy_orientation(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    close_figure = plt.close
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)
    haadf = _make_haadf_data(sampling=0.05)
    haadf._xs = np.array([0.0, 1.0])
    haadf._ys = np.array([0.0, 1.0])
    haadf._array = np.arange(len(haadf._xs) * len(haadf._ys)).reshape(
        len(haadf._xs), len(haadf._ys)
    )

    haadf.plot(tmp_path / "haadf.png")

    image = plt.gcf().axes[0].images[0]
    assert image.origin == "lower"
    np.testing.assert_array_equal(np.asarray(image.get_array()), haadf.array.T)
    close_figure(plt.gcf())


def test_depth_resolved_plot_tiles_nonzero_origin_without_flipping(tmp_path):
    import matplotlib.pyplot as plt

    haadf = _make_haadf_data()
    haadf._xs = np.array([2.0, 3.0])
    haadf._ys = np.array([5.0, 7.0, 9.0])
    haadf._array = np.arange(12).reshape(2, 2, 3)
    haadf.plot(tmp_path / "depth_tiles.png", layer=1, tiling=(2, 2))
    image = plt.gcf().axes[0].images[0]
    np.testing.assert_array_equal(image.get_array(),
                                  np.tile(haadf._array[1], (2, 2)).T)
    np.testing.assert_allclose(image.get_extent(), [1.5, 5.5, 4, 16])
    assert image.origin == "lower"
    plt.close(plt.gcf())


def test_plot_rejects_unknown_singleton_tiling_period(tmp_path):
    import matplotlib.pyplot as plt

    haadf = _make_haadf_data()
    haadf._array = np.ones((1, 1))
    with pytest.raises(ValueError, match="singleton"):
        haadf.plot(tmp_path / "invalid.png", tiling=(2, 1))
    plt.close(plt.gcf())
