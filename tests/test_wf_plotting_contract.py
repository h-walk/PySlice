from types import SimpleNamespace

import numpy as np

from pyslice import WFData
from pyslice.backend import NumpyBackend


def test_plot_reciprocal_averages_intensity_not_amplitude(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    close_figure = plt.close
    monkeypatch.setattr(plt, "close", lambda *_args, **_kwargs: None)
    backend = NumpyBackend()
    array = np.empty((1, 2, 2, 2, 1), dtype=np.complex128)
    array[:, 0] = 1.0
    array[:, 1] = 3.0
    wf = WFData(
        probe_positions=[(0.0, 0.0)],
        probe_xs=[0.0],
        probe_ys=[0.0],
        time=np.array([0.0, 0.01]),
        kxs=np.array([-0.5, 0.0]),
        kys=np.array([-0.5, 0.0]),
        xs=np.array([0.0, 1.0]),
        ys=np.array([0.0, 1.0]),
        layer=np.array([0]),
        array=array,
        probe=SimpleNamespace(
            _array=np.ones((1, 1, 2, 2)), eV=100e3,
            wavelength=0.037, mrad=0.0,
        ),
        backend=backend,
    )

    wf.plot_reciprocal(
        filename=tmp_path / "mean-intensity.png",
        whichTimestep="mean",
        powerscaling=1.0,
    )

    rendered = np.asarray(plt.gcf().axes[0].images[0].get_array())
    np.testing.assert_allclose(rendered, 5.0)
    close_figure(plt.gcf())


def test_counts_samples_normalized_intensity():
    backend = NumpyBackend()
    array = np.array([1.0 + 0j, 2.0 + 0j]).reshape(1, 1, 2, 1, 1)
    wf = WFData(
        probe_positions=[(0.0, 0.0)],
        probe_xs=[0.0],
        probe_ys=[0.0],
        time=np.array([0.0]),
        kxs=np.array([-0.5, 0.0]),
        kys=np.array([0.0]),
        xs=np.array([0.0, 1.0]),
        ys=np.array([0.0]),
        layer=np.array([0]),
        array=array,
        probe=SimpleNamespace(
            _array=np.ones((1, 1, 2, 1)), eV=100e3,
            wavelength=0.037, mrad=0.0,
        ),
        backend=backend,
    )

    wf.counts(100)

    np.testing.assert_allclose(wf.probability.ravel(), [0.2, 0.8])
    assert wf.array.sum() == 100
