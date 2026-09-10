"""Reject nonphysical FFT time axes and retain known temporal frequencies."""

from types import SimpleNamespace

import numpy as np
import pytest

from pyslice import TACAWData, WFData
from pyslice.backend import NumpyBackend


def _wave(time, cache_dir):
    time = np.asarray(time, dtype=float)
    signal = np.exp(2j * np.pi * 5.0 * time)
    array = np.broadcast_to(signal[None, :, None, None, None], (1, len(time), 2, 2, 1)).copy()
    return WFData(
        probe_positions=[(0.0, 0.0)], probe_xs=[0.0], probe_ys=[0.0],
        time=time, kxs=np.array([-1.0, 0.0]), kys=np.array([-1.0, 0.0]),
        xs=np.array([0.0, 0.5]), ys=np.array([0.0, 0.5]), layer=np.array([0]),
        array=array, backend=NumpyBackend(), cache_dir=cache_dir,
        probe=SimpleNamespace(eV=100e3, wavelength=0.037, mrad=0.0),
    )


@pytest.mark.parametrize(
    ('times', 'message'),
    [
        ([0.0], 'at least two'),
        ([0.0, 0.0, 0.0], 'positive chronological'),
        ([0.1, 0.05, 0.0], 'positive chronological'),
        ([0.0, 0.1, 0.05], 'uniformly spaced'),
        ([0.0, 0.05, 0.11], 'uniformly spaced'),
        ([0.0, np.nan, 0.1], 'uniformly spaced'),
    ],
)
def test_tacaw_rejects_invalid_time_axes(tmp_path, times, message):
    with pytest.raises(ValueError, match=message):
        TACAWData(_wave(times, tmp_path))


def test_tacaw_resolves_known_frequency_from_saved_frame_spacing(tmp_path):
    result = TACAWData(_wave(np.arange(4) * 0.05, tmp_path))
    assert result.frequencies[np.argmax(result.spectrum())] == pytest.approx(5.0)


def test_tacaw_rejects_missing_returned_layers(tmp_path):
    wave = _wave([0.0, 0.05], tmp_path)
    wave._array = wave._array[..., :0]
    wave._layer = np.array([], dtype=int)
    with pytest.raises(ValueError, match='at least one returned wavefunction layer'):
        TACAWData(wave)
