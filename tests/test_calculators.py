"""MultisliceCalculator regressions: caching, probe handling, PRISM."""
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


def test_force_rerun_bypasses_frame_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        cache_wavefunctions=True,
    )
    calc.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(calc.output_dir / "frame_0.npy", np.zeros((1, 1, 1, 1, 1), dtype=np.complex128))

    original_check_cache = calculators_module.checkCache
    cache_read_flags = []

    def spy_check_cache(cache_file, cache_wavefunctions, *args, **kwargs):
        cache_read_flags.append(cache_wavefunctions)
        return original_check_cache(cache_file, cache_wavefunctions, *args, **kwargs)

    monkeypatch.setattr(calculators_module, "checkCache", spy_check_cache)

    calc.run(force_rerun=True)

    assert cache_read_flags == [False]

def test_existing_tacaw_file_does_not_skip_returned_wavefunctions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        cache_wavefunctions=False,
    )
    calc.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(calc.output_dir / "tacaw.npy", np.zeros((1, 1, 1, 1), dtype=np.float64))

    wave = calc.run()

    assert np.any(np.abs(to_numpy(wave.array)) > 0)

def test_loop_probes_keeps_selected_indices_integer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_xs=[1.0, 2.0],
        probe_ys=[1.0, 2.0],
        loop_probes=2,
        cache_wavefunctions=False,
    )

    wave = calc.run(force_rerun=True)

    assert to_numpy(wave.array).shape[0] == 4

def test_preview_probes_uses_flatten_compatibility(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
    )

    calc.preview_probes()

def test_prism_probe_copy_preserves_backend_for_chunk_shifts():
    backend = NumpyBackend()
    xs = np.linspace(0.0, 4.0, 4, endpoint=False)
    ys = np.linspace(0.0, 4.0, 4, endpoint=False)
    probe = PrismProbe(xs, ys, 5, 60e3, backend=backend, nkx=2)
    probe.applyShifts()

    copied = probe.copy(selected_probes=np.array([0, 1]))
    copied.applyShifts()

    assert copied._backend is backend
    assert to_numpy(copied._array).shape[1] == 2

def test_prism_loop_probes_smoke_preserves_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_xs=[1.0, 2.0],
        probe_ys=[1.0],
        prism=2,
        loop_probes=1,
        cache_wavefunctions=False,
    )

    wave = calc.run(force_rerun=True)

    assert to_numpy(wave.array).shape[0] == 2

@pytest.mark.parametrize("use_memmap", [False, True])
def test_prism_adf_reconstructs_default_and_multilayer_outputs(
    tmp_path, monkeypatch, use_memmap
):
    def run(subdir, return_layers):
        work = tmp_path / subdir
        work.mkdir()
        monkeypatch.chdir(work)
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(
            _make_tiny_trajectory(), aperture=30, voltage_eV=60e3,
            sampling=1.0, slice_thickness=1.0,
            probe_xs=[1.0, 2.0], probe_ys=[1.0], prism=2,
            loop_probes=1, cache_wavefunctions=False, ADF=(5, 40),
            return_layers=return_layers, use_memmap=use_memmap)
        return calc, calc.run(force_rerun=True)

    _, (wave, adf) = run("exit", -1)
    assert to_numpy(wave.array).shape == (2, 1, 5, 5, 1)
    assert np.any(np.abs(to_numpy(wave.array)) > 0)
    assert to_numpy(adf.array).shape == (2, 1)

    calc, (wave_all, adf_all) = run("all", "all")
    assert to_numpy(wave_all.array).shape == (2, 1, 5, 5, calc.nz)
    assert np.all(np.any(np.abs(to_numpy(wave_all.array)) > 0, axis=(0, 1, 2, 3)))
    assert to_numpy(adf_all.array).shape == (calc.nz, 2, 1)

def test_prism_cache_validates_component_rows_not_scan_positions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def calculator():
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(
            _make_tiny_trajectory(), aperture=5, voltage_eV=60e3,
            sampling=1.0, slice_thickness=1.0,
            probe_xs=[1.0, 2.0], probe_ys=[1.0], prism=2,
            loop_probes=1, cache_wavefunctions=True)
        return calc

    first = calculator()
    expected = to_numpy(first.run(force_rerun=True).array).copy()

    def should_not_propagate(*args, **kwargs):
        raise AssertionError("valid PRISM component cache was not reused")

    second = calculator()
    monkeypatch.setattr(calculators_module, "Propagate", should_not_propagate)
    np.testing.assert_allclose(to_numpy(second.run().array), expected)

def test_static_trajectory_offcentre_probe_gives_frame_invariant_exit_wave(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    n_frames = 4
    static = np.tile(np.array([[[3.1, 4.7, 1.5]]], dtype=float), (n_frames, 1, 1))
    traj = Trajectory(
        atom_types=np.array([14]),
        positions=static,
        velocities=np.zeros_like(static),
        box_matrix=np.diag([8.0, 8.0, 3.0]),
        timestep=0.1,
    )
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        traj,
        aperture=30,
        voltage_eV=60e3,
        sampling=0.25,
        slice_thickness=1.0,
        probe_positions=[_OFFCENTRE],
        cache_wavefunctions=False,
    )
    arr = to_numpy(calc.run(force_rerun=True).array)  # (probe, frame, kx, ky, layer)

    # atoms are frozen -> a correctly-fixed probe reproduces the exit wave every
    # frame; the drift bug moved the probe each frame, so patterns diverged.
    reference = arr[:, 0]
    for t in range(1, n_frames):
        np.testing.assert_allclose(arr[:, t], reference, atol=1e-10, rtol=0)

def test_explicit_probe_position_list_is_simulated_verbatim(tmp_path, monkeypatch):
    # A non-grid probe_positions list (e.g. two atomic columns on a diagonal)
    # used to be silently rebuilt into the outer-product grid of its unique x/y
    # values.  That both changed the physics and desynced n_probes from the
    # number of probes actually simulated -> a shape-mismatch crash as soon as
    # wavefunctions were returned.
    monkeypatch.chdir(tmp_path)
    n_frames = 2
    static = np.tile(np.array([[[3.1, 4.7, 1.5]]], dtype=float), (n_frames, 1, 1))
    traj = Trajectory(
        atom_types=np.array([14]),
        positions=static,
        velocities=np.zeros_like(static),
        box_matrix=np.diag([8.0, 8.0, 3.0]),
        timestep=0.1,
    )
    requested = [(2.0, 2.0), (6.0, 6.0)]  # diagonal -> not a full 2x2 grid
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        traj,
        aperture=30,
        voltage_eV=60e3,
        sampling=0.25,
        slice_thickness=1.0,
        probe_positions=requested,
        cache_wavefunctions=False,
    )
    wave = calc.run(force_rerun=True)  # must not raise

    # exactly the requested probes are simulated (not expanded to a 4-point grid)
    np.testing.assert_allclose(
        np.asarray(calc.base_probe.probe_positions, dtype=float), requested)
    assert len(calc.probe_positions) == len(calc.base_probe.probe_positions)
    assert to_numpy(wave.array).shape[0] == len(requested)

def test_full_grid_probe_list_canonicalised_so_image_maps_correctly(tmp_path, monkeypatch):
    # A full grid handed in as an explicit list in a non-meshgrid order must
    # produce the SAME 2D image as the probe_xs/probe_ys path: WFData.reshaped()
    # assumes meshgrid order, so setup() canonicalises full grids to it.
    reference = _adf_image_for(dict(probe_xs=[2.0, 6.0], probe_ys=[2.0, 6.0]),
                               tmp_path / "ref", monkeypatch)
    scrambled = _adf_image_for(dict(probe_positions=[(6., 6.), (2., 2.), (6., 2.), (2., 6.)]),
                               tmp_path / "scr", monkeypatch)
    nested = _adf_image_for(dict(probe_positions=[(2., 2.), (2., 6.), (6., 2.), (6., 6.)]),
                            tmp_path / "nest", monkeypatch)
    np.testing.assert_allclose(scrambled, reference, atol=1e-10, rtol=0)
    np.testing.assert_allclose(nested, reference, atol=1e-10, rtol=0)

def test_malformed_probe_positions_raise_clear_error(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    traj = _make_tiny_trajectory()
    for bad in [(4.0, 4.0), [], [(2, 2, 4), (6, 6, 4)]]:
        calc = MultisliceCalculator(force_cpu=True)
        with pytest.raises(ValueError, match="probe_positions must be"):
            calc.setup(traj, aperture=5, voltage_eV=60e3, sampling=1.0,
                       slice_thickness=1.0, probe_positions=bad)


# ---------------------------------------------------------------------------
# Regression: probe cropping (min_dk) must (a) not crash on the b.roll call and
# (b) read the potential sub-window centred on each probe, not the grid corner,
# for every coherent (decoherence) copy.  For a single-slice potential there is
# no inter-slice propagation, so a cropped exit wave must EXACTLY equal the
# uncropped exit wave restricted to that probe's window.
# ---------------------------------------------------------------------------

def test_min_dk_calculator_run_completes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    static = np.tile(np.array([[[4.0, 4.0, 1.5]]], dtype=float), (2, 1, 1))
    traj = Trajectory(
        atom_types=np.array([14]),
        positions=static,
        velocities=np.zeros_like(static),
        box_matrix=np.diag([8.0, 8.0, 3.0]),
        timestep=0.1,
    )
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(traj, aperture=30, voltage_eV=100e3, sampling=0.25,
               slice_thickness=1.0, probe_positions=[(4.0, 4.0), (2.0, 6.0)],
               min_dk=0.25, cache_wavefunctions=False)
    arr = to_numpy(calc.run(force_rerun=True).array)  # previously TypeError in b.roll
    assert np.all(np.isfinite(arr))
    assert np.max(np.abs(arr)) > 0

def test_wavefunction_cache_key_uses_full_trajectory(tmp_path, monkeypatch):
    # The key hashes the whole trajectory, so runs differing only in a later
    # frame or in y/z (e.g. a temperature sweep from the same initial structure)
    # no longer collide on the frame-0-x fingerprint.
    monkeypatch.chdir(tmp_path)
    base = (np.random.RandomState(0).rand(5, 2, 3) * 3).astype(np.float32)

    def key(pos):
        traj = Trajectory(
            atom_types=np.array([14, 14]), positions=pos.astype(np.float32),
            velocities=np.zeros_like(pos, dtype=np.float32),
            box_matrix=np.diag([8.0, 8.0, 3.0]), timestep=0.1)
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(traj, aperture=30, voltage_eV=100e3, sampling=0.25,
                   slice_thickness=1.0, probe_positions=[(4.0, 4.0)],
                   cache_wavefunctions=False)
        return calc.cache_key

    k = key(base.copy())
    assert key(base.copy()) == k                       # identical -> shared cache
    later = base.copy(); later[3, 0, 0] += 0.5
    assert key(later) != k                             # differs only in frame 3
    ycomp = base.copy(); ycomp[0, 0, 1] += 0.5
    assert key(ycomp) != k                             # differs only in frame-0 y

def test_wavefunction_cache_key_partitions_skip_vacuum(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    traj = _make_tiny_trajectory()

    def key(skip_vacuum):
        calc = MultisliceCalculator(force_cpu=True)
        calc.setup(
            traj, aperture=30, voltage_eV=60e3, sampling=0.25,
            slice_thickness=1.0, probe_xs=[1.0, 2.0], probe_ys=[1.0],
            min_dk=0.25, skip_vacuum=skip_vacuum)
        return calc.cache_key

    assert key(False) != key(True)

def test_calculator_decoherence_flows_into_folded_tacaw(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    positions = np.tile(
        np.array([[[1.5, 1.5, 1.5]]], dtype=np.float32), (4, 1, 1))
    trajectory = Trajectory(
        atom_types=np.array([14]), positions=positions,
        velocities=np.zeros_like(positions),
        box_matrix=np.diag([4.0, 4.0, 3.0]), timestep=0.1)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        trajectory, aperture=20, voltage_eV=60e3, sampling=1.0,
        slice_thickness=1.0, probe_xs=[1.0, 2.0], probe_ys=[1.0],
        cache_wavefunctions=False)
    calc.base_probe.addTemporalDecoherence(2.0, 3)
    wf = calc.run(force_rerun=True)
    assert to_numpy(wf.array).shape[:2] == (6, 4)

    raw = to_numpy(wf.array[:, :, :, :, -1])
    raw_fft = np.fft.fftshift(
        np.fft.fft(raw - raw.mean(axis=1, keepdims=True), axis=1), axes=1)
    expected = (np.abs(raw_fft) ** 2).reshape(
        3, 2, *raw_fft.shape[1:]).sum(axis=0)
    tacaw = TACAWData(wf, force_rerun=True)
    assert tacaw.array.shape[0] == 2
    np.testing.assert_allclose(tacaw.array, expected)

def test_cache_versions_are_derived_from_source(tmp_path):
    from pyslice.backend import source_files_version
    # the helper is content-sensitive and tolerates missing files
    f = tmp_path / "a.py"
    f.write_bytes(b"x = 1\n")
    v1 = source_files_version([f])
    f.write_bytes(b"x = 2\n")
    assert source_files_version([f]) != v1
    source_files_version([tmp_path / "does_not_exist.py"])  # must not raise
    # both cache tags embed a source hash (not a bare manual constant) and are
    # independent of each other
    wf_v = MultisliceCalculator._CACHE_VERSION
    tacaw_v = TACAWData._TACAW_CACHE_VERSION
    assert wf_v.startswith("v3-") and len(wf_v) > len("v3-")
    assert tacaw_v.startswith("v1-") and len(tacaw_v) > len("v1-")
    assert wf_v != tacaw_v

def test_decoherence_default_wavefunction_storage_keeps_every_copy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = np.array([[[4., 4., 1.0], [3., 5., 2.5]]], dtype=np.float32)
    traj = Trajectory(
        atom_types=np.array([14, 14]), positions=s, velocities=np.zeros_like(s),
        box_matrix=np.diag([8., 8., 4.]), timestep=0.1)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        traj, aperture=30, voltage_eV=100e3, sampling=0.25,
        slice_thickness=1.0, probe_xs=[2., 4., 6.], probe_ys=[2., 4., 6.],
        ADF=(45, 150), cache_wavefunctions=True,
        loop_probes=2)  # default return_layers=-1
    calc.base_probe.addTemporalDecoherence(2.0, 3)
    wf, adf = calc.run(force_rerun=True)
    assert wf.array.shape[0] == 3 * 9
    assert adf.array.shape == (3, 3)
    assert np.isfinite(to_numpy(wf.array)).all()

    # A second calculator must fold all cached copies into the same detector
    # result; the old cache path indexed only the first copy.
    expected = to_numpy(adf.array).copy()
    cached_calc = MultisliceCalculator(force_cpu=True)
    cached_calc.setup(
        traj, aperture=30, voltage_eV=100e3, sampling=0.25,
        slice_thickness=1.0, probe_xs=[2., 4., 6.], probe_ys=[2., 4., 6.],
        ADF=(45, 150), cache_wavefunctions=True, loop_probes=2)
    cached_calc.base_probe.addTemporalDecoherence(2.0, 3)
    cached_wf, cached_adf = cached_calc.run()
    assert cached_wf.array.shape[0] == 3 * 9
    np.testing.assert_allclose(to_numpy(cached_adf.array), expected, rtol=1e-7)
