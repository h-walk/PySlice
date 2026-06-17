import numpy as np

from pyslice import MultisliceCalculator, Trajectory


def test_on_the_fly_adf_returns_image_without_full_wavefunction_stack():
    trajectory = Trajectory(
        atom_types=np.array([14]),
        positions=np.array([[[2.0, 2.0, 1.0]]], dtype=np.float32),
        velocities=np.zeros((1, 1, 3), dtype=np.float32),
        box_matrix=np.diag([4.0, 4.0, 2.0]).astype(np.float32),
        timestep=0.005,
    )

    probe_xs = np.linspace(1.8, 2.2, 3)
    probe_ys = np.linspace(1.8, 2.2, 3)

    calculator = MultisliceCalculator()
    calculator.setup(
        trajectory,
        aperture=20,
        voltage_eV=100e3,
        sampling=0.5,
        slice_thickness=0.5,
        probe_xs=probe_xs,
        probe_ys=probe_ys,
        max_kx=1.0,
        max_ky=1.0,
        loop_probes=2,
        cache=False,
        ADF=(5, 40),
        keep_wavefunctions=False,
    )
    wf_data, haadf = calculator.run()

    assert tuple(wf_data.array.shape) == (9, 1, 1, 1, 1)
    assert haadf.array.shape == (3, 3)
    assert np.isfinite(haadf.array).all()
    assert haadf.array[-1, -1] > 0
