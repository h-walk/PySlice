import numpy as np
import pytest

from pyslice import (
    ProbeAberrationModel,
    simulate_raytem_wave,
)
from pyslice.backend import NumpyBackend, TORCH_AVAILABLE, TorchBackend
from pyslice.optics.elements import Lens
from pyslice.postprocessing.wf_data import WFData


def _raytem_column():
    return {
        "Microscope name": "aberration placement",
        "Sections": [
            {
                "Section name": "column",
                "position": 0.0,
                "Elements": [
                    {
                        "Element name": "gun",
                        "kind": "Source",
                        "position": 0.0,
                        "length": 0.0,
                    },
                    {
                        "Element name": "upstream",
                        "kind": "Drift",
                        "position": 0.0,
                        "length": 1e-6,
                    },
                    {
                        "Element name": "sample",
                        "kind": "Drift",
                        "position": 1e-6,
                        "length": 0.0,
                    },
                    {
                        "Element name": "downstream",
                        "kind": "Drift",
                        "position": 1e-6,
                        "length": 1e-6,
                    },
                    {
                        "Element name": "detector",
                        "kind": "Drift",
                        "position": 2e-6,
                        "length": 0.0,
                    },
                ],
            }
        ],
    }


def test_probe_aberration_model_validates_coefficients_and_composition_policy():
    with pytest.raises(ValueError, match="Invalid aberration key"):
        ProbeAberrationModel({"spherical": 1.0})
    with pytest.raises(TypeError, match="replaces_upstream"):
        ProbeAberrationModel(
            {}, replaces_upstream_element_aberrations="yes"
        )


def test_angular_aperture_uses_physical_semiangle():
    wave = WFData.from_probe(
        extent_A=64.0,
        sampling=1.0,
        voltage_eV=100e3,
        aperture=0.0,
        backend=NumpyBackend(),
    )
    wave.array[...] = 1.0

    wave.apply_angular_aperture(5.0)

    radius = np.sqrt(wave.kxs[:, None] ** 2 + wave.kys[None, :] ** 2)
    expected = radius <= 5e-3 / float(wave.probe.wavelength)
    np.testing.assert_array_equal(wave.array[0, 0, :, :, 0] != 0, expected)


def test_probe_aberration_model_warns_for_under_sampled_phase():
    wave = WFData.from_probe(
        extent_A=32.0,
        sampling=1.0,
        voltage_eV=100e3,
        aperture=10.0,
        backend=NumpyBackend(),
    )
    model = ProbeAberrationModel({"C30": 1e12}, semiangle_mrad=10.0)

    report = model.phase_sampling_report(wave)

    assert report["max_phase_step_rad"] > np.pi / 2.0
    with pytest.warns(RuntimeWarning, match="under-sampled"):
        model.apply(wave)


def test_simulation_inserts_measured_aberrations_at_reference_plane():
    model = ProbeAberrationModel(
        {"C30": 1e6},
        reference_plane="sample",
        semiangle_mrad=5.0,
    )

    result = simulate_raytem_wave(
        _raytem_column(),
        start="gun",
        stop="detector",
        voltage_eV=100e3,
        extent_A=64.0,
        sampling_A=1.0,
        convergence_mrad=10.0,
        probe_aberrations=model,
        record=True,
    )

    names = [plane.name for plane in result.planes]
    aberration_index = names.index(model.name)
    assert names[aberration_index - 1] == "upstream"
    assert names[aberration_index + 1] == "downstream"
    assert result.planes[aberration_index].z_A == pytest.approx(10.0)
    assert result.column.metadata["probe_aberration_reference_z_raytem"] == 1e-6
    assert result.column.metadata["probe_aberration_model"]["semiangle_mrad"] == 5.0


def test_probe_aberration_reference_must_be_inside_selected_segment():
    model = ProbeAberrationModel(
        {"C30": 1e6},
        reference_plane="sample",
    )

    with pytest.raises(ValueError, match="outside the selected RayTEM segment"):
        simulate_raytem_wave(
            _raytem_column(),
            start=1.1e-6,
            stop="detector",
            voltage_eV=100e3,
            extent_A=32.0,
            sampling_A=1.0,
            probe_aberrations=model,
        )


def _raytem_column_with_local_lens_aberrations():
    return {
        "Sections": [
            {
                "position": 0.0,
                "Elements": [
                    {"Element name": "gun", "kind": "Source", "position": 0.0},
                    {
                        "Element name": "upstream lens",
                        "kind": "Thin lens",
                        "position": 0.5e-6,
                        "length": 0.0,
                        "strength": 0.01,
                        "aberrations": {"C30": 1e-7},
                    },
                    {
                        "Element name": "sample",
                        "kind": "Drift",
                        "position": 1e-6,
                        "length": 0.0,
                    },
                    {
                        "Element name": "downstream lens",
                        "kind": "Thin lens",
                        "position": 1.5e-6,
                        "length": 0.0,
                        "strength": 0.01,
                        "aberrations": {"C30": 2e-7},
                    },
                    {
                        "Element name": "detector",
                        "kind": "Drift",
                        "position": 2e-6,
                        "length": 0.0,
                    },
                ],
            }
        ]
    }


def test_system_model_replaces_upstream_but_keeps_downstream_lens_aberrations():
    model = ProbeAberrationModel({"C12": 10.0}, reference_plane="sample")

    result = simulate_raytem_wave(
        _raytem_column_with_local_lens_aberrations(),
        start="gun",
        stop="detector",
        voltage_eV=100e3,
        convergence_mrad=5.0,
        extent_A=64.0,
        sampling_A=1.0,
        probe_aberrations=model,
        record=False,
    )

    lenses = {
        element.name: element
        for element in result.column.elements
        if isinstance(element, Lens)
    }
    assert lenses["upstream lens"].aberrations == {}
    assert lenses["downstream lens"].aberrations == {"C30": 2.0}
    assert not result.column.metadata["upstream_element_aberrations_included"]
    assert result.column.metadata["downstream_element_aberrations_included"]


def test_system_model_can_compose_with_upstream_lens_aberrations():
    model = ProbeAberrationModel(
        {"C12": 10.0},
        reference_plane="sample",
        replaces_upstream_element_aberrations=False,
    )

    result = simulate_raytem_wave(
        _raytem_column_with_local_lens_aberrations(),
        start="gun",
        stop="detector",
        voltage_eV=100e3,
        convergence_mrad=5.0,
        extent_A=64.0,
        sampling_A=1.0,
        probe_aberrations=model,
        record=False,
    )

    lenses = {
        element.name: element
        for element in result.column.elements
        if isinstance(element, Lens)
    }
    assert lenses["upstream lens"].aberrations == {"C30": 1.0}
    assert lenses["downstream lens"].aberrations == {"C30": 2.0}
    assert result.column.metadata["upstream_element_aberrations_included"]


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="Torch is optional")
def test_probe_aberrations_preserve_torch_backend():
    wave = WFData.from_probe(
        extent_A=32.0,
        sampling=1.0,
        voltage_eV=100e3,
        aperture=5.0,
        backend=TorchBackend(device="cpu"),
    )
    model = ProbeAberrationModel({"C30": 1e5}, semiangle_mrad=4.0)

    model.apply(wave)

    assert wave.array.device.type == "cpu"
