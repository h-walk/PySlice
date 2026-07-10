import math

import numpy as np
import pytest

from pyslice.optics.column import OpticalColumn
from pyslice.optics.elements import (
    Aberration,
    Aperture,
    BeamTilt,
    FreeSpace,
    Lens,
    SeparableParaxialMap,
)
from pyslice.optics.raytem import (
    RAYTEM_MM_TO_ANGSTROM,
    optical_column_from_raytem,
    raytem_aberrations,
    raytem_dipole_tilt,
    raytem_lens_focal_length,
    raytem_lens_principal_plane_drift,
    raytem_prism_matrices,
    raytem_quadrupole_matrices,
)


class FakeWFData:
    def __init__(self):
        self.calls = []

    def propagate_free_space(self, dz):
        self.calls.append(("free", dz))

    def propagate_through_lens(self, f):
        self.calls.append(("lens", f))

    def applyMask(self, radius, realOrReciprocal="reciprocal"):
        self.calls.append(("aperture", radius, realOrReciprocal))

    def aberrate(self, aberrations):
        self.calls.append(("aberrate", aberrations))

    def propagate_anisotropic_free_space(self, dz_x, dz_y):
        self.calls.append(("anisotropic_free", dz_x, dz_y))

    def propagate_through_astigmatic_lens(self, f_x, f_y):
        self.calls.append(("astigmatic_lens", f_x, f_y))

    def apply_beam_tilt(self, theta_x, theta_y):
        self.calls.append(("tilt", theta_x, theta_y))

    def rotate_real_space(self, angle_rad):
        self.calls.append(("rotate", angle_rad))


def test_lens_uses_principal_plane_drifts():
    wf = FakeWFData()

    Lens(
        f_A=10.0,
        thickness_A=4.0,
        principal_plane_drift_A=2.25,
    ).apply(wf)

    assert wf.calls == [
        ("free", 2.25),
        ("lens", 10.0),
        ("free", 2.25),
    ]


def test_thin_lens_is_single_phase_operation():
    wf = FakeWFData()

    Lens(f_A=10.0).apply(wf)

    assert wf.calls == [("lens", 10.0)]


def test_raytem_lens_focal_length_matches_thin_and_finite_lens_power():
    assert raytem_lens_focal_length(
        {"kind": "Thin lens", "strength": 2.0, "length": 0.0}
    ) == pytest.approx(0.25)

    expected = 1.0 / (0.5 * math.sin(0.5 * 0.2))
    assert raytem_lens_focal_length(
        {"kind": "QLens", "strength": 0.5, "length": 0.2}
    ) == pytest.approx(expected)

    assert math.isinf(raytem_lens_focal_length(
        {"kind": "QLens", "strength": 0.0, "length": 0.2}
    ))


def test_raytem_thick_lens_has_exact_symmetric_abcd_factorization():
    element = {"kind": "QLens", "strength": 0.5, "length": 0.2}
    expected_drift = math.tan(0.5 * 0.2 / 2.0) / 0.5

    assert raytem_lens_principal_plane_drift(element) == pytest.approx(
        expected_drift
    )


def test_aberration_element_calls_wfdata_aberrate():
    wf = FakeWFData()

    Aberration({"C30": 1000.0, "C12": (100.0, 0.25)}).apply(wf)

    assert wf.calls == [("aberrate", {"C30": 1000.0, "C12": (100.0, 0.25)})]


def test_raytem_aberrations_accept_nested_and_top_level_cnm_keys():
    extracted = raytem_aberrations(
        {
            "aberrations": {"C30": 2.0, "C12": [3.0, 0.25]},
            "C10": 4.0,
        },
        coefficient_scale_A=10.0,
    )

    assert extracted == {
        "C30": 20.0,
        "C12": (30.0, 0.25),
        "C10": 40.0,
    }


def test_raytem_json_converts_named_segment_to_optical_column():
    raytem_json = {
        "Microscope name": "toy",
        "Sections": [
            {
                "Section name": "post-sample",
                "position": 0.0,
                "Elements": [
                    {"Element name": "sample", "kind": "Drift", "position": 0.0, "length": 2e-7},
                    {
                        "Element name": "OL",
                        "kind": "QLens",
                        "position": 5e-7,
                        "length": 1e-7,
                        "strength": 5e6,
                        "aberrations": {"C30": 2e-7, "C12": [3e-7, 0.25]},
                    },
                    {"Element name": "screen", "kind": "Drift", "position": 1e-6, "length": 0.0},
                ],
            }
        ],
    }

    column = optical_column_from_raytem(
        raytem_json,
        start="sample",
        start_at="exit",
        stop="screen",
        stop_at="entrance",
    )

    assert isinstance(column, OpticalColumn)
    assert [type(element).__name__ for element in column.elements] == [
        "FreeSpace",
        "Lens",
        "Aberration",
        "FreeSpace",
    ]
    assert column.elements[0].dz_A == pytest.approx(3.0)
    assert column.elements[1].name == "OL"
    assert column.elements[1].thickness_A == pytest.approx(1.0)
    assert column.elements[1].f_A == pytest.approx(
        raytem_lens_focal_length({"strength": 5e6, "length": 1e-7})
        * RAYTEM_MM_TO_ANGSTROM
    )
    assert column.elements[2].aberrations == {
        "C30": 2.0,
        "C12": (3.0, 0.25),
    }
    assert column.elements[3].dz_A == pytest.approx(4.0)

    wf = FakeWFData()
    column.apply(wf)
    exact_drift = raytem_lens_principal_plane_drift(
        {"strength": 5e6, "length": 1e-7}
    ) * RAYTEM_MM_TO_ANGSTROM
    assert wf.calls == [
        ("free", 3.0),
        ("free", pytest.approx(exact_drift)),
        ("lens", pytest.approx(column.elements[1].f_A)),
        ("free", pytest.approx(exact_drift)),
        ("rotate", -0.5),
        ("aberrate", {"C30": 2.0, "C12": (3.0, 0.25)}),
        ("free", pytest.approx(4.0)),
    ]


def test_raytem_defaults_to_millimeters_and_exact_abcd_lenses():
    raytem_json = {
        "Sections": [
            {
                "position": 0.0,
                "Elements": [
                    {"Element name": "start", "kind": "Drift", "position": 0.0, "length": 0.0},
                    {
                        "Element name": "L1",
                        "kind": "QLens",
                        "position": 0.002,
                        "length": 0.01,
                        "strength": 10.0,
                    },
                    {"Element name": "screen", "kind": "Drift", "position": 0.02, "length": 0.0},
                ],
            }
        ]
    }

    column = optical_column_from_raytem(raytem_json, start="start", stop="screen")
    lens = column.elements[1]

    assert column.elements[0].dz_A == pytest.approx(0.002 * RAYTEM_MM_TO_ANGSTROM)
    assert lens.thickness_A == pytest.approx(0.01 * RAYTEM_MM_TO_ANGSTROM)
    assert lens.f_A == pytest.approx(
        raytem_lens_focal_length(raytem_json["Sections"][0]["Elements"][1])
        * RAYTEM_MM_TO_ANGSTROM
    )
    assert lens.principal_plane_drift_A == pytest.approx(
        raytem_lens_principal_plane_drift(
            raytem_json["Sections"][0]["Elements"][1]
        )
        * RAYTEM_MM_TO_ANGSTROM
    )
    assert column.elements[2].dz_A == pytest.approx(0.008 * RAYTEM_MM_TO_ANGSTROM)
    assert column.metadata["raytem_length_unit"] == "mm"


def test_column_applies_elements_in_order():
    wf = FakeWFData()
    column = OpticalColumn([FreeSpace(1.5), Lens(20.0)])

    returned = column.apply(wf)

    assert returned is wf
    assert wf.calls == [("free", 1.5), ("lens", 20.0)]


def test_separable_paraxial_map_factors_each_axis_exactly():
    wf = FakeWFData()
    x_matrix = [[0.8, 2.6], [-0.1, 0.925]]
    y_matrix = [[1.1, 0.8], [0.2, 1.0545454545454545]]

    element = SeparableParaxialMap(x_matrix, y_matrix)
    element.apply(wf)

    assert wf.calls == [
        ("anisotropic_free", pytest.approx(0.75), pytest.approx(0.2727272727)),
        ("astigmatic_lens", pytest.approx(10.0), pytest.approx(-5.0)),
        ("anisotropic_free", pytest.approx(2.0), pytest.approx(0.5)),
    ]


def test_non_symplectic_ray_map_is_rejected():
    with pytest.raises(ValueError, match="not symplectic"):
        SeparableParaxialMap([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 2.0]])


def test_raytem_thin_quadrupole_and_dipole_parameters():
    x_matrix, y_matrix = raytem_quadrupole_matrices(
        {"kind": "Thin quad", "strength": 2.0, "length": 0.0}
    )
    np.testing.assert_allclose(x_matrix, [[1.0, 0.0], [4.0, 1.0]])
    np.testing.assert_allclose(y_matrix, [[1.0, 0.0], [-4.0, 1.0]])
    assert raytem_dipole_tilt(
        {"kind": "Dipole", "strength": 0.2, "length": 2.0, "axis": "y"}
    ) == pytest.approx((0.0, 0.4))


def test_raytem_regular_prism_is_symplectic_but_current_fringe_map_is_not():
    x_matrix, y_matrix = raytem_prism_matrices(
        {"kind": "Prism", "length": 0.2, "radius": 1.0, "strength": 0.1}
    )
    assert np.linalg.det(x_matrix) == pytest.approx(1.0)
    assert np.linalg.det(y_matrix) == pytest.approx(1.0)

    _, fringe_y = raytem_prism_matrices(
        {
            "kind": "Prism",
            "length": 0.2,
            "radius": 1.0,
            "strength": 0.1,
            "K1": 0.2,
        }
    )
    with pytest.raises(ValueError, match="not symplectic"):
        SeparableParaxialMap(np.eye(2), fringe_y)


def test_raytem_adapter_adds_quadrupole_dipole_and_prism_wave_elements():
    raytem_json = {
        "Sections": [
            {
                "position": 0.0,
                "Elements": [
                    {"Element name": "start", "kind": "Drift", "position": 0.0, "length": 0.0},
                    {
                        "Element name": "Q1",
                        "kind": "Thin quad",
                        "position": 1e-6,
                        "length": 0.0,
                        "strength": 100.0,
                    },
                    {
                        "Element name": "D1",
                        "kind": "Thin dipole",
                        "position": 2e-6,
                        "length": 0.0,
                        "strength": 1e-3,
                        "axis": "y",
                    },
                    {
                        "Element name": "P1",
                        "kind": "Prism",
                        "position": 3e-6,
                        "length": 1e-6,
                        "radius": 1e-5,
                        "strength": 0.1,
                    },
                    {"Element name": "screen", "kind": "Drift", "position": 5e-6, "length": 0.0},
                ],
            }
        ]
    }

    column = optical_column_from_raytem(raytem_json, start="start", stop="screen")

    assert [type(element).__name__ for element in column.elements] == [
        "FreeSpace",
        "SeparableParaxialMap",
        "FreeSpace",
        "BeamTilt",
        "FreeSpace",
        "SeparableParaxialMap",
        "FreeSpace",
    ]
    assert isinstance(column.elements[3], BeamTilt)
    assert column.elements[3].theta_y_rad == pytest.approx(1e-3)


def test_raytem_adapter_rejects_current_non_symplectic_thick_quadrupole():
    raytem_json = {
        "Sections": [
            {
                "position": 0.0,
                "Elements": [
                    {"Element name": "start", "kind": "Drift", "position": 0.0, "length": 0.0},
                    {
                        "Element name": "Q1",
                        "kind": "Quad",
                        "position": 1e-6,
                        "length": 1e-6,
                        "strength": 100.0,
                    },
                    {"Element name": "screen", "kind": "Drift", "position": 3e-6, "length": 0.0},
                ],
            }
        ]
    }

    with pytest.raises(ValueError, match="not symplectic"):
        optical_column_from_raytem(raytem_json, start="start", stop="screen")


def test_raytem_stop_at_element_entrance_excludes_that_element():
    raytem_json = {
        "Sections": [
            {
                "position": 0.0,
                "Elements": [
                    {
                        "Element name": "start",
                        "kind": "Drift",
                        "position": 0.0,
                        "length": 0.0,
                    },
                    {
                        "Element name": "L1",
                        "kind": "QLens",
                        "position": 0.01,
                        "length": 0.002,
                        "strength": 10.0,
                    },
                    {
                        "Element name": "screen",
                        "kind": "Drift",
                        "position": 0.02,
                        "length": 0.0,
                    },
                ],
            }
        ]
    }

    entrance_column = optical_column_from_raytem(
        raytem_json, start="start", stop="L1"
    )
    assert [type(element).__name__ for element in entrance_column.elements] == [
        "FreeSpace"
    ]
    assert entrance_column.elements[0].dz_A == pytest.approx(
        0.01 * RAYTEM_MM_TO_ANGSTROM
    )

    exit_column = optical_column_from_raytem(
        raytem_json, start="start", stop="L1", stop_at="exit"
    )
    assert [type(element).__name__ for element in exit_column.elements] == [
        "FreeSpace",
        "Lens",
    ]


@pytest.mark.parametrize("stop_at", ["entrance", "center", "exit"])
def test_numeric_stop_is_an_absolute_coordinate(stop_at):
    raytem_json = {
        "Sections": [
            {
                "position": 0.0,
                "Elements": [
                    {
                        "Element name": "start",
                        "kind": "Drift",
                        "position": 0.0,
                        "length": 0.0,
                    },
                    {
                        "Element name": "L1",
                        "kind": "QLens",
                        "position": 0.01,
                        "length": 0.002,
                        "strength": 10.0,
                    },
                ],
            }
        ]
    }

    column = optical_column_from_raytem(
        raytem_json,
        start="start",
        stop=0.01,
        stop_at=stop_at,
    )

    assert [type(element).__name__ for element in column.elements] == [
        "FreeSpace"
    ]
    assert column.elements[0].dz_A == pytest.approx(
        0.01 * RAYTEM_MM_TO_ANGSTROM
    )


def test_named_boundaries_preserve_transfer_order_at_a_shared_plane():
    raytem_json = {
        "Sections": [
            {
                "position": 0.0,
                "Elements": [
                    {
                        "Element name": "start",
                        "kind": "Drift",
                        "position": 0.0,
                        "length": 0.0,
                    },
                    {
                        "Element name": "L1",
                        "kind": "Thin lens",
                        "position": 0.0,
                        "length": 0.0,
                        "strength": 2.0,
                    },
                    {
                        "Element name": "screen",
                        "kind": "Drift",
                        "position": 0.0,
                        "length": 0.0,
                    },
                    {
                        "Element name": "L2",
                        "kind": "Thin lens",
                        "position": 0.0,
                        "length": 0.0,
                        "strength": 3.0,
                    },
                    {
                        "Element name": "end",
                        "kind": "Drift",
                        "position": 0.0,
                        "length": 0.0,
                    },
                ],
            }
        ]
    }

    column = optical_column_from_raytem(
        raytem_json,
        start="start",
        stop="screen",
    )

    assert [type(element).__name__ for element in column.elements] == ["Lens"]
    assert column.elements[0].name == "L1"

    through_first_lens = optical_column_from_raytem(
        raytem_json,
        start="start",
        stop="L1",
        stop_at="exit",
    )
    assert [element.name for element in through_first_lens.elements] == ["L1"]

    after_first_lens = optical_column_from_raytem(
        raytem_json,
        start="L1",
        start_at="exit",
        stop="end",
    )
    assert [element.name for element in after_first_lens.elements] == ["L2"]


@pytest.mark.parametrize(
    "segment",
    [
        {"start": "L1", "start_at": "center", "stop": "screen"},
        {"start": "start", "stop": "L1", "stop_at": "center"},
        {"start": 0.011, "stop": "screen"},
    ],
)
def test_raytem_segment_rejects_boundaries_inside_finite_elements(segment):
    raytem_json = {
        "Sections": [
            {
                "position": 0.0,
                "Elements": [
                    {
                        "Element name": "start",
                        "kind": "Drift",
                        "position": 0.0,
                        "length": 0.0,
                    },
                    {
                        "Element name": "L1",
                        "kind": "QLens",
                        "position": 0.01,
                        "length": 0.004,
                        "strength": 10.0,
                    },
                    {
                        "Element name": "screen",
                        "kind": "Drift",
                        "position": 0.02,
                        "length": 0.0,
                    },
                ],
            }
        ]
    }

    with pytest.raises(ValueError, match="cuts through finite RayTEM element 'L1'"):
        optical_column_from_raytem(raytem_json, **segment)


def test_raytem_calibrated_drift_uses_effective_length_and_nominal_z():
    raytem_json = {
        "Sections": [
            {
                "position": 0.0,
                "Elements": [
                    {
                        "Element name": "start",
                        "kind": "Source",
                        "position": 0.0,
                        "length": 0.0,
                    },
                    {
                        "Element name": "Dcal",
                        "kind": "Drift",
                        "position": 0.0,
                        "length": 0.01,
                        "calibration": 0.4,
                    },
                    {
                        "Element name": "screen",
                        "kind": "Drift",
                        "position": 0.01,
                        "length": 0.0,
                    },
                ],
            }
        ]
    }
    column = optical_column_from_raytem(
        raytem_json, start="start", stop="screen"
    )

    assert len(column.elements) == 1
    drift = column.elements[0]
    assert isinstance(drift, FreeSpace)
    assert drift.dz_A == pytest.approx(0.004 * RAYTEM_MM_TO_ANGSTROM)
    assert drift.metadata["physical_dz_A"] == pytest.approx(
        0.01 * RAYTEM_MM_TO_ANGSTROM
    )

    wf = FakeWFData()
    propagation = column.propagate(wf, record=True)
    assert wf.calls == [("free", pytest.approx(0.004 * RAYTEM_MM_TO_ANGSTROM))]
    assert propagation.planes[-1].z_A == pytest.approx(
        0.01 * RAYTEM_MM_TO_ANGSTROM
    )


def test_raytem_reciprocal_aperture_mode_is_rejected():
    raytem_json = {
        "Sections": [
            {
                "position": 0.0,
                "Elements": [
                    {
                        "Element name": "A1",
                        "kind": "Aperture",
                        "position": 0.0,
                        "length": 0.0,
                        "radius": 0.01,
                    }
                ],
            }
        ]
    }

    column = optical_column_from_raytem(raytem_json)
    assert len(column.elements) == 1
    assert isinstance(column.elements[0], Aperture)
    assert column.elements[0].radius == pytest.approx(
        0.01 * RAYTEM_MM_TO_ANGSTROM
    )

    with pytest.raises(ValueError, match="physical lengths"):
        optical_column_from_raytem(raytem_json, aperture_space="reciprocal")


def test_recorded_aberration_plane_stays_at_finite_lens_exit():
    raytem_json = {
        "Sections": [
            {
                "position": 0.0,
                "Elements": [
                    {
                        "Element name": "start",
                        "kind": "Drift",
                        "position": 0.0,
                        "length": 0.0,
                    },
                    {
                        "Element name": "L1",
                        "kind": "QLens",
                        "position": 0.01,
                        "length": 0.002,
                        "strength": 10.0,
                        "aberrations": {"C30": 1e-4},
                    },
                    {
                        "Element name": "screen",
                        "kind": "Drift",
                        "position": 0.02,
                        "length": 0.0,
                    },
                ],
            }
        ]
    }
    column = optical_column_from_raytem(
        raytem_json, start="start", stop="screen"
    )

    lens = next(element for element in column.elements if isinstance(element, Lens))
    aberration = next(
        element for element in column.elements if isinstance(element, Aberration)
    )
    expected_exit_A = lens.z_A + lens.thickness_A
    assert aberration.z_A == pytest.approx(expected_exit_A)

    propagation = column.propagate(FakeWFData(), record=True)
    plane_z = [plane.z_A for plane in propagation.planes]
    assert plane_z == sorted(plane_z)
    assert next(
        plane.z_A for plane in propagation.planes if plane.element is aberration
    ) == pytest.approx(expected_exit_A)
