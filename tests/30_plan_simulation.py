"""Tests for the prompt→simulation planner (pyslice_plan_simulation).

Encodes Eric's two exemplar prompts as structured-request planning tests:

(a) "I want an atomic-resolution 4D-STEM simulation of a 40 nm thick,
    110-oriented diamond sample with slices every 10 nm."
(b) "I want a vibrational EELS dispersion simulation of graphene out
    to ±2g."

The planner must decode the supplied parameters, fill the rest from the
physics rules with explicit origins, and surface the guessed ones as open
questions for confirmation.
"""
import numpy as np
import pytest
from ase.build import bulk, graphene
from ase.io import write as ase_write

from pyslice.io.build import first_bragg_g
from pyslice.mcp.service import PlanSimulationInput, PySliceService, _electron_wavelength_A


@pytest.fixture()
def service(tmp_path):
    """Service with an isolated workspace."""
    return PySliceService(workspace=tmp_path / "ws")


@pytest.fixture()
def diamond_handle(service, tmp_path):
    """Conventional-cell diamond unit-cell handle."""
    path = tmp_path / "diamond.cif"
    ase_write(str(path), bulk("C", "diamond", a=3.567, cubic=True))
    return service.load_structure(str(path), None, None)["handle"]


@pytest.fixture()
def graphene_handle(service, tmp_path):
    """Graphene unit-cell handle (vacuum along z)."""
    sheet = graphene(a=2.46)
    sheet.cell[2, 2] = 6.7
    path = tmp_path / "graphene.cif"
    ase_write(str(path), sheet)
    return service.load_structure(str(path), None, None)["handle"]


def _by_name(plan):
    return {p["name"]: p for p in plan["parameters"]}


def test_exemplar_a_diamond_110_4dstem(service, diamond_handle):
    plan = service.plan_simulation(PlanSimulationInput(
        technique="4dstem",
        structure_handle=diamond_handle,
        zone_axis=[1, 1, 0],
        thickness_A=400.0,            # 40 nm
        slice_output_interval_A=100.0,  # slices every 10 nm
    ))
    params = _by_name(plan)

    # supplied vs default origins are explicit
    assert params["zone_axis"]["origin"] == "supplied"
    assert params["voltage_eV"]["origin"] == "default"
    # atomic-size probe defaulted to 30 mrad
    assert params["aperture_mrad"]["value"] == 30.0
    assert params["aperture_mrad"]["origin"] == "default"
    # probe step: 10x the Nyquist step of the first Bragg spacing (d/20)
    d1 = plan["structure"]["d_first_A"]
    assert params["scan_step_A"]["value"] == pytest.approx(d1 / 20.0, rel=1e-3)
    # thermal model derived: frozen phonon for 4D-STEM (no phonons requested)
    assert params["thermal"]["value"] == "frozen_phonon"
    assert plan["thermal_plan"]["kind"] == "frozen_phonon"
    # slices every 10 nm through 40 nm -> 4 stored layers incl. exit plane
    layers = plan["simulation_setup"]["return_layers"]
    assert isinstance(layers, list) and len(layers) == 4
    # build plan targets an exact 110 slab
    assert plan["build_plan"]["indices"] == [1, 1, 0]
    assert plan["build_plan"]["thickness_A"] == 400.0
    # unsupplied critical parameters surface as open questions
    open_params = {q["parameter"] for q in plan["open_questions"]}
    assert {"lateral_A", "scan_extent_A"} <= open_params
    # sampling honors the band limit for the largest k requested
    lam = _electron_wavelength_A(100e3)
    k_max = 3.0 * 30e-3 / lam
    assert plan["simulation_setup"]["sampling_A"] == pytest.approx(1.0 / (3.0 * k_max), rel=1e-2)
    assert plan["estimated_wavefunction_GiB"] > 0


def test_exemplar_b_graphene_dispersion_pm_2g(service, graphene_handle):
    plan = service.plan_simulation(PlanSimulationInput(
        technique="tacaw_dispersion",
        structure_handle=graphene_handle,
        k_range_g=2.0,
    ))
    params = _by_name(plan)

    # momentum-resolved dispersion -> parallel beam, derived (not guessed)
    assert params["aperture_mrad"]["value"] == 0.0
    assert params["aperture_mrad"]["origin"] == "derived"
    # +-2g captured exactly
    trajectory = service._get(graphene_handle)
    g1 = first_bragg_g(trajectory.box_matrix)
    assert plan["simulation_setup"]["max_kx"] == pytest.approx(2.0 * g1, rel=1e-3)
    # phonons need MD; production in NVE with a Nyquist-consistent plan
    assert params["thermal"]["value"] == "md"
    md = plan["thermal_plan"]
    assert md["production_ensemble"] == "nve"
    assert md["frame_spacing_ps"] == pytest.approx(1.0 / (2.0 * 30.0), rel=1e-3)
    assert md["n_frames"] * md["frame_spacing_ps"] == pytest.approx(1.0 / 0.3, rel=0.02)
    # exit wave only (no slice interval requested)
    assert plan["simulation_setup"]["return_layers"] == -1
    # lateral size guessed (10 nm floor) and flagged
    assert params["lateral_A"]["value"] >= 100.0
    open_params = {q["parameter"] for q in plan["open_questions"]}
    assert {"lateral_A", "max_frequency_THz"} <= open_params
    # high-symmetry path for the hexagonal cell, in-plane points in 1/A
    assert plan["k_path"]["labels"] == "GMKG"
    K = plan["k_path"]["points_invA"]["K"]
    assert np.hypot(*K) == pytest.approx(2.0 / (3.0 * 2.46), rel=1e-3)
    # full datacube then iso-energy + dispersion visuals
    assert any("iso-energy" in step for step in plan["postprocess_plan"])
    assert any("dispersion" in step.lower() for step in plan["postprocess_plan"])


def test_supplied_values_pass_through_verbatim(service, diamond_handle):
    plan = service.plan_simulation(PlanSimulationInput(
        technique="haadf",
        structure_handle=diamond_handle,
        voltage_eV=200e3,
        aperture_mrad=21.0,
        detector_mrad=[75.0, 210.0],
        lateral_A=50.0,
        scan_extent_A=25.0,
    ))
    params = _by_name(plan)
    for name, value in [("voltage_eV", 200e3), ("aperture_mrad", 21.0), ("lateral_A", 50.0)]:
        assert params[name]["value"] == value
        assert params[name]["origin"] == "supplied"
    assert plan["simulation_setup"]["adf"] == [75.0, 210.0]
    # supplied parameters never appear as open questions
    open_params = {q["parameter"] for q in plan["open_questions"]}
    assert not {"voltage_eV", "aperture_mrad", "detector_mrad"} & open_params
    # HAADF runs on-the-fly ADF without storing wavefunctions
    assert plan["simulation_setup"]["return_layers"] is None
