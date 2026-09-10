"""Tests for the PySlice MCP service layer (pyslice.mcp.service).

Covers the handle registry, structure loading and building, the parameter
advisor's physics rules, workspace safety, and a miniature end-to-end
multislice run with ``.sea`` export — all on the numpy backend, no GPU or
network access required.
"""
import math

import numpy as np
import pytest
from ase import Atoms
from ase.io import write as ase_write

from pyslice.mcp.service import (
    ComputeHAADFInput,
    ComputeTACAWInput,
    PySliceService,
    ResponseFormat,
    SetupMultisliceInput,
    SuggestParametersInput,
    TrajectoryOperation,
    _electron_wavelength_A,
)


@pytest.fixture()
def service(tmp_path):
    """Provide a PySliceService with an isolated workspace."""
    return PySliceService(workspace=tmp_path / "ws")


@pytest.fixture()
def diamond_cif(tmp_path):
    """Write a 2-atom diamond-cell CIF and return its path."""
    a = 3.57
    atoms = Atoms("C2", positions=[[0, 0, 0], [a / 4, a / 4, a / 4]], cell=[a, a, a], pbc=True)
    path = tmp_path / "diamond.cif"
    ase_write(str(path), atoms)
    return path


def test_conventions_cover_units_and_workflow(service):
    conventions = service.get_conventions()
    assert conventions["call_first"] == "pyslice_get_conventions"
    assert "Angstrom" in conventions["units"]["length"]
    assert any("suggest_parameters" in step for step in conventions["workflow"])
    markdown = service.format_response(conventions, ResponseFormat.MARKDOWN)
    assert markdown.startswith("# Result")


def test_load_and_transform_trajectory(service, diamond_cif):
    loaded = service.load_structure(str(diamond_cif), None, None)
    assert loaded["type"] == "Trajectory"
    assert loaded["n_atoms"] == 2

    built = service.transform_trajectory(
        loaded["handle"],
        [
            TrajectoryOperation(op="tile", params={"repeats": [2, 2, 1]}),
            TrajectoryOperation(op="frozen_phonon", params={"n": 3, "sigma_A": 0.05, "seed": 1}),
        ],
        name="built",
    )
    assert built["n_atoms"] == 8
    assert built["n_frames"] == 3
    assert built["applied_operations"] == ["tile", "frozen_phonon"]
    # source trajectory is untouched
    assert service.describe_handle(loaded["handle"])["n_frames"] == 1


def test_transform_tilt_uses_degrees(service, diamond_cif):
    handle = service.load_structure(str(diamond_cif), None, None)["handle"]
    rotated = service.transform_trajectory(
        handle, [TrajectoryOperation(op="tilt", params={"alpha_deg": 90.0})], name="tilted"
    )
    original = service._get(handle)
    tilted = service._get(rotated["handle"])
    # 90 deg about x maps +z onto -y for the second atom's offset
    np.testing.assert_allclose(
        tilted.positions[0, 1], [original.positions[0, 1, 0], -original.positions[0, 1, 2], original.positions[0, 1, 1]],
        atol=1e-5,
    )


def test_suggest_parameters_haadf_rules(service, diamond_cif):
    handle = service.load_structure(str(diamond_cif), None, None)["handle"]
    result = service.suggest_parameters(
        SuggestParametersInput(trajectory_handle=handle, goal="haadf", voltage_eV=100e3)
    )
    suggested = result["suggested"]
    wavelength = _electron_wavelength_A(100e3)
    # sampling from the antialiasing band limit at 1.2x the ADF outer angle
    expected_sampling = wavelength / (3.0 * 1.2 * 200e-3)
    assert suggested["sampling_A"] == pytest.approx(expected_sampling, rel=1e-3)
    # probe step from image Nyquist lambda/(4 alpha)
    assert suggested["probe_step_A"] == pytest.approx(wavelength / (4 * 25e-3), rel=1e-3)
    assert suggested["adf"] == [60.0, 200.0]
    assert suggested["return_layers"] is None
    # 3.57 A cell needs tiling for a 25 mrad probe
    assert suggested["tile_repeats"][0] >= 2
    assert set(result["justification"]) >= {"sampling_A", "slice_thickness_A", "probe_grid"}


def test_suggest_parameters_tacaw_md_plan(service, diamond_cif):
    handle = service.load_structure(str(diamond_cif), None, None)["handle"]
    result = service.suggest_parameters(
        SuggestParametersInput(
            trajectory_handle=handle,
            goal="tacaw",
            target_max_frequency_THz=30.0,
            target_frequency_resolution_THz=0.3,
        )
    )
    plan = result["suggested"]["md_plan"]
    # time Nyquist: frame spacing 1/(2*30 THz) ps; resolution 1/(N*dt)
    assert plan["frame_spacing_ps"] == pytest.approx(1.0 / 60.0, rel=1e-3)
    assert plan["n_frames"] == math.ceil(1.0 / (0.3 * (1.0 / 60.0)))
    assert plan["production_ensemble"] == "nve"


def test_workspace_path_rejects_escape(service):
    with pytest.raises(ValueError, match="escapes the workspace"):
        service._workspace_path("../outside.sea")


def test_unknown_handle_and_wrong_type_errors(service, diamond_cif):
    with pytest.raises(KeyError, match="Unknown handle"):
        service.describe_handle("Trajectory:nope")
    handle = service.load_structure(str(diamond_cif), None, None)["handle"]
    with pytest.raises(TypeError, match="expected MultisliceCalculator"):
        service.run_multislice(handle, False)


def test_multislice_haadf_tacaw_and_export(service, diamond_cif, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep psi_data/ cache inside the test dir
    handle = service.load_structure(str(diamond_cif), None, None)["handle"]
    # Synthetic chronological vibration, not independent frozen-phonon frames.
    from pyslice import Trajectory
    structure = service._get(handle)
    positions = np.repeat(structure.positions, 8, axis=0)
    positions[:, 1, 0] += 0.05 * np.sin(2 * np.pi * np.arange(8) / 8)
    trajectory = Trajectory(structure.atom_types, positions,
                            np.zeros_like(positions), structure.box_matrix, 0.005)
    built = service._register(trajectory, preferred="synthetic-time-series")["handle"]

    setup = service.setup_multislice(
        SetupMultisliceInput(
            trajectory_handle=built,
            aperture_mrad=25.0,
            voltage_eV=100e3,
            sampling_A=0.05,
            slice_thickness_A=1.19,
            probe_grid={"x": [0.0, 2.0, 2], "y": [0.0, 2.0, 2]},
            adf=[60.0, 200.0],
            force_cpu=True,
        )
    )
    assert setup["n_probes"] == 4
    assert setup["grid"]["n_slices"] >= 2

    run = service.run_multislice(setup["handle"], False)
    assert run["wf"]["shape_probe_time_kx_ky_layer"][0] == 4
    assert "haadf" in run  # on-the-fly ADF detector was configured

    haadf = service.compute_haadf(
        ComputeHAADFInput(wf_handle=run["wf"]["handle"], inner_mrad=60, outer_mrad=200, save_png="haadf.png")
    )
    assert haadf["image_shape"] == [2, 2]
    assert (service.workspace / "haadf.png").exists()

    tacaw = service.compute_tacaw(ComputeTACAWInput(wf_handle=run["wf"]["handle"]))
    assert tacaw["frequency_range_THz"][0] < 0 < tacaw["frequency_range_THz"][1] or tacaw["frequency_range_THz"][1] >= 0
    assert service._get(tacaw["handle"]).gain_loss_folded is False
    folded = service.compute_tacaw(ComputeTACAWInput(
        wf_handle=run["wf"]["handle"], fold=True, apply_bose=True, temperature_K=300))
    assert service._get(folded["handle"]).gain_loss_folded is True
    assert service._get(tacaw["handle"]).gain_loss_folded is False

    from pyslice.data.seashell import sea_available
    if not sea_available:
        with pytest.raises(ImportError, match="sea-eco"):
            service.export_sea(tacaw["handle"], "results/tacaw_test")
        return
    exported = service.export_sea(tacaw["handle"], "results/tacaw_test")
    from pyslice.postprocessing.tacaw_data import TACAWData

    reloaded = TACAWData.load(exported["sea_path"])
    assert list(np.shape(reloaded.data)) == tacaw["shape_probe_freq_kx_ky"]


def test_compute_haadf_rejects_bad_angles(service, diamond_cif, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handle = service.load_structure(str(diamond_cif), None, None)["handle"]
    setup = service.setup_multislice(
        SetupMultisliceInput(trajectory_handle=handle, sampling_A=0.5, slice_thickness_A=1.19, force_cpu=True)
    )
    run = service.run_multislice(setup["handle"], False)
    with pytest.raises(ValueError, match="outer_mrad"):
        service.compute_haadf(ComputeHAADFInput(wf_handle=run["wf"]["handle"], inner_mrad=100, outer_mrad=50))


def test_tacaw_requires_multiple_frames(service, diamond_cif, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handle = service.load_structure(str(diamond_cif), None, None)["handle"]
    setup = service.setup_multislice(
        SetupMultisliceInput(trajectory_handle=handle, sampling_A=0.5, slice_thickness_A=1.19, force_cpu=True)
    )
    run = service.run_multislice(setup["handle"], False)
    with pytest.raises(ValueError, match="multi-frame"):
        service.compute_tacaw(ComputeTACAWInput(wf_handle=run["wf"]["handle"]))


def test_build_server_registers_tools():
    mcp_module = pytest.importorskip("mcp", reason="mcp extra not installed")
    from pyslice.mcp.server import build_server

    server = build_server(workspace=None)
    assert type(server).__name__ in ("FastMCP", "MCPServer")


def test_service_md_preserves_fixed_cell_contract(service, diamond_cif, monkeypatch):
    """Pass the material modulus through while rejecting NPT production."""
    from types import SimpleNamespace
    from pydantic import ValidationError
    from pyslice.mcp.service import RunMDInput
    import pyslice.md.molecular_dynamics as md

    handle = service.load_structure(str(diamond_cif), None, None)["handle"]
    with pytest.raises(ValidationError):
        RunMDInput(trajectory_handle=handle, production_ensemble="npt")
    captured = {}
    fake = SimpleNamespace(setup=lambda atoms, **kw: captured.update(kw),
                           run=lambda: service._get(handle))
    monkeypatch.setattr(md, "ORBMDCalculator", lambda **kw: fake)
    service.run_md(RunMDInput(trajectory_handle=handle, ensemble="npt",
                             bulk_modulus_GPa=140, production_ensemble="nve"))
    assert captured["bulk_modulus_GPa"] == 140
    assert captured["production_ensemble"] == "nve"
