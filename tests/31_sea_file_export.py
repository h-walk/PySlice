"""Tests for SEAFile packaging with Materials provenance (pyslice_export_sea_file).

Verifies the agreed layout: simulation results in ``SEAFile.Simulations``,
the unit cell as a *Material* entry and the built structure as a *Sample*
entry in ``SEAFile.Materials`` (both in the ``atomic-structure`` profile v1), the build record under
``Sample.Metadata.build``, database info under ``Metadata.Database``, and
the Sample's SEAID rooted at the Material's (the provenance relation) —
all surviving a ``.sea`` round-trip through sea-eco's generic loader.
"""
import json

import numpy as np
import pytest
from ase.build import bulk
from ase.io import write as ase_write

from pyslice.data.seashell import sea_available

pytestmark = pytest.mark.skipif(not sea_available, reason="compatible sea-eco unavailable")

from pyslice.mcp.service import (
    BuildSlabInput,
    ExportSeaFileInput,
    PySliceService,
    RenderSignalInput,
    SetupMultisliceInput,
    TrajectoryOperation,
)


def _norm_id(value) -> str:
    """Normalize a SEAID string for comparison (Crockford O/0, hyphens)."""
    return str(value).replace("-", "").upper().replace("O", "0")


@pytest.fixture()
def service(tmp_path):
    """Service with an isolated workspace."""
    return PySliceService(workspace=tmp_path / "ws")


@pytest.fixture()
def packaged(service, tmp_path, monkeypatch):
    """Run a miniature pipeline and export a SEAFile; return its parts."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "diamond.cif"
    ase_write(str(path), bulk("C", "diamond", a=3.567))
    unit = service.load_structure(str(path), None, None)["handle"]
    slab = service.build_slab(BuildSlabInput(
        structure_handle=unit, indices=[1, 1, 0], layers=2, repeats=[2, 2], name="dia110"))
    sample = service.transform_trajectory(
        slab["handle"],
        [TrajectoryOperation(op="frozen_phonon", params={"n": 2, "sigma_A": 0.05, "seed": 7})],
        name="dia110-fp",
    )
    setup = service.setup_multislice(SetupMultisliceInput(
        trajectory_handle=sample["handle"], sampling_A=0.5, slice_thickness_A=1.26, force_cpu=True))
    run = service.run_multislice(setup["handle"], False)
    exported = service.export_sea_file(ExportSeaFileInput(
        filename="results/demo",
        signal_handles=[run["wf"]["handle"]],
        sample_handle=sample["handle"],
        material_handle=unit,
        name="demo package",
        material_name="C (diamond)",
    ))
    return service, run, exported


def test_sea_file_round_trip_materials_and_provenance(packaged):
    service, run, exported = packaged
    from pySEA.sea_eco.io import load as sea_load

    sea_file = sea_load(exported["sea_path"])
    names = [d.name for d in sea_file.Materials.datasets]
    assert names == ["C (diamond)", "Sample"]
    assert len(sea_file.Simulations) == 1

    material = sea_file.Materials.datasets[0]
    sample = sea_file.Materials.datasets[1]
    # provenance relation: Sample roots at the Material
    assert _norm_id(sample.Provenance.root) == _norm_id(material.Provenance)
    assert _norm_id(exported["material_seaid"]) == _norm_id(material.Provenance)

    # Material metadata: formula + box; Sample metadata: build record
    assert material.metadata.Material.formula == "C2"
    assert material.metadata.Material.kind == "Material"
    build = sample.metadata.build
    assert build.indices is not None
    operations = [json.loads(str(op)) for op in build.operations]
    assert operations[-1]["op"] == "frozen_phonon"

    # atomic-structure profile v1 (sea-eco `signal-containers` schema):
    # a marked SignalCollection with `atoms` and `cell` SignalSets.
    from pySEA.sea_eco.signal_containers import validate_atomic_structure

    validate_atomic_structure(sample)
    assert sample.schema_id == "signal-containers"
    assert sample.schema_profile == "atomic-structure"
    assert sample.schema_version == 1
    assert sample.get_dataset_names() == ["atoms", "cell"]

    position = sample["atoms"]["position"]
    # multi-frame -> contextual form: (time, atom, coordinate)
    assert position.dimensions.get_names() == ["time", "atom", "coordinate"]
    assert position.data.shape == (2, position.data.shape[1], 3)
    # units live on the scalar quantity, not the coordinate axis
    assert position.signal_quantities.dimensions[0].units == "Å"
    assert position.dimensions["coordinate"].units in ("", None)
    # categorical selection works per CONT-2
    assert position(coordinate="x").name == "x"

    elements = sample["atoms"]["element"]
    assert elements.dimensions.get_names() == ["atom"]
    assert set(str(e) for e in elements.data) == {"C"}
    from pyslice.backend import to_numpy
    assert to_numpy(sample["atoms"]["clamp_boundary_condition"].data).dtype == bool
    # CONT-1: a member view exposes the exact registry axis object
    assert elements.dimensions["atom"] is sample["atoms"].dimensions["atom"]

    cell = sample["cell"]["cell"]
    assert cell.dimensions.get_names() == ["cell_vector", "coordinate"]
    assert [str(v) for v in cell.dimensions["cell_vector"].values] == ["a", "b", "c"]
    assert sample["cell"]["periodic_boundary_condition"].data.all()


def test_simulation_signal_is_plain_and_calibrated(packaged):
    service, run, exported = packaged
    from pySEA.sea_eco.architecture.base_structure import Signal
    from pySEA.sea_eco.io import load as sea_load

    sea_file = sea_load(exported["sea_path"])
    simulation = sea_file.Simulations.datasets[0]
    assert type(simulation).__name__ == "Signal"  # readable without PySlice
    wf = service._get(run["wf"]["handle"])
    np.testing.assert_allclose(np.asarray(simulation.data), wf.data)
    dim_names = [d.name for d in simulation._local_dimensions.dimensions]
    assert dim_names == ["probe", "time", "kx", "ky", "layer"]


def test_render_signal_uses_sea_eco(packaged):
    service, run, exported = packaged
    result = service.render_signal(RenderSignalInput(handle=run["wf"]["handle"], filename="wf.png"))
    assert result["renderer"] == "sea-eco matplotlib"
    assert result["artifact_path"].endswith("wf.png")


def test_database_source_flows_into_material_metadata(service, tmp_path, monkeypatch):
    import pyslice.io.databases as databases

    monkeypatch.setattr(
        databases, "_http_get",
        lambda url, headers=None, timeout=30.0: (
            b"data_test\n_symmetry_space_group_name_H-M 'P 1'\n"
            b"_cell_length_a 3.0\n_cell_length_b 3.0\n_cell_length_c 3.0\n"
            b"_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
            b"loop_\n_atom_site_type_symbol\n_atom_site_label\n"
            b"_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n_atom_site_occupancy\n"
            b"Si Si1 0 0 0 1\n"
        ),
    )
    fetched = service.fetch_structure("cod", "12345", None, True, None, None)
    structure = service._trajectory_to_atomic_structure(
        service._get(fetched["handle"]), name="Material", kind="Material",
        source=service._source_info[fetched["handle"]],
    )
    assert structure.metadata.Database.provider == "cod"
    assert structure.metadata.Database.entry_id == "12345"
    # single-frame -> static profile form (no context axis)
    assert structure["atoms"]["position"].dimensions.get_names() == ["atom", "coordinate"]
