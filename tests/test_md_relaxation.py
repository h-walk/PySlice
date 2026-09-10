from ase.build import bulk
from ase.calculators.emt import EMT
from ase import units
from ase.md.npt import NPT
from ase.md.langevin import Langevin
from ase.io import read as ase_read
import inspect
import numpy as np
import pytest

from pyslice.md import MDCalculator


class EMTMDCalculator(MDCalculator):
    def __init__(self):
        super().__init__(model_name="emt-test", device="cpu")
        self.setup_calls = 0

    def _setup_calculator(self) -> bool:
        self.setup_calls += 1
        self.calculator = EMT()
        return True


def test_setup_docstring_matches_failed_equilibration_control_flow():
    docstring = " ".join(inspect.getdoc(MDCalculator.setup).split())
    assert "before forcing production" not in docstring
    assert "aborts without starting production" in docstring


def test_relax_structure_reuses_calculator_for_md_setup(tmp_path):
    calc = EMTMDCalculator()
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)

    relaxed = calc.relax_structure(
        atoms,
        fmax=0.2,
        steps=2,
        output_dir=tmp_path,
    )

    assert relaxed is atoms
    assert atoms.calc is calc.calculator
    assert (tmp_path / "relaxation.traj").exists()
    assert (tmp_path / "relaxation.log").exists()
    assert (tmp_path / "relaxed_structure.xyz").exists()
    assert calc.setup_calls == 1

    calc.setup(
        atoms=atoms,
        temperature=300,
        timestep=1.0,
        min_equilibration_steps=1,
        max_equilibration_steps=1,
        production_steps=1,
        output_dir=tmp_path,
        save_xyz=False,
    )

    assert calc.setup_calls == 1
    assert calc.atoms is atoms


def test_relax_structure_can_relax_cell(tmp_path):
    calc = EMTMDCalculator()
    atoms = bulk("Cu", "fcc", a=4.2, cubic=True)
    initial_cell = atoms.cell.array.copy()

    relaxed = calc.relax_structure(
        atoms,
        fmax=0.5,
        steps=3,
        output_dir=tmp_path,
        relax_cell=True,
    )

    assert relaxed is atoms
    assert atoms.calc is calc.calculator
    assert not np.allclose(atoms.cell.array, initial_cell)
    assert (tmp_path / "relaxed_structure.xyz").exists()


def test_md_setup_allows_npt_equilibration_then_defaults_to_nvt_production(tmp_path):
    calc = EMTMDCalculator()
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)

    calc.setup(
        atoms,
        ensemble="npt",
        pressure=2.0,
        bulk_modulus_GPa=140.0,
        output_dir=tmp_path,
    )

    assert isinstance(calc.dyn, NPT)
    assert calc.production_ensemble == "nvt"
    assert np.isclose(calc.dyn.externalstress[0], -2.0 * units.bar)
    assert np.isclose(
        calc.dyn.pfactor_given,
        (75.0 * units.fs) ** 2 * (140.0 * units.GPa),
    )


def test_md_setup_requires_bulk_modulus_for_npt(tmp_path):
    calc = EMTMDCalculator()
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)

    with pytest.raises(ValueError, match="bulk_modulus_GPa"):
        calc.setup(atoms, ensemble="npt", output_dir=tmp_path)


def test_md_setup_rejects_npt_production(tmp_path):
    calc = EMTMDCalculator()
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)

    with pytest.raises(ValueError, match="production_ensemble"):
        calc.setup(
            atoms,
            ensemble="nvt",
            production_ensemble="npt",
            output_dir=tmp_path,
        )


def test_npt_to_nvt_production_returns_fixed_cell_trajectory(tmp_path):
    calc = EMTMDCalculator()
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)
    calc.setup(
        atoms,
        ensemble="npt",
        pressure=1.01325,
        bulk_modulus_GPa=140.0,
        production_steps=2,
        save_interval=1,
        check_interval=1,
        output_dir=tmp_path,
        save_xyz=False,
    )

    # Exercise ASE's variable-cell integrator before freezing its final cell.
    calc.dyn.run(1)
    trajectory = calc.run_production()
    production_frames = ase_read(tmp_path / "production.traj", index=":")

    assert calc.production_ensemble == "nvt"
    assert trajectory.n_frames == 2
    assert all(
        np.allclose(frame.cell.array, production_frames[0].cell.array)
        for frame in production_frames
    )
    assert np.allclose(trajectory.box_matrix, production_frames[0].cell.array)


def test_md_setup_rejects_unknown_ensemble(tmp_path):
    calc = EMTMDCalculator()
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)

    with pytest.raises(ValueError, match="must be one of"):
        calc.setup(atoms, ensemble="canonical-ish", output_dir=tmp_path)


def test_langevin_setup_explicitly_fixes_center_of_mass(tmp_path):
    calc = EMTMDCalculator()
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)

    calc.setup(
        atoms,
        ensemble="nvt",
        output_dir=tmp_path,
        rng=np.random.default_rng(4),
    )

    assert isinstance(calc.dyn, Langevin)
    assert calc.dyn.fix_com is True
    np.testing.assert_allclose(atoms.get_momenta().sum(axis=0), 0.0, atol=1e-12)


def test_md_outputs_are_not_overwritten_by_default(tmp_path):
    calc = EMTMDCalculator()
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)
    calc.setup(atoms, output_dir=tmp_path)
    (tmp_path / "equilibration.log").write_text("existing result")

    with pytest.raises(FileExistsError, match="overwrite"):
        calc.run_equilibration()


def test_md_run_does_not_start_production_after_failed_equilibration(monkeypatch):
    calc = EMTMDCalculator()
    production_called = False

    monkeypatch.setattr(calc, "run_equilibration", lambda: False)

    def production():
        nonlocal production_called
        production_called = True

    monkeypatch.setattr(calc, "run_production", production)

    with pytest.raises(RuntimeError, match="production was not started"):
        calc.run()
    assert production_called is False


def test_failed_equilibration_log_does_not_claim_production_will_continue(tmp_path, caplog):
    calc = EMTMDCalculator()
    atoms = bulk("Cu", "fcc", a=3.8, cubic=True)
    calc.setup(
        atoms,
        min_equilibration_steps=2,
        max_equilibration_steps=1,
        check_interval=1,
        save_interval=1,
        output_dir=tmp_path,
        save_xyz=False,
    )

    assert calc.run_equilibration() is False
    assert "will abort before production" in caplog.text
    assert "Proceeding to production" not in caplog.text
