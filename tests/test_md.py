"""Molecular-dynamics regressions: NPT barostat units."""
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


def test_npt_barostat_params_have_physical_units():
    from ase import units
    from pyslice.md.molecular_dynamics import MDCalculator
    externalstress, pfactor = MDCalculator._npt_barostat_params(1.01325, 100.0)
    # externalstress is ~1 atm in eV/A^3, not 1.01325 eV/A^3 (~162 GPa)
    np.testing.assert_allclose(externalstress / units.bar, 1.01325, rtol=1e-6)
    assert externalstress < 1e-6                       # ~6.3e-7, not ~1
    # pfactor = ptime^2 * B (was 75*fs**2, ~1e5x too small)
    np.testing.assert_allclose(pfactor, (75 * units.fs) ** 2 * (100.0 * units.GPa))


# ---------------------------------------------------------------------------
# MDConvergenceChecker — equilibration logic (no ML potential required)
# ---------------------------------------------------------------------------

def _atoms_at(temperature_K, energy_per_atom_eV, n=8):
    """A small Atoms object with an EXACT temperature and a fixed energy.

    Momenta p_i = sqrt(3 m_i kB T) along x give kinetic energy 1.5 kB T per
    atom, so ase's get_temperature() (2*ekin/(dof*kB), dof=3N) returns T
    exactly. A SinglePointCalculator supplies the (per-atom) potential energy.
    """
    from ase import Atoms, units
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms = Atoms("Cu" * n,
                  positions=[[i * 3.0, 0.0, 0.0] for i in range(n)],
                  cell=[n * 3.0, 10.0, 10.0], pbc=True)
    masses = atoms.get_masses()
    momenta = np.zeros((n, 3))
    momenta[:, 0] = np.sqrt(3.0 * masses * units.kB * temperature_K)
    atoms.set_momenta(momenta)
    atoms.calc = SinglePointCalculator(
        atoms, energy=energy_per_atom_eV * n, forces=np.zeros((n, 3)))
    return atoms


def _checker(**kw):
    from pyslice.md.molecular_dynamics import MDConvergenceChecker
    kw.setdefault("target_temperature", 300.0)
    kw.setdefault("min_steps", 10)
    kw.setdefault("temperature_window", 5)
    kw.setdefault("energy_window", 5)
    return MDConvergenceChecker(**kw)


def test_convergence_checker_reads_temperature_and_energy():
    checker = _checker()
    checker.update(_atoms_at(300.0, -3.5))
    assert checker.steps == 1
    assert checker.temperatures[-1] == pytest.approx(300.0, rel=1e-6)
    assert checker.energies[-1] == pytest.approx(-3.5, rel=1e-6)   # per atom
    assert checker.reached_target is True                          # 300 >= 300


def test_convergence_requires_minimum_steps():
    checker = _checker(min_steps=10)
    for _ in range(3):
        checker.update(_atoms_at(300.0, -3.5))
    converged, reason = checker.check_convergence()
    assert converged is False and "Not enough steps" in reason


def test_convergence_requires_target_temperature_reached():
    checker = _checker(target_temperature=300.0, min_steps=5)
    for _ in range(8):                       # always below target
        checker.update(_atoms_at(100.0, -3.5))
    converged, reason = checker.check_convergence()
    assert converged is False
    assert "not yet reached" in reason and checker.reached_target is False


def test_convergence_true_for_stable_series():
    checker = _checker(target_temperature=300.0, min_steps=10)
    for _ in range(20):                      # dead steady at target
        checker.update(_atoms_at(300.0, -3.5))
    converged, reason = checker.check_convergence()
    assert bool(converged), reason           # check_convergence returns a numpy bool


def test_convergence_false_when_energy_drifts():
    # Temperature pinned at target, but a large per-step energy swing keeps the
    # relative energy std above threshold -> not converged.
    checker = _checker(target_temperature=300.0, min_steps=10, energy_window=5)
    for i in range(20):
        checker.update(_atoms_at(300.0, -3.5 + (0.5 if i % 2 else -0.5)))
    converged, _ = checker.check_convergence()
    assert not converged


def test_get_statistics_uses_window():
    checker = _checker(temperature_window=3, energy_window=3, min_steps=1)
    temps = [290.0, 295.0, 300.0, 305.0, 310.0]
    for T in temps:
        checker.update(_atoms_at(T, -3.0))
    stats = checker.get_statistics()
    assert stats["total_steps"] == len(temps)
    # windowed over the last 3 samples (300, 305, 310)
    assert stats["temperature_mean"] == pytest.approx(305.0, rel=1e-4)
    assert stats["temperature_std"] == pytest.approx(np.std([300, 305, 310]), rel=1e-4)
