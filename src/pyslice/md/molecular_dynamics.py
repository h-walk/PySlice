"""
Molecular dynamics module for PySlice package.

Provides a base MDCalculator class and backend-specific subclasses
(ORBMDCalculator, FAIRChemMDCalculator) for running MD simulations
with machine-learning force fields.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
import logging,traceback
import time
from typing import Optional, Dict, List, TYPE_CHECKING
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.npt import NPT
from ase.optimize import BFGS, FIRE, LBFGS
from ase.filters import FrechetCellFilter
from ase import units
from ase.io import write, Trajectory as ASETrajectory
from ase import Atoms

from ..multislice.trajectory import Trajectory
from ..io.loader import Loader

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit import InferenceSettings

logger = logging.getLogger(__name__)

__all__ = ['MDCalculator', 'ORBMDCalculator', 'FAIRChemMDCalculator', 'MDConvergenceChecker', 'analyze_md_trajectory']


class MDConvergenceChecker:
    """
    Checks for MD equilibration/convergence based on multiple criteria.
    """
    def __init__(self,
                 target_temperature: float,
                 temperature_window: int = 50,
                 energy_window: int = 50,
                 temp_threshold: float = 5.0,  # K
                 temp_tolerance: float = 20.0,  # K
                 energy_threshold: float = 0.01,  # eV/atom relative std
                 min_steps: int = 100):
        """
        Initialize convergence checker.

        Args:
            target_temperature: Target temperature for the simulation (K)
            temperature_window: Number of steps to check for temp convergence
            energy_window: Number of steps to check for energy convergence
            temp_threshold: Max std dev of temperature (K)
            temp_tolerance: Max deviation of mean temperature from target (K)
            energy_threshold: Max relative std dev of energy
            min_steps: Minimum steps before checking convergence
        """
        self.target_temperature = target_temperature
        self.temp_window = temperature_window
        self.energy_window = energy_window
        self.temp_threshold = temp_threshold
        self.temp_tolerance = temp_tolerance
        self.energy_threshold = energy_threshold
        self.min_steps = min_steps

        self.temperatures = []
        self.energies = []
        self.steps = 0
        self.reached_target = False  # Track if we've ever hit the target temperature

    def update(self, atoms: Atoms):
        """Update with current state."""
        self.steps += 1

        temp = atoms.get_temperature()
        energy = atoms.get_potential_energy() / len(atoms)  # per atom

        self.temperatures.append(temp)
        self.energies.append(energy)

        # Check if instantaneous temperature has ever reached target
        if not self.reached_target and temp >= self.target_temperature:
            self.reached_target = True
            logger.info(f"Temperature reached target: T={temp:.1f}K (target: {self.target_temperature}K)")

    def check_convergence(self):
        """Check if system has converged/equilibrated."""
        if self.steps < self.min_steps:
            return False, "Not enough steps"

        # Must have hit the target temperature at least once
        if not self.reached_target:
            max_temp = max(self.temperatures) if self.temperatures else 0
            return False, f"max(T)={max_temp:.1f}K - target {self.target_temperature:.1f}K not yet reached"

        # Check temperature stability
        recent_temps = self.temperatures[-self.temp_window:]
        temp_mean = np.mean(recent_temps)
        temp_std = np.std(recent_temps)
        temp_stable = temp_std < self.temp_threshold

        # Check temperature is near target
        temp_deviation = abs(temp_mean - self.target_temperature)
        temp_at_target = temp_deviation < self.temp_tolerance

        # Check energy stability
        recent_energies = self.energies[-self.energy_window:]
        energy_mean = np.mean(recent_energies)
        energy_std = np.std(recent_energies)
        rel_energy_std = energy_std / abs(energy_mean) if energy_mean != 0 else float('inf')
        energy_converged = rel_energy_std < self.energy_threshold

        converged = temp_stable and temp_at_target and energy_converged

        status = (f"<T>={temp_mean:.1f}K (target={self.target_temperature:.1f}+/-{self.temp_tolerance:.1f}K), "
                 f"std_T={temp_std:.2f}K (target<{self.temp_threshold}K), "
                 f"std_E/|<E>|={rel_energy_std:.4f} (target<{self.energy_threshold})")

        return converged, status

    def get_statistics(self) -> Dict:
        """Return statistics from converged portion."""
        converged_temps = self.temperatures[-self.temp_window:]
        converged_energies = self.energies[-self.energy_window:]

        return {
            'temperature_mean': np.mean(converged_temps),
            'temperature_std': np.std(converged_temps),
            'energy_mean': np.mean(converged_energies),
            'energy_std': np.std(converged_energies),
            'total_steps': self.steps
        }


class MDCalculator:
    """
    Base class for molecular dynamics calculators.

    Subclasses must implement ``__init__`` (to store backend-specific
    options) and ``_setup_calculator`` (to load the ML force-field).

    The generic MD workflow — ``setup``, ``run_equilibration``,
    ``run_production``, ``run`` — lives here and is shared by all backends.
    """

    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Initialize base MD calculator.

        Args:
            model_name: Name/identifier of the ML model (used in log messages)
            device: Device for calculations ('cpu', 'cuda', 'mps', etc.)
        """
        self.model_name = model_name
        self.device = device
        self.calculator = None

    def _setup_calculator(self) -> bool:
        """Load the ML force-field calculator. Subclasses must override."""
        raise NotImplementedError(
            "Subclasses must implement _setup_calculator(). "
            "Use ORBMDCalculator or FAIRChemMDCalculator instead of MDCalculator directly."
        )

    def _ensure_calculator(self):
        """Load the backend calculator if it has not already been initialized."""
        if self.calculator is None:
            if not self._setup_calculator():
                raise RuntimeError(f"Failed to setup calculator ({self.model_name})")
        return self.calculator

    def relax_structure(
        self,
        atoms: Atoms,
        fmax: float = 0.05,
        steps: int = 200,
        optimizer: str = 'FIRE',
        output_dir: Optional[Path] = None,
        trajectory_file: Optional[str] = 'relaxation.traj',
        logfile: Optional[str] = 'relaxation.log',
        final_file: Optional[str] = 'relaxed_structure.xyz',
        relax_cell: bool = False,
        cell_filter_kwargs: Optional[Dict] = None,
    ) -> Atoms:
        """
        Relax atomic positions with the backend force field before running MD.

        This is a zero-temperature geometry optimization. It does not initialize
        velocities and does not change the MD timestep or ensemble settings.
        By default, the simulation cell is held fixed. Set ``relax_cell=True``
        to optimize the cell with ASE's FrechetCellFilter.

        Args:
            atoms: ASE Atoms object to relax in place
            fmax: Maximum force convergence criterion in eV/Angstrom
            steps: Maximum optimizer steps
            optimizer: ASE optimizer name: 'FIRE', 'BFGS', or 'LBFGS'
            output_dir: Directory for optimizer outputs
            trajectory_file: Optional optimizer trajectory filename
            logfile: Optional optimizer log filename
            final_file: Optional final relaxed structure filename
            relax_cell: If True, relax both atomic positions and the cell
            cell_filter_kwargs: Optional keyword arguments passed to
                                ase.filters.FrechetCellFilter

        Returns:
            The same ASE Atoms object, after relaxation.
        """
        self._ensure_calculator()
        atoms.calc = self.calculator

        output_path = Path(output_dir) if output_dir is not None else Path.cwd()
        output_path.mkdir(parents=True, exist_ok=True)

        optimizer_classes = {
            'fire': FIRE,
            'bfgs': BFGS,
            'lbfgs': LBFGS,
        }
        optimizer_cls = optimizer_classes.get(optimizer.lower())
        if optimizer_cls is None:
            raise ValueError(
                f"Unknown optimizer {optimizer!r}. Use one of: FIRE, BFGS, LBFGS."
            )

        trajectory_path = None if trajectory_file is None else output_path / trajectory_file
        logfile_path = None if logfile is None else output_path / logfile

        initial_energy = atoms.get_potential_energy()
        initial_cell = atoms.cell.array.copy()
        relaxation_target = atoms
        if relax_cell:
            try:
                atoms.get_stress()
            except Exception as exc:
                raise RuntimeError(
                    "relax_cell=True requires a calculator that provides stress. "
                    f"{self.model_name} did not return stress for this structure."
                ) from exc
            filter_kwargs = {} if cell_filter_kwargs is None else dict(cell_filter_kwargs)
            relaxation_target = FrechetCellFilter(atoms, **filter_kwargs)

        logger.info("="*70)
        logger.info("STRUCTURAL RELAXATION")
        logger.info("="*70)
        logger.info(f"Optimizer: {optimizer_cls.__name__}")
        logger.info(f"Relax cell: {relax_cell}")
        logger.info(f"Convergence: fmax <= {fmax} eV/Angstrom")
        logger.info(f"Maximum steps: {steps}")
        logger.info(f"Initial potential energy: {initial_energy:.6f} eV")

        dyn = optimizer_cls(
            relaxation_target,
            trajectory=None if trajectory_path is None else str(trajectory_path),
            logfile=None if logfile_path is None else str(logfile_path),
        )
        dyn.run(fmax=fmax, steps=steps)

        final_energy = atoms.get_potential_energy()
        logger.info(f"Final potential energy: {final_energy:.6f} eV")
        logger.info(f"Energy change: {final_energy - initial_energy:.6f} eV")
        if relax_cell:
            cell_delta = np.linalg.norm(atoms.cell.array - initial_cell)
            logger.info(f"Cell change Frobenius norm: {cell_delta:.6f} Angstrom")

        if final_file is not None:
            final_path = output_path / final_file
            write(str(final_path), atoms)
            logger.info(f"Saved relaxed structure: {final_path}")

        return atoms

    def setup(
        self,
        atoms: Atoms,
        temperature: float = 300,
        timestep: float = 1.0,
        ensemble: str = 'nvt',
        pressure: float = 1.01325,
        friction: float = 0.02,
        production_ensemble: Optional[str] = None,
        production_friction: Optional[float] = None,
        production_relaxation_steps: int = 0,
        temp_threshold: float = 5.0,
        temp_tolerance: float = 20.0,
        energy_threshold: float = 0.01,
        min_equilibration_steps: int = 100,
        max_equilibration_steps: int = 20000,
        production_steps: int = 10000,
        check_interval: int = 100,
        save_interval: int = 10,
        output_dir: Optional[Path] = None,
        save_xyz: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Set up MD simulation.

        Args:
            atoms: ASE Atoms object
            temperature: Target temperature (K)
            timestep: MD timestep (fs)
            ensemble: 'nvt', 'npt', or 'nve' (for equilibration)
            pressure: Target pressure for NPT (bar)
            friction: Friction coefficient for Langevin thermostat during equilibration (fs^-1)
            production_ensemble: Ensemble for production run (default: same as equilibration)
                                 Use 'nve' for noise-free dynamics suitable for TACAW analysis
            production_friction: Friction for production run (default: same as equilibration)
                                 Use lower value (e.g., 0.01) for cleaner dynamics
            production_relaxation_steps: Number of steps to run after switching ensemble but before
                                        recording trajectory. Allows thermostat artifacts to decay.
                                        Recommended: 100 steps for NVE after high-friction NVT
            temp_threshold: Max std dev of temperature for equilibration (K)
            temp_tolerance: Max deviation of mean temperature from target for equilibration (K)
            energy_threshold: Max relative std dev of energy for equilibration
            min_equilibration_steps: Minimum equilibration steps before checking convergence
            max_equilibration_steps: Maximum equilibration steps before forcing production
            production_steps: Number of production MD steps
            check_interval: Steps between convergence/progress checks
            save_interval: Save trajectory every N steps
            output_dir: Directory for output files
            save_xyz: If True, also save trajectories in XYZ format for OVITO compatibility
            rng: Optional numpy random generator (avoids FAIRChem global seed issue)
        """
        self.atoms = atoms
        self.temperature = temperature
        self.timestep = timestep
        self.ensemble = ensemble
        self.pressure = pressure
        self.friction = friction
        self.production_ensemble = production_ensemble if production_ensemble is not None else ensemble
        self.production_friction = production_friction if production_friction is not None else friction
        self.production_relaxation_steps = production_relaxation_steps
        self.temp_threshold = temp_threshold
        self.temp_tolerance = temp_tolerance
        self.energy_threshold = energy_threshold
        self.min_equilibration_steps = min_equilibration_steps
        self.max_equilibration_steps = max_equilibration_steps
        self.production_steps = production_steps
        self.check_interval = check_interval
        self.save_interval = save_interval
        self.output_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        self.save_xyz = save_xyz

        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

        self.atoms.calc = self._ensure_calculator()

        # Initialize velocities
        logger.info(f"Initializing velocities for T = {temperature} K")
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature, rng=self.rng)

        # Remove center of mass motion
        self.atoms.arrays['momenta'] -= self.atoms.arrays['momenta'].mean(axis=0)

        # Set up MD integrator
        logger.info(f"Setting up {ensemble.upper()} ensemble")

        if ensemble.lower() == 'nvt':
            self.dyn = Langevin(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                friction=friction / units.fs,
                rng=self.rng,
            )
        elif ensemble.lower() == 'npt':
            self.dyn = NPT(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                externalstress=pressure,
                ttime=25*units.fs,
                pfactor=75*units.fs**2,
            )
        elif ensemble.lower() == 'nve':
            from ase.md.verlet import VelocityVerlet
            self.dyn = VelocityVerlet(
                self.atoms,
                timestep * units.fs,
            )
        else:
            logger.warning(f"Unknown ensemble {ensemble}, using NVT")
            self.dyn = Langevin(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                friction=friction / units.fs,
                rng=self.rng,
            )

    def run_equilibration(
        self,
        temperature_window: int = 100,
        energy_window: int = 100,
    ) -> bool:
        """
        Run equilibration phase with convergence checking.

        Uses convergence criteria and output directory set in setup().

        Args:
            temperature_window: Number of steps to check for temp convergence
            energy_window: Number of steps to check for energy convergence

        Returns:
            True if equilibrated, False if max steps reached
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.output_dir / 'equilibration.log'
        traj_file = self.output_dir / 'equilibration.traj'

        logger.info("="*70)
        logger.info("EQUILIBRATION PHASE")
        logger.info("="*70)
        logger.info(f"Convergence criteria:")
        logger.info(f"  Temperature stability: std_T < {self.temp_threshold} K over {temperature_window} steps")
        logger.info(f"  Energy stability: std_E/|<E>| < {self.energy_threshold} over {energy_window} steps")
        logger.info(f"  Minimum steps: {self.min_equilibration_steps}")
        logger.info(f"  Maximum steps: {self.max_equilibration_steps}")

        # Set up convergence checker
        convergence_checker = MDConvergenceChecker(
            target_temperature=self.temperature,
            temperature_window=temperature_window,
            energy_window=energy_window,
            temp_threshold=self.temp_threshold,
            temp_tolerance=self.temp_tolerance,
            energy_threshold=self.energy_threshold,
            min_steps=self.min_equilibration_steps
        )

        logger.info(f"  Temperature tolerance: |<T> - {self.temperature}K| < {self.temp_tolerance}K")

        # Open log and trajectory files
        log = open(log_file, 'w')
        log.write(f"# MD Equilibration\n")
        log.write(f"# System: {len(self.atoms)} atoms\n")
        log.write(f"# Ensemble: {self.ensemble.upper()}\n")
        log.write(f"# Temperature: {self.temperature} K\n")
        log.write(f"# Timestep: {self.timestep} fs\n")
        log.write(f"# Model: {self.model_name}\n")
        log.write(f"# Step Time(ps) Temp(K) Epot(eV) Ekin(eV) Etot(eV)\n")

        eq_traj = ASETrajectory(str(traj_file), 'w', self.atoms)

        # Run equilibration
        logger.info(f"\nRunning equilibration...")
        start_time = time.time()

        equilibrated = False
        for step in range(self.max_equilibration_steps):
            self.dyn.run(1)

            time_ps = self.dyn.nsteps * self.timestep / 1000
            temp = self.atoms.get_temperature()
            epot = self.atoms.get_potential_energy()
            ekin = self.atoms.get_kinetic_energy()
            etot = epot + ekin

            # Log data
            log.write(f"{self.dyn.nsteps} {time_ps:.3f} {temp:.2f} {epot:.6f} {ekin:.6f} {etot:.6f}\n")
            log.flush()

            # Update convergence checker
            convergence_checker.update(self.atoms)

            # Check convergence periodically
            if (step + 1) % self.check_interval == 0:
                converged, status = convergence_checker.check_convergence()
                logger.info(f"Step {step + 1}: T={temp:.1f}K, E={epot/len(self.atoms):.4f}eV/atom")
                logger.info(f"  {status}")

                if converged:
                    logger.info(f"\n[OK] System equilibrated after {step + 1} steps")
                    equilibrated = True
                    break

            # Save to trajectory
            if (step + 1) % self.save_interval == 0:
                eq_traj.write(self.atoms)

        eq_traj.close()
        log.close()

        # Save XYZ for OVITO compatibility
        if self.save_xyz:
            xyz_file = self.output_dir / 'equilibration.xyz'
            from ase.io import read as aseread
            eq_frames = aseread(str(traj_file), index=':')
            write(str(xyz_file), eq_frames, format='extxyz')
            logger.info(f"Saved OVITO-compatible XYZ: {xyz_file}")

        eq_time = time.time() - start_time
        eq_steps = self.dyn.nsteps

        if not equilibrated:
            logger.warning(f"\n[WARN] Maximum steps ({self.max_equilibration_steps}) reached without full convergence")
            logger.info(f"Proceeding to production anyway...")

        # Get statistics
        eq_stats = convergence_checker.get_statistics()
        logger.info(f"\nEquilibration Statistics:")
        logger.info(f"  Steps: {eq_steps}")
        logger.info(f"  Time: {eq_time:.1f}s ({eq_time/eq_steps*1000:.2f}ms/step)")
        logger.info(f"  <T>: {eq_stats['temperature_mean']:.2f} +/- {eq_stats['temperature_std']:.2f} K (target: {self.temperature} K)")
        logger.info(f"  <E>: {eq_stats['energy_mean']:.4f} +/- {eq_stats['energy_std']:.4f} eV/atom")

        # Save equilibrated structure for future runs
        eq_structure_file = self.output_dir / 'equilibrated_structure.xyz'
        write(str(eq_structure_file), self.atoms)
        logger.info(f"Saved equilibrated structure: {eq_structure_file}")

        return equilibrated

    def run_production(
        self,
    ) -> Trajectory:
        """
        Run production MD phase and return PySlice Trajectory.

        Uses production_steps, check_interval, save_interval, and output_dir set in setup().

        Returns:
            PySlice Trajectory object containing the production MD data
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.output_dir / 'production.log'
        traj_file = self.output_dir / 'production.traj'

        logger.info("="*70)
        logger.info("PRODUCTION PHASE")
        logger.info("="*70)

        # Switch to production ensemble if different from equilibration
        if self.production_ensemble != self.ensemble or self.production_friction != self.friction:
            logger.info(f"Switching from {self.ensemble.upper()} (friction={self.friction}) to "
                       f"{self.production_ensemble.upper()} (friction={self.production_friction}) for production")

            if self.production_ensemble.lower() == 'nvt':
                self.dyn = Langevin(
                    self.atoms,
                    self.timestep * units.fs,
                    temperature_K=self.temperature,
                    friction=self.production_friction / units.fs,
                    rng=self.rng,
                )
            elif self.production_ensemble.lower() == 'npt':
                self.dyn = NPT(
                    self.atoms,
                    self.timestep * units.fs,
                    temperature_K=self.temperature,
                    externalstress=self.pressure,
                    ttime=25*units.fs,
                    pfactor=75*units.fs**2,
                )
            elif self.production_ensemble.lower() == 'nve':
                from ase.md.verlet import VelocityVerlet
                self.dyn = VelocityVerlet(
                    self.atoms,
                    self.timestep * units.fs,
                )
                logger.info("  NVE ensemble: microcanonical dynamics (no thermostat noise)")
            else:
                logger.warning(f"Unknown production ensemble {self.production_ensemble}, keeping current")

        # Run relaxation period to let thermostat artifacts decay
        if self.production_relaxation_steps > 0:
            logger.info(f"\nRunning {self.production_relaxation_steps} relaxation steps to remove thermostat artifacts...")
            for step in range(self.production_relaxation_steps):
                self.dyn.run(1)
                if (step + 1) % 50 == 0:
                    temp = self.atoms.get_temperature()
                    logger.info(f"  Relaxation step {step + 1}/{self.production_relaxation_steps}: T={temp:.1f}K")
            logger.info(f"Relaxation complete. Starting recorded production run.\n")

        # Open log and trajectory
        log = open(log_file, 'w')
        log.write(f"# MD Production\n")
        log.write(f"# System: {len(self.atoms)} atoms\n")
        log.write(f"# Ensemble: {self.production_ensemble.upper()}\n")
        log.write(f"# Temperature: {self.temperature} K\n")
        log.write(f"# Timestep: {self.timestep} fs\n")
        log.write(f"# Model: {self.model_name}\n")
        if self.production_ensemble.lower() in ['nvt', 'langevin']:
            log.write(f"# Friction: {self.production_friction} fs^-1\n")
        log.write(f"# Step Time(ps) Temp(K) Epot(eV) Ekin(eV) Etot(eV)\n")

        prod_traj = ASETrajectory(str(traj_file), 'w', self.atoms)

        production_temps = []
        production_energies = []

        # Run production
        logger.info(f"Running production MD ({self.production_steps} steps)...")
        start_time = time.time()

        for step in range(self.production_steps):
            self.dyn.run(1)

            time_ps = self.dyn.nsteps * self.timestep / 1000
            temp = self.atoms.get_temperature()
            epot = self.atoms.get_potential_energy()
            ekin = self.atoms.get_kinetic_energy()
            etot = epot + ekin

            production_temps.append(temp)
            production_energies.append(epot / len(self.atoms))

            # Log data
            log.write(f"{self.dyn.nsteps} {time_ps:.3f} "
                     f"{temp:.2f} {epot:.6f} {ekin:.6f} {etot:.6f}\n")
            log.flush()

            # Print progress
            if (step + 1) % self.check_interval == 0:
                avg_temp = np.mean(production_temps[-self.check_interval:])
                avg_energy = np.mean(production_energies[-self.check_interval:])
                logger.info(f"Production step {step + 1}/{self.production_steps}: "
                          f"T={temp:.1f}K (<T>={avg_temp:.1f}K), "
                          f"E={epot/len(self.atoms):.4f}eV/atom (<E>={avg_energy:.4f})")

            # Save to trajectory
            if (step + 1) % self.save_interval == 0:
                prod_traj.write(self.atoms)

        prod_traj.close()
        log.close()

        prod_time = time.time() - start_time

        # Final statistics
        logger.info("="*70)
        logger.info("PRODUCTION RUN COMPLETE")
        logger.info("="*70)
        logger.info(f"Production Statistics:")
        logger.info(f"  Steps: {self.production_steps}")
        logger.info(f"  Time: {prod_time:.1f}s ({prod_time/self.production_steps*1000:.2f}ms/step)")
        logger.info(f"  <T>: {np.mean(production_temps):.2f} +/- {np.std(production_temps):.2f} K")
        logger.info(f"  <E>: {np.mean(production_energies):.4f} +/- {np.std(production_energies):.4f} eV/atom")

        logger.info(f"Output files:")
        logger.info(f"  Production trajectory: {traj_file}")
        logger.info(f"  Log file: {log_file}")

        # Save XYZ for OVITO compatibility
        if self.save_xyz:
            xyz_file = self.output_dir / 'production.xyz'
            from ase.io import read as aseread
            prod_frames = aseread(str(traj_file), index=':')
            write(str(xyz_file), prod_frames, format='extxyz')
            logger.info(f"  OVITO-compatible XYZ: {xyz_file}")

        # Save final structure
        final_xyz = self.output_dir / 'final_structure.xyz'
        write(str(final_xyz), self.atoms)
        logger.info(f"  Final structure: {final_xyz}")

        # Convert to PySlice Trajectory automatically
        # Use ASE directly to preserve velocities (OVITO strips them)
        # Account for save_interval: actual time between frames = timestep * save_interval
        timestep_ps = self.timestep * self.save_interval / 1000.0  # Convert fs to ps
        logger.info(f"Converting trajectory to PySlice format...")

        from ase.io import read as aseread
        ase_trajectory = aseread(str(traj_file), index=':')

        loader = Loader(atoms=ase_trajectory, timestep=timestep_ps)
        trajectory = loader.load()

        logger.info(f"Converted trajectory: {trajectory.n_frames} frames, "
                   f"{trajectory.n_atoms} atoms")

        return trajectory

    def run(self) -> Trajectory:
        """
        Run full MD simulation (equilibration + production) and return PySlice Trajectory.

        This is the main entry point that combines equilibration and production phases.
        Uses all parameters set in setup().

        Returns:
            PySlice Trajectory object containing the production MD data
        """
        # Run equilibration with convergence criteria
        self.run_equilibration()

        # Run production and return trajectory
        return self.run_production()


class ORBMDCalculator(MDCalculator):
    """
    Molecular dynamics calculator using ORB force fields.

    Integrates with PySlice's Trajectory and Loader infrastructure.
    """

    def __init__(self, model_name: str = 'orb-v3-direct-inf-omat', device: str = 'cpu', precision: str = 'float32-high', weights_path: Optional[str] = None):
        """
        Initialize ORB MD calculator.

        Args:
            model_name: ORB model to use
            device: Device for ORB calculations ('cpu', 'cuda', etc.)
            precision: Precision for ORB calculations ('float32-high', 'float32-highest', etc.)
            weights_path: Optional local path to model weights (bypasses network download)
        """
        super().__init__(model_name=model_name, device=device)
        self.precision = precision
        self.weights_path = weights_path

        # Available ORB models
        self.model_functions = {
            'orb-v3-direct-omol': 'orb_v3_direct_omol',
            'orb-v3-conservative-omol': 'orb_v3_conservative_omol',
            'orb-v3-direct-inf-omat': 'orb_v3_direct_inf_omat',
            'orb-v3-conservative-inf-omat': 'orb_v3_conservative_inf_omat',
            'orb-v3-direct-20-omat': 'orb_v3_direct_20_omat',
            'orb-v3-conservative-20-omat': 'orb_v3_conservative_20_omat',
        }

        logger.info(f"Initialized ORBMDCalculator with model: {model_name}")

    def _setup_calculator(self) -> bool:
        """Load ORB calculator."""
        try:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.inference.calculator import ORBCalculator

            logger.info(f"Loading {self.model_name} model...")

            model_func_name = self.model_functions.get(self.model_name)
            if model_func_name is None:
                logger.error(f"Unknown model: {self.model_name}")
                return False

            model_func = getattr(pretrained, model_func_name)

            # Build kwargs for model loading
            model_kwargs = {'precision': self.precision, 'compile': False}
            if self.weights_path:
                logger.info(f"Using local weights: {self.weights_path}")
                model_kwargs['weights_path'] = self.weights_path

            # ORB doesn't natively support MPS, so we load on CPU then move
            if self.device == 'mps':
                logger.info("MPS detected: loading on CPU, converting to float32, moving to MPS...")
                # compile=False required for MPS (dynamo has float64 issues)
                orbff, atoms_adapter = model_func(device='cpu', **model_kwargs)
                orbff = orbff.float()  # Ensure float32 (MPS doesn't support float64)
                orbff = orbff.to('mps')
            else:
                # compile=False to avoid slow torch.compile on first run
                orbff, atoms_adapter = model_func(device=self.device, **model_kwargs)

            self.calculator = ORBCalculator(orbff, atoms_adapter=atoms_adapter, device=self.device)

            logger.info(f"Successfully loaded {self.model_name} on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load ORB: {e}")
            print(traceback.format_exc())
            return False


def analyze_md_trajectory(
    trajectory_file: str = 'production.traj',
    log_file: str = 'production.log',
    skip_frames: int = 1,
    output_file: str = 'md_analysis.png'
):
    """
    Analyze MD trajectory and create plots.

    Args:
        trajectory_file: Trajectory to analyze
        log_file: MD log file with thermodynamic data
        skip_frames: Analyze every Nth frame
        output_file: Output plot filename
    """
    logger.info("Analyzing MD trajectory...")

    try:
        # Read log file
        data = np.loadtxt(log_file, comments='#')
        steps = data[:, 0]
        time_ps = data[:, 1]
        temps = data[:, 2]
        epots = data[:, 3]
        ekins = data[:, 4]
        etots = data[:, 5]

        logger.info(f"Loaded {len(steps)} timesteps")
        logger.info(f"Simulation time: {time_ps[-1]:.2f} ps")
        logger.info(f"<T>: {np.mean(temps):.2f} +/- {np.std(temps):.2f} K")
        logger.info(f"<Epot>: {np.mean(epots):.2f} +/- {np.std(epots):.2f} eV")
        logger.info(f"<Etot>: {np.mean(etots):.2f} +/- {np.std(etots):.2f} eV")

        # Calculate structural properties
        traj = ASETrajectory(trajectory_file, 'r')

        rmsds = []
        ref_pos = traj[0].get_positions()

        for i, atoms in enumerate(traj[::skip_frames]):
            pos = atoms.get_positions()
            rmsd = np.sqrt(np.mean((pos - ref_pos)**2))
            rmsds.append(rmsd)

        # Plotting
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Temperature
            axes[0, 0].plot(time_ps, temps, alpha=0.5)
            axes[0, 0].axhline(y=np.mean(temps), color='r', linestyle='--',
                              label=f'Mean: {np.mean(temps):.1f} K')
            axes[0, 0].set_xlabel('Time (ps)')
            axes[0, 0].set_ylabel('Temperature (K)')
            axes[0, 0].set_title('Temperature Evolution')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)

            # Energy
            axes[0, 1].plot(time_ps, epots, label='Potential', alpha=0.7)
            axes[0, 1].plot(time_ps, ekins, label='Kinetic', alpha=0.7)
            axes[0, 1].plot(time_ps, etots, label='Total', alpha=0.7)
            axes[0, 1].set_xlabel('Time (ps)')
            axes[0, 1].set_ylabel('Energy (eV)')
            axes[0, 1].set_title('Energy Evolution')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)

            # RMSD from initial
            axes[1, 0].plot(rmsds)
            axes[1, 0].set_xlabel('Frame')
            axes[1, 0].set_ylabel('RMSD (A)')
            axes[1, 0].set_title('RMSD from Initial Structure')
            axes[1, 0].grid(alpha=0.3)

            # Energy distribution
            axes[1, 1].hist(epots, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=np.mean(epots), color='r', linestyle='--',
                              label=f'Mean: {np.mean(epots):.2f} eV')
            axes[1, 1].set_xlabel('Potential Energy (eV)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Energy Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_file, dpi=300)
            logger.info(f"Saved analysis plot: {output_file}")

        except ImportError:
            logger.warning("Matplotlib not available for plotting")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")


class FAIRChemMDCalculator(MDCalculator):
    """
    Molecular dynamics calculator using FAIRChem force fields.

    Integrates with PySlice's Trajectory and Loader infrastructure.
    """

    def __init__(self,
                 model_name: str = 'uma-s-1p1',
                 task_name: str = 'omat',
                 device: str = 'cpu',
                 workers: int = 1,
                 inference_settings: 'InferenceSettings | str' = 'default'):
        """
        Initialize FAIRChem MD calculator.

        Args:
            model_name: FAIRChem model to use
            task_name: Task name for FAIRChem model
            device: Device for FAIRChem calculations ('cpu', 'cuda', etc.)
            workers: Number of workers for FAIRChem calculations
            inference_settings: InferenceSettings object or 'default'
        """
        # Lazy import of InferenceSettings - only when FAIRChemMDCalculator is instantiated
        try:
            from fairchem.core.units.mlip_unit import InferenceSettings
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "FAIRChem is required to use FAIRChemMDCalculator. "
                "Install it with: pip install fairchem-core"
            )

        super().__init__(model_name=model_name, device=device)
        self.task_name = task_name
        self.workers = workers
        self.inference_settings = inference_settings

        logger.info(f"Initialized FAIRChemMDCalculator with model: {model_name}")

    def _setup_calculator(self) -> bool:
        """Load FAIRChem calculator."""
        try:
            logger.info(f"Loading {self.model_name} model...")

            from fairchem.core import pretrained_mlip, FAIRChemCalculator

            # Not sure if this works also for fairchem, but in case MPS support is similar to ORB, we load on CPU then move
            if self.device == 'mps':
                logger.info("MPS detected: loading on CPU, converting to float32, moving to MPS...")
                predictor = pretrained_mlip.get_predict_unit(
                    self.model_name, device='cpu', workers=self.workers,
                    inference_settings=self.inference_settings,
                )
                predictor = predictor.to('mps')
            else:
                predictor = pretrained_mlip.get_predict_unit(
                    self.model_name, device=self.device, workers=self.workers,
                    inference_settings=self.inference_settings,
                )

            self.calculator = FAIRChemCalculator(predictor, task_name=self.task_name)

            logger.info(f"Successfully loaded {self.model_name} on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load FAIRChem model {self.model_name}: {e}")
            print(traceback.format_exc())
            return False
