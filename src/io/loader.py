"""
Trajectory loading module for LAMMPS dump files.
"""
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Optional


from ..tacaw.trajectory import Trajectory

# Try to import OVITO, but don't fail if it's not available
try:
    from ovito.io import import_file
    from ovito.modifiers import UnwrapTrajectoriesModifier
    OVITO_AVAILABLE = True
except ImportError as e:
    logging.error(f"OVITO import failed: {e}")
    OVITO_AVAILABLE = False

logger = logging.getLogger(__name__)

class TrajectoryLoader:
    def __init__(self, filename: str, timestep: float = 1.0, atomic_numbers: Optional[dict[int, int]] = None):
        """
        Initialize trajectory loader for LAMMPS dump files.

        Args:
            filename: Path to LAMMPS dump file
            timestep: Timestep in picoseconds (LAMMPS files only contain frame numbers,
                not timestep values, so this must be provided)
            atomic_numbers: Optional dictionary mapping LAMMPS atom types to actual atomic numbers.
                If provided, LAMMPS atom types will be converted to actual atomic numbers.
                Example: {1: 6, 2: 8} maps LAMMPS type 1 to carbon (6) and type 2 to oxygen (8)
        """
        if timestep <= 0:
            raise ValueError("timestep must be positive.")
        self.filepath = Path(filename)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Trajectory file not found: {filename}")
        self.timestep = timestep
        self.atomic_numbers = atomic_numbers

    def _apply_atomic_number_mapping(self, atom_types: np.ndarray) -> np.ndarray:
        """
        Apply atomic number mapping to convert LAMMPS atom types to actual atomic numbers.

        Args:
            atom_types: Array of LAMMPS atom types

        Returns:
            Array of atomic numbers (same shape as input)
        """
        if self.atomic_numbers is None:
            return atom_types

        # Validate that atomic numbers are reasonable (between 1 and 118)
        for lammps_type, atomic_num in self.atomic_numbers.items():
            if not (1 <= atomic_num <= 118):
                raise ValueError(f"Invalid atomic number {atomic_num} for LAMMPS type {lammps_type}. Atomic numbers must be between 1 and 118.")

        # Create a copy to avoid modifying the original
        mapped_types = atom_types.copy()

        # Apply mapping for each unique LAMMPS atom type
        unique_types = np.unique(atom_types)
        unmapped_types = []

        for lammps_type in unique_types:
            if lammps_type in self.atomic_numbers:
                mapped_types[atom_types == lammps_type] = self.atomic_numbers[lammps_type]
            else:
                unmapped_types.append(lammps_type)

        if unmapped_types:
            logger.warning(f"No atomic number mapping provided for LAMMPS atom types {unmapped_types}. These will keep their original LAMMPS type values.")

        return mapped_types

    def load(self) -> Trajectory:
        """Load trajectory from LAMMPS dump file."""
        cache_stem = self.filepath.parent / self.filepath.stem
        npy_files = {
            'positions': cache_stem.with_suffix('.positions.npy'),
            'velocities': cache_stem.with_suffix('.velocities.npy'),
            'atom_types': cache_stem.with_suffix('.atom_types.npy'),
            'box_matrix': cache_stem.with_suffix('.box_matrix.npy')
        }

        # Try to load from cached .npy files first
        if all(f.exists() for f in npy_files.values()):
            logger.info(f"Loading trajectory from cached .npy files for {self.filepath.name}.")
            try:
                pos = np.load(npy_files['positions'])
                vel = np.load(npy_files['velocities'])
                atom_types = np.load(npy_files['atom_types'])
                box_mat = np.load(npy_files['box_matrix'])

                if box_mat.shape != (3,3):
                    raise ValueError(f"Cached box_matrix has shape {box_mat.shape}, expected (3,3).")

                # Note: Cached data already contains mapped atomic numbers if mapping was applied

                trajectory = Trajectory(
                    atom_types=atom_types,
                    positions=pos,
                    velocities=vel,
                    box_matrix=box_mat,
                    timestep=self.timestep
                )

                logger.info(f"Loaded trajectory: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms.")
                return trajectory

            except Exception as e:
                logger.warning(f"Loading .npy cache failed: {e}. Falling back to OVITO.")

        # Load via OVITO if cache doesn't exist or failed
        logger.info(f"No complete .npy cache for {self.filepath.name}; loading via OVITO.")
        trajectory = self._load_via_ovito()

        # Save the trajectory data to .npy files
        self._save_trajectory_npy(trajectory)

        return trajectory

    def _load_via_ovito(self) -> Trajectory:
        """Load trajectory via OVITO from LAMMPS dump format."""
        if not OVITO_AVAILABLE:
            raise ImportError("OVITO is not available. Please install OVITO Python to load trajectory files.")

        logger.info(f"Loading '{self.filepath.name}' with OVITO (LAMMPS dump format).")

        try:
            pipeline = import_file(str(self.filepath), input_format='lammps/dump')
            pipeline.modifiers.append(UnwrapTrajectoriesModifier())
        except Exception as e:
            logger.error(f"OVITO failed to load file '{self.filepath.name}': {e}")
            raise RuntimeError(f"OVITO import failed: {e}")

        n_frames = pipeline.source.num_frames
        if n_frames == 0:
            raise ValueError("OVITO: 0 frames in trajectory.")

        try:
            frame0_data = pipeline.compute(0)
        except Exception as e:
            logger.error(f"OVITO failed to compute frame 0: {e}")
            raise RuntimeError(f"OVITO compute failed: {e}")

        if not (frame0_data and hasattr(frame0_data, 'cell') and frame0_data.cell):
            raise ValueError("OVITO: Could not read cell data from frame 0.")
        if not (hasattr(frame0_data, 'particles') and frame0_data.particles):
            raise ValueError("OVITO: Could not read particle data from frame 0.")

        n_atoms = len(frame0_data.particles.positions) if hasattr(frame0_data.particles, 'positions') and frame0_data.particles.positions is not None else 0
        if n_atoms == 0:
            raise ValueError("OVITO: 0 atoms in frame 0.")

        has_vel = hasattr(frame0_data.particles, 'velocities') and frame0_data.particles.velocities is not None
        if not has_vel:
            logger.warning("OVITO: No velocity data found. Velocities set to zero.")

        pos_all = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        vel_all = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)

        h_matrix = np.array(frame0_data.cell.matrix, dtype=np.float32)[:3,:3]

        for i in tqdm(range(n_frames), desc=f"Loading OVITO frames from {self.filepath.name}", unit="fr"):
            try:
                frame_data = pipeline.compute(i)
            except Exception as e:
                logger.error(f"OVITO failed to compute frame {i}: {e}")
                continue

            if not (frame_data and hasattr(frame_data, 'particles') and frame_data.particles):
                logger.error(f"OVITO: Could not compute frame {i}. Data will be zero.")
                continue

            if hasattr(frame_data.particles, 'positions') and frame_data.particles.positions is not None:
                frame_pos = np.array(frame_data.particles.positions, dtype=np.float32)
                if frame_pos.shape == (n_atoms, 3):
                    pos_all[i] = frame_pos
                else:
                    logger.warning(f"OVITO: Pos shape mismatch frame {i}. Expected ({n_atoms},3), got {frame_pos.shape}.")
            else:
                logger.warning(f"OVITO: No position data frame {i}.")

            if has_vel and hasattr(frame_data.particles, 'velocities') and frame_data.particles.velocities is not None:
                frame_vel = np.array(frame_data.particles.velocities, dtype=np.float32)
                if frame_vel.shape == (n_atoms, 3):
                    vel_all[i] = frame_vel
                else:
                    logger.warning(f"OVITO: Vel shape mismatch frame {i}. Expected ({n_atoms},3), got {frame_vel.shape}.")

        types_data = frame0_data.particles.particle_types if hasattr(frame0_data.particles, 'particle_types') and frame0_data.particles.particle_types is not None else None
        if types_data is not None and len(types_data) == n_atoms:
            atom_types_arr = np.array(types_data, dtype=np.int32)
        else:
            logger.warning(f"OVITO: Particle types missing/mismatched (expected {n_atoms}). Defaulting types to 1.")
            atom_types_arr = np.ones(n_atoms, dtype=np.int32)

        # Apply atomic number mapping if provided
        atom_types_arr = self._apply_atomic_number_mapping(atom_types_arr)

        logger.info(f"Trajectory '{self.filepath.name}' loaded via OVITO: {n_frames} frames, {n_atoms} atoms.")

        return Trajectory(
            atom_types=atom_types_arr,
            positions=pos_all,
            velocities=vel_all,
            box_matrix=h_matrix,
            timestep=self.timestep
        )

    def _save_trajectory_npy(self, traj: Trajectory) -> None:
        """Save trajectory data to .npy files for caching."""
        cache_stem = self.filepath.parent / self.filepath.stem
        npy_files = {
            'positions': cache_stem.with_suffix('.positions.npy'),
            'velocities': cache_stem.with_suffix('.velocities.npy'),
            'atom_types': cache_stem.with_suffix('.atom_types.npy'),
            'box_matrix': cache_stem.with_suffix('.box_matrix.npy')
        }

        if all(f.exists() for f in npy_files.values()):
            logger.info(f".npy cache for {self.filepath.name} exists; skipping save.")
            return

        logger.info(f"Saving trajectory '{self.filepath.name}' to .npy (stem: {cache_stem.name}).")
        cache_stem.parent.mkdir(parents=True, exist_ok=True)

        # Save the core trajectory data
        np.save(npy_files['positions'], traj.positions)
        np.save(npy_files['velocities'], traj.velocities)
        np.save(npy_files['atom_types'], traj.atom_types)
        np.save(npy_files['box_matrix'], traj.box_matrix)

        # Save derived data for convenience
        mean_pos = np.mean(traj.positions, axis=0)
        disp = traj.positions - mean_pos[None, :, :]
        np.save(cache_stem.with_suffix('.mean_positions.npy'), mean_pos)
        np.save(cache_stem.with_suffix('.displacements.npy'), disp)
        logger.info(f"Trajectory data for {self.filepath.name} saved to .npy.")



