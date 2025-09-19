"""
Trajectory loading module for LAMMPS dump files.
"""
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Optional, Dict, Union

from ..multislice.trajectory import Trajectory
from ..multislice.potentials import getZfromElementName

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
    def __init__(self,
                 filename: str,
                 timestep: Optional[float] = None,
                 atom_mapping: Optional[Dict[int, Union[int, str]]] = None,
                 # Keep old parameters for backward compatibility but deprecated
                 atomic_numbers: Optional[Dict[int, int]] = None,
                 element_names: Optional[Dict[int, str]] = None):
        """
        Initialize trajectory loader for various trajectory file formats.

        Args:
            filename: Path to trajectory/structure file
            timestep: Timestep in picoseconds. Defaults to 1.0 ps.
            atom_mapping: Dictionary mapping atom types to either:
                - Atomic numbers (int): {1: 6, 2: 8} for carbon and oxygen
                - Element names (str): {1: "C", 2: "O"} for carbon and oxygen
            atomic_numbers: (Deprecated) Use atom_mapping instead
            element_names: (Deprecated) Use atom_mapping instead
        """
        if timestep is not None and timestep <= 0:
            raise ValueError("timestep must be positive if specified.")

        self.filepath = Path(filename)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Trajectory file not found: {filename}")

        self.timestep = timestep if timestep is not None else 1.0

        # Handle backward compatibility
        if atomic_numbers is not None:
            logger.warning("atomic_numbers parameter is deprecated. Use atom_mapping instead.")
            atom_mapping = atomic_numbers
        elif element_names is not None:
            logger.warning("element_names parameter is deprecated. Use atom_mapping instead.")
            atom_mapping = element_names

        # Process atom mapping
        self.atomic_numbers = self._process_atom_mapping(atom_mapping)

    def _process_atom_mapping(self, mapping: Optional[Dict[int, Union[int, str]]]) -> Optional[Dict[int, int]]:
        """Convert atom mapping to atomic numbers."""
        if mapping is None:
            return None

        result = {}
        for atom_type, value in mapping.items():
            if isinstance(value, str):
                # Element name - convert to atomic number
                result[atom_type] = getZfromElementName(value)
            elif isinstance(value, int):
                # Already an atomic number
                if not (1 <= value <= 118):
                    raise ValueError(f"Invalid atomic number {value} for type {atom_type}. Must be between 1 and 118.")
                result[atom_type] = value
            else:
                raise ValueError(f"Invalid mapping value {value} for type {atom_type}. Must be int (atomic number) or str (element name).")

        return result

    def _apply_atomic_mapping(self, atom_types: np.ndarray) -> np.ndarray:
        """Apply atomic number mapping to atom types."""
        if self.atomic_numbers is None:
            return atom_types

        mapped_types = atom_types.copy()
        unique_types = np.unique(atom_types)
        unmapped_types = []

        for atom_type in unique_types:
            if atom_type in self.atomic_numbers:
                mapped_types[atom_types == atom_type] = self.atomic_numbers[atom_type]
            else:
                unmapped_types.append(atom_type)

        if unmapped_types:
            logger.warning(f"No mapping provided for atom types {unmapped_types}.")

        return mapped_types

    def _get_cache_files(self) -> Dict[str, Path]:
        """Get paths for cache files."""
        cache_stem = self.filepath.parent / self.filepath.stem
        return {
            'positions': cache_stem.with_suffix('.positions.npy'),
            'velocities': cache_stem.with_suffix('.velocities.npy'),
            'atom_types': cache_stem.with_suffix('.atom_types.npy'),
            'box_matrix': cache_stem.with_suffix('.box_matrix.npy')
        }

    def _load_from_cache(self) -> Optional[Trajectory]:
        """Try to load trajectory from cached .npy files."""
        cache_files = self._get_cache_files()

        if not all(f.exists() for f in cache_files.values()):
            return None

        try:
            logger.info(f"Loading from cache for {self.filepath.name}")

            pos = np.load(cache_files['positions'])
            vel = np.load(cache_files['velocities'])
            atom_types = np.load(cache_files['atom_types'])
            box_mat = np.load(cache_files['box_matrix'])

            if box_mat.shape != (3, 3):
                raise ValueError(f"Invalid box_matrix shape: {box_mat.shape}")

            trajectory = Trajectory(
                atom_types=atom_types,
                positions=pos,
                velocities=vel,
                box_matrix=box_mat,
                timestep=self.timestep
            )

            logger.info(f"Loaded: {trajectory.n_frames} frames, {trajectory.n_atoms} atoms")
            return trajectory

        except Exception as e:
            logger.warning(f"Cache loading failed: {e}")
            return None

    def _save_to_cache(self, trajectory: Trajectory) -> None:
        """Save trajectory to cache files."""
        cache_files = self._get_cache_files()

        logger.info(f"Saving to cache for {self.filepath.name}")
        cache_files['positions'].parent.mkdir(parents=True, exist_ok=True)

        np.save(cache_files['positions'], trajectory.positions)
        np.save(cache_files['velocities'], trajectory.velocities)
        np.save(cache_files['atom_types'], trajectory.atom_types)
        np.save(cache_files['box_matrix'], trajectory.box_matrix)

    def load(self) -> Trajectory:
        """Load trajectory from file."""
        # Try cache first
        trajectory = self._load_from_cache()
        if trajectory is not None:
            return trajectory

        # Load via OVITO or ASE
        if self.filepath.suffix in [".cif"]:
            logger.info(f"Loading {self.filepath.name} via ASE")
            trajectory = self._load_via_ase()
        else:
            logger.info(f"Loading {self.filepath.name} via OVITO")
            trajectory = self._load_via_ovito()

        # Save to cache
        self._save_to_cache(trajectory)

        return trajectory

    def _validate_frame_data(self, frame_data, frame_num: int = 0) -> None:
        """Validate OVITO frame data."""
        if not frame_data:
            raise ValueError(f"No data for frame {frame_num}")

        if not (hasattr(frame_data, 'cell') and frame_data.cell):
            raise ValueError(f"No cell data in frame {frame_num}")

        if not (hasattr(frame_data, 'particles') and frame_data.particles):
            raise ValueError(f"No particle data in frame {frame_num}")

        if not (hasattr(frame_data.particles, 'positions') and
                frame_data.particles.positions is not None and
                len(frame_data.particles.positions) > 0):
            raise ValueError(f"No position data in frame {frame_num}")

    def _load_via_ovito(self) -> Trajectory:
        """Load trajectory via OVITO."""
        if not OVITO_AVAILABLE:
            raise ImportError("OVITO is not available. Please install ovito package.")

        # Import file
        try:
            pipeline = import_file(str(self.filepath))
            # Add unwrap modifier for periodic boundary conditions
            if hasattr(pipeline.source, 'data') and pipeline.source.data:
                pipeline.modifiers.append(UnwrapTrajectoriesModifier())
        except Exception as e:
            raise RuntimeError(f"OVITO import failed: {e}")

        n_frames = pipeline.source.num_frames
        if n_frames == 0:
            raise ValueError("No frames found in trajectory")

        # Get frame 0 data for setup
        try:
            frame0_data = pipeline.compute(0)
        except Exception as e:
            raise RuntimeError(f"Failed to compute frame 0: {e}")

        self._validate_frame_data(frame0_data, 0)

        # Extract basic info
        n_atoms = len(frame0_data.particles.positions)
        h_matrix = np.array(frame0_data.cell.matrix, dtype=np.float32)[:3, :3]

        has_velocities = (hasattr(frame0_data.particles, 'velocities') and
                         frame0_data.particles.velocities is not None)

        if not has_velocities:
            logger.warning("No velocity data found. Setting velocities to zero.")

        # Allocate arrays
        positions = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        velocities = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)

        # Load frames
        for i in tqdm(range(n_frames), desc="Loading frames", unit="frame"):
            try:
                frame_data = pipeline.compute(i)

                if frame_data and hasattr(frame_data, 'particles'):
                    if hasattr(frame_data.particles, 'positions') and frame_data.particles.positions is not None:
                        positions[i] = np.array(frame_data.particles.positions, dtype=np.float32)

                    if has_velocities and hasattr(frame_data.particles, 'velocities') and frame_data.particles.velocities is not None:
                        velocities[i] = np.array(frame_data.particles.velocities, dtype=np.float32)

            except Exception as e:
                logger.error(f"Failed to load frame {i}: {e}")
                continue

        # Get atom types
        if (hasattr(frame0_data.particles, 'particle_types') and
            frame0_data.particles.particle_types is not None and
            len(frame0_data.particles.particle_types) == n_atoms):
            atom_types = np.array(frame0_data.particles.particle_types, dtype=np.int32)
        else:
            logger.warning("No particle type data found. Setting all types to 1.")
            atom_types = np.ones(n_atoms, dtype=np.int32)

        # Apply atomic mapping
        atom_types = self._apply_atomic_mapping(atom_types)

        logger.info(f"Loaded {n_frames} frames with {n_atoms} atoms")

        return Trajectory(
            atom_types=atom_types,
            positions=positions,
            velocities=velocities,
            box_matrix=h_matrix,
            timestep=self.timestep
        )

    def _load_via_ase(self) -> Trajectory:
        from ase.io import read as aseread
        atoms = aseread(str(self.filepath))
        return self.ase2Trajectory(atoms)

    def ase2Trajectory(self,atoms):
        pos=np.asarray([atoms.get_positions()])
        vel=np.asarray([atoms.get_velocities()])
        return Trajectory(
            atom_types=np.asarray( atoms.get_chemical_symbols() ),
            positions=pos,
            velocities=vel,
            box_matrix=atoms.get_cell(),
            timestep=self.timestep
        )
