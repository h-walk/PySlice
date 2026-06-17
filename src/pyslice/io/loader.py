"""Load structures and trajectories into PySlice ``Trajectory`` objects."""
import numpy as np
from pathlib import Path
import json
import logging
import os
import subprocess
import sys
import tempfile
from tqdm import tqdm
from typing import Optional, Dict, Union

from ..multislice.trajectory import Trajectory
from ..multislice.potentials import getZfromElementName

logger = logging.getLogger(__name__)

class Loader:
    """Load a file or ASE object into the internal ``Trajectory`` representation.

    Supported paths include LAMMPS-style files handled by OVITO and CIF/ASE
    inputs handled through ASE.  Loaded arrays are cached next to the source
    file as ``*.npy`` files so repeated loads avoid parser overhead.
    """

    def __init__(self,
                 filename: Optional[str] = None,
                 timestep: Optional[float] = None,
                 atom_mapping: Optional[Dict[int, Union[int, str]]] = None,
                 # Keep old parameters for backward compatibility but deprecated
                 atomic_numbers: Optional[Dict[int, int]] = None,
                 element_names: Optional[Dict[int, str]] = None,
                 ovitokwargs: Optional[Dict[str,str]] = None,
                 atoms = None ):
        """
        Initialize a trajectory loader.

        Args:
            filename: Path to structure/trajectory file (optional if atoms is provided).
            timestep: Timestep in picoseconds. Defaults to 1.0 ps.
            atom_mapping: Dictionary mapping atom types to either:
                - Atomic numbers (int): {1: 6, 2: 8} for carbon and oxygen
                - Element names (str): {1: "C", 2: "O"} for carbon and oxygen
            atomic_numbers: Deprecated; use atom_mapping instead.
            element_names: Deprecated; use atom_mapping instead.
            ovitokwargs: Additional keyword arguments forwarded to OVITO import.
            atoms: ASE Atoms object or trajectory. If provided, file loading is skipped.
        """
        if timestep is not None and timestep <= 0:
            raise ValueError("timestep must be positive if specified.")

        if filename is None and atoms is None:
            raise ValueError("Either filename or atoms must be provided")

        self.atoms = atoms
        self.filepath = Path(filename) if filename is not None else None

        if self.filepath is not None and not self.filepath.exists():
            raise FileNotFoundError(f"Trajectory file not found: {filename}")

        self.timestep = timestep if timestep is not None else 1.0

        self.ovitokwargs = ovitokwargs if ovitokwargs is not None else {}

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
        """Get paths for cache files.

        Uses full filename (with extension) as base to avoid collisions between
        files with the same stem but different extensions (e.g., data.xyz vs data.lammpstrj).
        """
        # Use full filename to avoid cache collisions (e.g., data.xyz vs data.positions)
        cache_base = self.filepath.parent / self.filepath.name
        return {
            'positions': cache_base.with_suffix(cache_base.suffix + '.positions.npy'),
            'velocities': cache_base.with_suffix(cache_base.suffix + '.velocities.npy'),
            'atom_types': cache_base.with_suffix(cache_base.suffix + '.atom_types.npy'),
            'box_matrix': cache_base.with_suffix(cache_base.suffix + '.box_matrix.npy')
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
        """Load structure/trajectory from file or ASE Atoms object and return as Trajectory."""
        # If atoms object provided, convert directly
        if self.atoms is not None:
            logger.info("Converting ASE Atoms object to Trajectory")
            return self.ase2Trajectory(self.atoms)

        # Try cache first
        trajectory = self._load_from_cache()
        if trajectory is not None:
            return trajectory

        # Load via OVITO or ASE
        # CIF files use ASE because OVITO's CIF parser is limited (e.g., fails on multi-block CIF files)
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
        if sys.platform == "win32" and os.environ.get("PYSLICE_OVITO_IN_PROCESS") != "1":
            return self._load_via_ovito_subprocess()
        return self._load_via_ovito_direct()

    def _load_via_ovito_subprocess(self) -> Trajectory:
        """Load trajectory via OVITO in a subprocess to avoid Windows DLL conflicts."""
        try:
            import_kwargs_json = json.dumps(self.ovitokwargs)
        except TypeError as exc:
            raise TypeError("ovitokwargs must be JSON serializable for Windows OVITO loading.") from exc

        worker_code = r'''
import json
import sys
from pathlib import Path

import numpy as np
from ovito.io import import_file
from ovito.modifiers import UnwrapTrajectoriesModifier


def validate_frame_data(frame_data, frame_num=0):
    """Validate the subset of OVITO frame data required by PySlice."""
    if not frame_data:
        raise ValueError(f"No data for frame {frame_num}")
    if not (hasattr(frame_data, "cell") and frame_data.cell):
        raise ValueError(f"No cell data in frame {frame_num}")
    if not (hasattr(frame_data, "particles") and frame_data.particles):
        raise ValueError(f"No particle data in frame {frame_num}")
    if not (
        hasattr(frame_data.particles, "positions")
        and frame_data.particles.positions is not None
        and len(frame_data.particles.positions) > 0
    ):
        raise ValueError(f"No position data in frame {frame_num}")


filepath = Path(sys.argv[1])
output_path = Path(sys.argv[2])
import_kwargs = json.loads(sys.argv[3])

try:
    if filepath.suffix.lower() == ".xyz":
        try:
            pipeline = import_file(str(filepath), bounding_box=True, **import_kwargs)
        except KeyError as exc:
            if "bounding_box" in str(exc):
                pipeline = import_file(str(filepath), **import_kwargs)
            else:
                raise
    else:
        pipeline = import_file(str(filepath), **import_kwargs)
except Exception as exc:
    raise RuntimeError(f"OVITO import failed: {exc}") from exc

if hasattr(pipeline.source, "data") and pipeline.source.data:
    pipeline.modifiers.append(UnwrapTrajectoriesModifier())

n_frames = pipeline.source.num_frames
if n_frames == 0:
    raise ValueError("No frames found in trajectory")

try:
    frame0_data = pipeline.compute(0)
except RuntimeError as exc:
    if "Unwrap trajectories" in str(exc):
        pipeline.modifiers.clear()
        frame0_data = pipeline.compute(0)
    else:
        raise RuntimeError(f"Failed to compute frame 0: {exc}") from exc

validate_frame_data(frame0_data, 0)

n_atoms = len(frame0_data.particles.positions)
h_matrix = np.array(frame0_data.cell.matrix, dtype=np.float32)[:3, :3]
has_velocities = (
    hasattr(frame0_data.particles, "velocities")
    and frame0_data.particles.velocities is not None
)

positions = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
velocities = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)

for i in range(n_frames):
    try:
        frame_data = pipeline.compute(i)
        if frame_data and hasattr(frame_data, "particles"):
            if (
                hasattr(frame_data.particles, "positions")
                and frame_data.particles.positions is not None
            ):
                positions[i] = np.array(frame_data.particles.positions, dtype=np.float32)
            if (
                has_velocities
                and hasattr(frame_data.particles, "velocities")
                and frame_data.particles.velocities is not None
            ):
                velocities[i] = np.array(frame_data.particles.velocities, dtype=np.float32)
    except Exception as exc:
        print(f"Failed to load frame {i}: {exc}", file=sys.stderr)

if (
    hasattr(frame0_data.particles, "particle_types")
    and frame0_data.particles.particle_types is not None
    and len(frame0_data.particles.particle_types) == n_atoms
):
    atom_types = np.array(frame0_data.particles.particle_types, dtype=np.int32)
else:
    print("No particle type data found. Setting all types to 1.", file=sys.stderr)
    atom_types = np.ones(n_atoms, dtype=np.int32)

np.savez(
    output_path,
    positions=positions,
    velocities=velocities,
    atom_types=atom_types,
    box_matrix=h_matrix,
)
'''

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ovito_trajectory.npz"
            result = subprocess.run(
                [sys.executable, "-c", worker_code, str(self.filepath), str(output_path), import_kwargs_json],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                raise RuntimeError(f"OVITO subprocess failed: {stderr}")
            if result.stderr.strip():
                logger.warning(result.stderr.strip())

            with np.load(output_path) as data:
                positions = data["positions"]
                velocities = data["velocities"]
                atom_types = self._apply_atomic_mapping(data["atom_types"])
                box_matrix = data["box_matrix"]

        logger.info(f"Loaded {positions.shape[0]} frames with {positions.shape[1]} atoms")

        return Trajectory(
            atom_types=atom_types,
            positions=positions,
            velocities=velocities,
            box_matrix=box_matrix,
            timestep=self.timestep
        )

    def _load_via_ovito_direct(self) -> Trajectory:
        """Load trajectory via OVITO in the current Python process."""
        try:
            from ovito.io import import_file
            from ovito.modifiers import UnwrapTrajectoriesModifier
        except ImportError as exc:
            raise ImportError("OVITO is not available. Please install ovito package.") from exc

        # Import file
        try:
            import_kwargs = self.ovitokwargs.copy()
            # OVITO 3.14+ changed bounding_box default to False for XYZ files
            # We need the bounding box to infer cell dimensions
            if self.filepath.suffix.lower() == '.xyz':
                try:
                    pipeline = import_file(str(self.filepath), bounding_box=True, **import_kwargs)
                except KeyError as e:
                    if 'bounding_box' in str(e):
                        # OVITO < 3.14 doesn't have this parameter (old behavior is default)
                        pipeline = import_file(str(self.filepath), **import_kwargs)
                    else:
                        raise
            else:
                pipeline = import_file(str(self.filepath), **import_kwargs)
        except Exception as e:
            raise RuntimeError(f"OVITO import failed: {e}")

        # Try to add unwrap modifier - it will be removed later if it fails
        if hasattr(pipeline.source, 'data') and pipeline.source.data:
            pipeline.modifiers.append(UnwrapTrajectoriesModifier())

        n_frames = pipeline.source.num_frames
        if n_frames == 0:
            raise ValueError("No frames found in trajectory")

        # Get frame 0 data for setup - if unwrap fails, retry without it
        try:
            frame0_data = pipeline.compute(0)
        except RuntimeError as e:
            if "Unwrap trajectories" in str(e):
                logger.info("Unwrap modifier not applicable - proceeding without unwrapping")
                pipeline.modifiers.clear()
                frame0_data = pipeline.compute(0)
            else:
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

    def ase2Trajectory(self, atoms):
        """Convert ASE Atoms or list of Atoms to Trajectory.

        Args:
            atoms: Either a single ASE Atoms object or a list/trajectory of Atoms objects
        """
        # Check if atoms is iterable (trajectory with multiple frames)
        try:
            # Try to iterate and check if it's a multi-frame trajectory
            iter(atoms)
            is_trajectory = True
            # Special case: single Atoms object is technically iterable (over atoms)
            # but we want to treat it as a single frame
            if hasattr(atoms, 'get_positions'):
                is_trajectory = False
        except TypeError:
            is_trajectory = False

        if is_trajectory:
            # Multiple frames
            frames = list(atoms)
            n_frames = len(frames)

            # Get dimensions from first frame
            first_frame = frames[0]
            n_atoms = len(first_frame)

            # Allocate arrays
            positions = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
            velocities = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)

            # Load each frame
            for i, frame in enumerate(frames):
                positions[i] = frame.get_positions()
                if frame.get_velocities() is not None:
                    velocities[i] = frame.get_velocities()

            atom_types = np.asarray(first_frame.get_chemical_symbols())
            box_matrix = np.array(first_frame.get_cell())
        else:
            # Single frame
            positions = np.asarray([atoms.get_positions()])
            velocities_data = atoms.get_velocities()
            if velocities_data is not None:
                velocities = np.asarray([velocities_data])
            else:
                velocities = np.zeros_like(positions)
            atom_types = np.asarray(atoms.get_chemical_symbols())
            box_matrix = np.array(atoms.get_cell())

        return Trajectory(
            atom_types=atom_types,
            positions=positions,
            velocities=velocities,
            box_matrix=box_matrix,
            timestep=self.timestep
        )
