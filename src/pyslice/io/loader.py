"""Load structures and trajectories into PySlice ``Trajectory`` objects."""
import numpy as np
from pathlib import Path
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from tqdm import tqdm
from typing import Optional, Dict, Union

from ..multislice.trajectory import Trajectory
from ..multislice.potentials import get_z_from_element
from ..backend import source_files_version

logger = logging.getLogger(__name__)


def _ovito_cell_to_row_convention(matrix):
    """Convert an OVITO cell matrix to PySlice's row convention plus origin.

    OVITO stores the three lattice vectors in the COLUMNS of a 3x4 matrix, with
    the cell origin in the 4th column. PySlice (and the ASE path) store lattice
    vectors as ROWS, so the 3x3 vector block is transposed. Returns
    ``(box_matrix_rows, origin)``.
    """
    cell_np = np.asarray(matrix, dtype=np.float32)
    h_matrix = cell_np[:3, :3].T
    origin = (cell_np[:3, 3].copy() if cell_np.shape[1] > 3
              else np.zeros(3, dtype=np.float32))
    return h_matrix, origin


def _parse_index(index):
    """Parse a frame selector into an ``int`` or ``slice``.

    Accepts an int, a ``slice``, or an ASE-like string:
    ``":"`` (all), ``":3"`` (first 3), ``"-3:"`` (last 3), ``"::2"`` (every
    other), ``"3:5"`` (Python slice, 3 and 4), ``"3-5"`` (inclusive range, 3 to
    5), or a bare integer like ``"1"`` / ``"-1"``.
    """
    if isinstance(index, (int, np.integer)):
        return int(index)
    if isinstance(index, slice):
        return index
    if not isinstance(index, str):
        raise ValueError(
            f"index must be an int, slice or str; got {type(index).__name__}")
    s = index.strip()
    if ":" in s:
        parts = s.split(":")
        if len(parts) > 3:
            raise ValueError(f"invalid slice index {index!r}")
        parts = (parts + ["", "", ""])[:3]
        to_int = lambda x: int(x) if x.strip() else None
        return slice(to_int(parts[0]), to_int(parts[1]), to_int(parts[2]))
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", s)
    if m:  # inclusive dash-range, e.g. "3-5" -> frames 3, 4, 5
        return slice(int(m.group(1)), int(m.group(2)) + 1)
    try:
        return int(s)
    except ValueError:
        raise ValueError(
            f"Unrecognised index {index!r}. Use ':', '3-5', ':3', '-3:', "
            "'::2', an int, or a slice.")


class Loader:
    """Load a file or ASE object into the internal ``Trajectory`` representation.

    Supported paths include LAMMPS-style files handled by OVITO and CIF/ASE
    inputs handled through ASE.  Loaded arrays are cached next to the source
    file as ``*.npy`` files so repeated loads avoid parser overhead.
    """

    # The cache stores parser output, so parser changes must invalidate old
    # arrays just as surely as changes to the source file or import options do.
    _CACHE_VERSION = "v1-" + source_files_version([__file__])

    def __init__(self,
                 filename: Optional[str] = None,
                 timestep: Optional[float] = None,
                 atom_mapping: Optional[Dict[int, Union[int, str]]] = None,
                 # Keep old parameters for backward compatibility but deprecated
                 atomic_numbers: Optional[Dict[int, int]] = None,
                 element_names: Optional[Dict[int, str]] = None,
                 ovitokwargs: Optional[Dict[str,str]] = None,
                 atoms = None,
                 index: Union[int, str, slice] = ":" ):
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
            index: Which frames/images to load, ASE-like. Default ":" loads all.
                Accepts an int, a slice, or a string: ":", ":3", "-3:", "::2",
                "3:5" (Python slice) or "3-5" (inclusive range). A step (e.g.
                "::2") rescales the trajectory timestep accordingly.
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

        # Frame selection (applied after loading; default ":" = all frames)
        self.index = index
        self._index = _parse_index(index)

    def _process_atom_mapping(self, mapping: Optional[Dict[int, Union[int, str]]]) -> Optional[Dict[int, int]]:
        """Convert atom mapping to atomic numbers."""
        if mapping is None:
            return None

        result = {}
        for atom_type, value in mapping.items():
            if isinstance(value, str):
                # Element name - convert to atomic number
                result[atom_type] = get_z_from_element(value)
            elif isinstance(value, int):
                # Already an atomic number
                if not (1 <= value <= 118):
                    raise ValueError(f"Invalid atomic number {value} for type {atom_type}. Must be between 1 and 118.")
                result[atom_type] = value
            else:
                raise ValueError(f"Invalid mapping value {value} for type {atom_type}. Must be int (atomic number) or str (element name).")

        return result

    def _resolve_ovito_types(self, atom_type_ids: np.ndarray,
                             type_names: Optional[Dict[int, str]]) -> np.ndarray:
        """Resolve OVITO particle-type IDs to element identities.

        Element identity comes from an explicit ``atom_mapping`` when provided,
        otherwise from the element names embedded in the file (OVITO's
        ``ParticleType.name``).  LAMMPS-style integer type IDs are NEVER used as
        atomic numbers; if neither source yields a valid element for every type
        present, loading aborts and asks for an explicit mapping.
        """
        atom_type_ids = np.asarray(atom_type_ids)
        used = [int(t) for t in np.unique(atom_type_ids)]

        # 1) Explicit mapping wins, but it must be exact (cover every type).
        if self.atomic_numbers is not None:
            missing = [t for t in used if t not in self.atomic_numbers]
            if missing:
                raise ValueError(
                    f"atom_mapping is missing entries for particle type(s) "
                    f"{missing}. Provide an exact mapping for every type, e.g. "
                    f"atom_mapping={{{used[0]}: 'Si', ...}}."
                )
            return np.array([self.atomic_numbers[int(t)] for t in atom_type_ids],
                            dtype=np.int32)

        # 2) Otherwise use the element names embedded in the file.
        type_names = type_names or {}
        symbols, unresolved = {}, []
        for t in used:
            name = str(type_names.get(t, "") or "").strip()
            try:
                get_z_from_element(name)  # validates it is a real element symbol
                symbols[t] = name
            except ValueError:
                unresolved.append((t, name))
        if unresolved:
            raise ValueError(
                "Could not identify elements from the file for particle "
                f"type(s) {[t for t, _ in unresolved]} (embedded names: "
                f"{[n for _, n in unresolved]!r}). LAMMPS type IDs are not "
                "element identities. Supply an exact atom_mapping, e.g. "
                f"atom_mapping={{{used[0]}: 'Si'}}."
            )
        return np.array([symbols[int(t)] for t in atom_type_ids])

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
            'box_matrix': cache_base.with_suffix(cache_base.suffix + '.box_matrix.npy'),
            'metadata': cache_base.with_suffix(cache_base.suffix + '.cache.json'),
        }

    def _cache_identity(self) -> dict:
        """Return all inputs that can change the cached parser output."""
        digest = hashlib.sha256()
        with open(self.filepath, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)

        mapping = None
        if self.atomic_numbers is not None:
            mapping = sorted(
                [int(atom_type), int(atomic_number)]
                for atom_type, atomic_number in self.atomic_numbers.items()
            )
        # Keep this as canonical JSON text. OVITO keyword values are normally
        # JSON-compatible; default=repr still gives uncommon custom values a
        # deterministic cache partition instead of disabling caching entirely.
        ovito_options = json.dumps(
            self.ovitokwargs, sort_keys=True, separators=(",", ":"), default=repr)
        return {
            "cache_version": self._CACHE_VERSION,
            "source_sha256": digest.hexdigest(),
            "atom_mapping": mapping,
            "ovito_options": ovito_options,
        }

    def _load_from_cache(self) -> Optional[Trajectory]:
        """Try to load trajectory from cached .npy files."""
        cache_files = self._get_cache_files()

        if not all(f.exists() for f in cache_files.values()):
            return None

        try:
            with open(cache_files['metadata']) as f:
                if json.load(f) != self._cache_identity():
                    logger.info("Ignoring stale cache for %s", self.filepath.name)
                    return None

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
        # Written last: the metadata file is the completion marker for the
        # four-array cache, as well as its provenance record.
        metadata_tmp = cache_files['metadata'].with_suffix(
            cache_files['metadata'].suffix + '.tmp')
        with open(metadata_tmp, 'w') as f:
            json.dump(self._cache_identity(), f, sort_keys=True)
        metadata_tmp.replace(cache_files['metadata'])

    def load(self) -> Trajectory:
        """Load structure/trajectory from file or ASE Atoms object and return as Trajectory."""
        # If atoms object provided, convert directly
        if self.atoms is not None:
            logger.info("Converting ASE Atoms object to Trajectory")
            return self._apply_frame_index(self.ase2Trajectory(self.atoms))

        # Try cache first (the cache always holds every frame; `index` is applied
        # as a view below, so different selections share one cache).
        trajectory = self._load_from_cache()
        if trajectory is None:
            # Load via OVITO or ASE
            # CIF files use ASE because OVITO's CIF parser is limited (e.g., fails on multi-block CIF files)
            if self.filepath.suffix in [".cif"]:
                logger.info(f"Loading {self.filepath.name} via ASE")
                trajectory = self._load_via_ase()
            else:
                logger.info(f"Loading {self.filepath.name} via OVITO")
                trajectory = self._load_via_ovito()
            self._save_to_cache(trajectory)

        return self._apply_frame_index(trajectory)

    def _apply_frame_index(self, trajectory: Trajectory) -> Trajectory:
        """Select frames per ``self._index`` (int or slice); ``:`` is a no-op."""
        n = trajectory.n_frames
        frame_ids = np.atleast_1d(np.arange(n)[self._index])
        if frame_ids.size == 0:
            raise ValueError(
                f"index {self.index!r} selected no frames from a {n}-frame source")
        if frame_ids.size == n and np.array_equal(frame_ids, np.arange(n)):
            return trajectory
        step = self._index.step if isinstance(self._index, slice) and self._index.step else 1
        return Trajectory(
            atom_types=trajectory.atom_types,
            positions=trajectory.positions[frame_ids],
            velocities=trajectory.velocities[frame_ids],
            box_matrix=trajectory.box_matrix,
            timestep=trajectory.timestep * abs(step),
        )

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
# OVITO stores lattice vectors in the columns of a 3x4 matrix (4th column is
# the origin); transpose to PySlice's row convention and keep the origin.
cell_np = np.array(frame0_data.cell.matrix, dtype=np.float32)
h_matrix = cell_np[:3, :3].T
cell_origin = cell_np[:3, 3].copy() if cell_np.shape[1] > 3 else np.zeros(3, dtype=np.float32)
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

if np.any(cell_origin != 0):
    positions -= cell_origin

pt = getattr(frame0_data.particles, "particle_types", None)
type_ids = []
type_names = []
if pt is not None and len(pt) == n_atoms:
    atom_types = np.array(pt, dtype=np.int32)
    for t in (getattr(pt, "types", None) or []):
        type_ids.append(int(t.id))
        type_names.append(str(t.name or ""))
else:
    print("No particle type data found. Treating all atoms as one type.", file=sys.stderr)
    atom_types = np.ones(n_atoms, dtype=np.int32)

np.savez(
    output_path,
    positions=positions,
    velocities=velocities,
    atom_types=atom_types,
    type_ids=np.array(type_ids, dtype=np.int32),
    type_names=np.array(type_names, dtype="<U16"),
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
                type_names = {int(i): str(n) for i, n
                              in zip(data["type_ids"], data["type_names"])}
                atom_types = self._resolve_ovito_types(data["atom_types"], type_names)
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
            raise ImportError(
                "OVITO is required to read this trajectory but is not installed. "
                "Install the optional reader with `pip install PySlice[io]` "
                "(OVITO is served from its own package index)."
            ) from exc

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

        # Extract basic info. OVITO stores the lattice vectors in the COLUMNS of
        # a 3x4 matrix (the 4th column is the cell origin). Transpose to the
        # row-vector convention used everywhere else in PySlice (and by the ASE
        # path), and keep the origin so positions can be referenced to it.
        n_atoms = len(frame0_data.particles.positions)
        h_matrix, cell_origin = _ovito_cell_to_row_convention(frame0_data.cell.matrix)

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

        # Reference positions to the cell origin so they live in [0, L), matching
        # the multislice grid (which starts at 0).
        if np.any(cell_origin != 0):
            positions -= cell_origin

        # Get per-atom type IDs and the element names OVITO embeds for each type.
        pt = getattr(frame0_data.particles, 'particle_types', None)
        if pt is not None and len(pt) == n_atoms:
            atom_type_ids = np.array(pt, dtype=np.int32)
            type_names = {int(t.id): (t.name or "").strip()
                          for t in (getattr(pt, 'types', None) or [])}
        else:
            logger.warning("No particle type data found. Treating all atoms as one type.")
            atom_type_ids = np.ones(n_atoms, dtype=np.int32)
            type_names = {}

        # Resolve to element identities (explicit mapping or embedded names;
        # never the raw LAMMPS type IDs).
        atom_types = self._resolve_ovito_types(atom_type_ids, type_names)

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
        # index=":" reads EVERY image: ase.io.read defaults to index=-1 (last
        # frame only), which silently dropped all but the last block of a
        # multi-frame/multi-block CIF. Frame selection is applied afterwards via
        # self._index so the cache still stores the full trajectory.
        atoms = aseread(str(self.filepath), index=":")
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
