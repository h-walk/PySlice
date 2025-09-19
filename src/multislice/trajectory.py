"""
Core trajectory data structure for molecular dynamics data.
"""
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Optional

@dataclass
class Trajectory:
    atom_types: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    box_matrix: np.ndarray
    timestep: float  # Timestep in picoseconds

    def __post_init__(self):
        """Validate trajectory data."""
        self._validate_shapes()

    def _validate_shapes(self):
        """Validate array shapes and consistency."""
        # Check dimensions
        if self.positions.ndim != 3 or self.positions.shape[2] != 3:
            raise ValueError(f"positions must be (frames, atoms, 3), got {self.positions.shape}")
        if self.velocities.ndim != 3 or self.velocities.shape[2] != 3:
            raise ValueError(f"velocities must be (frames, atoms, 3), got {self.velocities.shape}")
        if self.atom_types.ndim != 1:
            raise ValueError(f"atom_types must be 1D, got {self.atom_types.ndim}D")
        if self.box_matrix.shape != (3, 3):
            raise ValueError(f"box_matrix must be (3, 3), got {self.box_matrix.shape}")

        # Check consistency
        n_frames_pos, n_atoms_pos = self.positions.shape[:2]
        n_frames_vel, n_atoms_vel = self.velocities.shape[:2]
        n_atoms_types = len(self.atom_types)

        if n_frames_pos != n_frames_vel:
            raise ValueError(f"Frame count mismatch: {n_frames_pos} vs {n_frames_vel}")
        if not (n_atoms_pos == n_atoms_vel == n_atoms_types):
            raise ValueError(f"Atom count mismatch: {n_atoms_pos}, {n_atoms_vel}, {n_atoms_types}")

    @property
    def n_frames(self) -> int:
        """Number of frames in trajectory."""
        return self.positions.shape[0]

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the system."""
        return len(self.atom_types)

    @property
    def box_tilts(self) -> np.ndarray:
        """Extract box tilt angles from the box matrix off-diagonal elements."""
        return np.array([self.box_matrix[0, 1], self.box_matrix[0, 2], self.box_matrix[1, 2]])

    def get_mean_positions(self) -> np.ndarray:
        """Calculate the mean position for each atom over all frames."""
        if self.n_frames == 0:
            return np.empty((0, 3), dtype=self.positions.dtype)
        return np.mean(self.positions, axis=0)

    def tile_positions(self, repeats: Tuple[int, int, int]) -> 'Trajectory':
        """
        Tile the positions by repeating the system in 3D space.

        Args:
            repeats: Tuple of (nx, ny, nz) repeats in x, y, z directions

        Returns:
            New Trajectory with tiled positions
        """
        nx, ny, nz = repeats
        total_tiles = nx * ny * nz

        # Generate all tile offsets
        offsets = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    offset = self.box_matrix @ np.array([i, j, k])
                    offsets.append(offset)

        # Apply offsets to create tiled positions and velocities
        tiled_positions = []
        tiled_velocities = []
        tiled_atom_types = []

        for offset in offsets:
            tiled_positions.append(self.positions + offset)
            tiled_velocities.append(self.velocities)
            tiled_atom_types.append(self.atom_types)

        # Concatenate all tiles
        new_positions = np.concatenate(tiled_positions, axis=1)
        new_velocities = np.concatenate(tiled_velocities, axis=1)
        new_atom_types = np.concatenate(tiled_atom_types)

        # Scale box matrix
        new_box_matrix = self.box_matrix.copy()
        new_box_matrix[:, 0] *= nx  # Scale x-direction
        new_box_matrix[:, 1] *= ny  # Scale y-direction
        new_box_matrix[:, 2] *= nz  # Scale z-direction

        return Trajectory(
            atom_types=new_atom_types,
            positions=new_positions,
            velocities=new_velocities,
            box_matrix=new_box_matrix,
            timestep=self.timestep
        )

    def _validate_range(self, range_val: Optional[Tuple[float, float]], axis_name: str) -> Optional[Tuple[float, float]]:
        """Validate a coordinate range."""
        if range_val is None:
            return None

        min_val, max_val = range_val
        if min_val > max_val:
            raise ValueError(f"{axis_name} range invalid: min={min_val} > max={max_val}")

        return range_val

    def slice_positions(self,
                       x_range: Optional[Tuple[float, float]] = None,
                       y_range: Optional[Tuple[float, float]] = None,
                       z_range: Optional[Tuple[float, float]] = None) -> 'Trajectory':
        """
        Slice trajectory to include only atoms within specified spatial ranges.

        Args:
            x_range: (min, max) for X-axis filtering in Angstroms
            y_range: (min, max) for Y-axis filtering in Angstroms
            z_range: (min, max) for Z-axis filtering in Angstroms

        Returns:
            New Trajectory with only atoms in the specified spatial ranges
        """
        if self.n_atoms == 0:
            return self

        # Validate ranges
        x_range = self._validate_range(x_range, "X")
        y_range = self._validate_range(y_range, "Y")
        z_range = self._validate_range(z_range, "Z")

        # Check if any filtering is needed
        if all(r is None for r in [x_range, y_range, z_range]):
            return self

        # Use mean positions for spatial filtering
        mean_pos = self.get_mean_positions()
        if mean_pos.shape[0] == 0:
            return self

        # Apply filters
        atom_mask = np.ones(self.n_atoms, dtype=bool)
        new_box = self.box_matrix.copy()

        if x_range is not None:
            min_x, max_x = x_range
            atom_mask &= (mean_pos[:, 0] >= min_x) & (mean_pos[:, 0] <= max_x)
            new_box[0, 0] = max_x - min_x

        if y_range is not None:
            min_y, max_y = y_range
            atom_mask &= (mean_pos[:, 1] >= min_y) & (mean_pos[:, 1] <= max_y)
            new_box[1, 1] = max_y - min_y

        if z_range is not None:
            min_z, max_z = z_range
            atom_mask &= (mean_pos[:, 2] >= min_z) & (mean_pos[:, 2] <= max_z)
            new_box[2, 2] = max_z - min_z

        # Check results
        n_filtered = np.sum(atom_mask)
        if n_filtered == 0:
            ranges_desc = []
            if x_range: ranges_desc.append(f"X∈[{x_range[0]:.2f},{x_range[1]:.2f}]")
            if y_range: ranges_desc.append(f"Y∈[{y_range[0]:.2f},{y_range[1]:.2f}]")
            if z_range: ranges_desc.append(f"Z∈[{z_range[0]:.2f},{z_range[1]:.2f}]")
            raise ValueError(f"Filter {' AND '.join(ranges_desc)} resulted in 0 atoms")

        if n_filtered == self.n_atoms:
            return self

        # Create filtered trajectory
        return Trajectory(
            atom_types=self.atom_types[atom_mask],
            positions=self.positions[:, atom_mask, :],
            velocities=self.velocities[:, atom_mask, :],
            box_matrix=new_box,
            timestep=self.timestep
        )

    def slice_timesteps(self, frame_indices: List[int]) -> 'Trajectory':
        """
        Slice trajectory to include only specified timesteps.

        Args:
            frame_indices: List of frame indices to keep

        Returns:
            New Trajectory with only the specified timesteps
        """
        # Handle both lists and numpy arrays
        if isinstance(frame_indices, np.ndarray):
            if frame_indices.size == 0:
                raise ValueError("frame_indices cannot be empty")
        elif len(frame_indices) == 0:
            raise ValueError("frame_indices cannot be empty")

        # Validate indices
        max_idx = max(frame_indices)
        if max_idx >= self.n_frames:
            raise ValueError(f"Frame index {max_idx} out of range [0, {self.n_frames-1}]")

        return Trajectory(
            atom_types=self.atom_types,
            positions=self.positions[frame_indices, :, :],
            velocities=self.velocities[frame_indices, :, :],
            box_matrix=self.box_matrix,
            timestep=self.timestep
        )

    def generate_random_displacements(self,n_displacements,sigma):
        na=len(self.positions[0])
        dxyz=np.random.random(size=(n_displacements,na,3))*sigma
        positions = self.positions[0]+dxyz

        return Trajectory(
            atom_types=self.atom_types,
            positions=positions,
            velocities=np.ones(n_displacements)[:,None,None]*self.velocities[0, :, :],
            box_matrix=self.box_matrix,
            timestep=self.timestep
        )