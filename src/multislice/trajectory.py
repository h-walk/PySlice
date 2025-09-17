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
        # Validate required fields
        if self.positions.ndim != 3 or self.positions.shape[2] != 3:
            raise ValueError("Positions must be 3D (frames, atoms, xyz) and last dimension must be 3.")
        if self.velocities.ndim != 3 or self.velocities.shape[2] != 3:
            raise ValueError("Velocities must be 3D (frames, atoms, xyz) and last dimension must be 3.")
        if self.atom_types.ndim != 1:
            raise ValueError("atom_types must be 1D")
        if self.box_matrix.shape != (3, 3):
            raise ValueError(f"Box matrix must be 3x3, got {self.box_matrix.shape}")

        # Validate consistency
        if not (self.positions.shape[0] == self.velocities.shape[0]):
            raise ValueError("Frame count mismatch: positions, velocities.")
        if not (self.positions.shape[1] == self.velocities.shape[1] == len(self.atom_types)):
            raise ValueError("Atom count mismatch: positions, velocities, atom_types.")

    @property
    def n_frames(self) -> int:
        return self.positions.shape[0]

    @property
    def n_atoms(self) -> int:
        return len(self.atom_types)

    @property
    def box_tilts(self) -> np.ndarray:
        """Extract box tilt angles from the box matrix off-diagonal elements."""
        return np.array([self.box_matrix[0,1], self.box_matrix[0,2], self.box_matrix[1,2]])

    def get_mean_positions(self) -> np.ndarray:
        """Calculate the mean position for each atom over all frames."""
        if self.n_frames == 0:
            # Return empty array with correct shape if no frames
            return np.empty((0, 3), dtype=self.positions.dtype) 
        return np.mean(self.positions, axis=0)




    def tile_positions(self, repeats: tuple[int, int, int]) -> 'Trajectory':
        """
        Tile the positions by repeating the system in 3D space.

        Args:
            repeats: Tuple of (nx, ny, nz) repeats in x, y, z directions

        Returns:
            New Trajectory with tiled positions
        """
        nx, ny, nz = repeats
        n_atoms = self.n_atoms
        n_frames = self.n_frames

        # Calculate new positions
        new_positions = []
        new_velocities = []
        new_atom_types = []

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if i == j == k == 0:
                        continue  # Skip the original system

                    offset = np.dot(self.box_matrix, np.array([i, j, k]))
                    shifted_positions = self.positions + offset
                    shifted_velocities = self.velocities  # Keep same velocities

                    new_positions.append(shifted_positions)
                    new_velocities.append(shifted_velocities)
                    new_atom_types.extend(self.atom_types)

        # Combine with original
        all_positions = [self.positions] + new_positions
        all_velocities = [self.velocities] + new_velocities

        tiled_positions = np.concatenate(all_positions, axis=1)
        tiled_velocities = np.concatenate(all_velocities, axis=1)
        tiled_atom_types = np.concatenate([self.atom_types] + [np.array(new_atom_types)])

        return Trajectory(
            atom_types=tiled_atom_types,
            positions=tiled_positions,
            velocities=tiled_velocities,
            box_matrix=self.box_matrix * np.array([[nx, ny, nz]]).T,  # Scale box matrix
            timestep=self.timestep,

        )

    def slice_positions(self, x_range: Optional[Tuple[float, float]] = None,
                       y_range: Optional[Tuple[float, float]] = None,
                       z_range: Optional[Tuple[float, float]] = None) -> 'Trajectory':
        """
        Slice trajectory to include only atoms within specified spatial ranges.

        Args:
            x_range: Tuple (min_x, max_x) for X-axis filtering in Angstroms. None to skip X filtering.
            y_range: Tuple (min_y, max_y) for Y-axis filtering in Angstroms. None to skip Y filtering.
            z_range: Tuple (min_z, max_z) for Z-axis filtering in Angstroms. None to skip Z filtering.

        Returns:
            New Trajectory with only atoms in the specified spatial ranges

        Example:
            # Keep atoms with x between 0-5 Å, y between 0-5 Å, z between 0-5 Å
            sliced_traj = trajectory.slice_positions(x_range=(0, 5), y_range=(0, 5), z_range=(0, 5))
        """
        if self.n_atoms == 0:
            return self

        # Check if any filtering is requested
        if x_range is None and y_range is None and z_range is None:
            return self

        # Use mean positions for spatial filtering
        mean_pos = self.get_mean_positions()
        if mean_pos.shape[0] == 0:
            return self

        # Start with all atoms included
        atom_mask = np.ones(self.n_atoms, dtype=bool)
        new_box = np.zeros( self.box_matrix.shape ) + self.box_matrix

        # Apply X filtering if requested
        if x_range is not None:
            min_x, max_x = x_range
            if min_x > max_x:
                raise ValueError(f"X min_coord ({min_x}) cannot be greater than max_coord ({max_x}).")
            x_mask = (mean_pos[:, 0] >= min_x) & (mean_pos[:, 0] <= max_x)
            atom_mask &= x_mask
            new_box[0,0] = max_x - min_x

        # Apply Y filtering if requested
        if y_range is not None:
            min_y, max_y = y_range
            if min_y > max_y:
                raise ValueError(f"Y min_coord ({min_y}) cannot be greater than max_coord ({max_y}).")
            y_mask = (mean_pos[:, 1] >= min_y) & (mean_pos[:, 1] <= max_y)
            atom_mask &= y_mask
            new_box[1,1] = max_y - min_y

        # Apply Z filtering if requested
        if z_range is not None:
            min_z, max_z = z_range
            if min_z > max_z:
                raise ValueError(f"Z min_coord ({min_z}) cannot be greater than max_coord ({max_z}).")
            z_mask = (mean_pos[:, 2] >= min_z) & (mean_pos[:, 2] <= max_z)
            atom_mask &= z_mask
            new_box[2,2] = max_z - min_z

        num_original_atoms = self.n_atoms
        filtered_atom_indices = np.where(atom_mask)[0]
        num_filtered_atoms = len(filtered_atom_indices)

        if num_filtered_atoms == 0:
            ranges = []
            if x_range: ranges.append(f"X ∈ [{x_range[0]:.3f}, {x_range[1]:.3f}]")
            if y_range: ranges.append(f"Y ∈ [{y_range[0]:.3f}, {y_range[1]:.3f}]")
            if z_range: ranges.append(f"Z ∈ [{z_range[0]:.3f}, {z_range[1]:.3f}]")
            filter_desc = " AND ".join(ranges)
            raise ValueError(f"Spatial filter criteria ({filter_desc}) resulted in 0 atoms. Please adjust filter ranges.")

        if num_filtered_atoms == num_original_atoms:
            # No change, return self
            return self

        # Create new Trajectory with filtered data
        new_positions = self.positions[:, atom_mask, :]
        new_velocities = self.velocities[:, atom_mask, :]
        new_atom_types = self.atom_types[atom_mask]

        return Trajectory(
            atom_types=new_atom_types,
            positions=new_positions,
            velocities=new_velocities,
            box_matrix=new_box,
            timestep=self.timestep,

        )

    def slice_timesteps(self, frame_indices: List[int]) -> 'Trajectory':
        """
        Slice trajectory to include only specified timesteps.

        Args:
            frame_indices: List of frame indices to keep

        Returns:
            New Trajectory with only the specified timesteps
        """
        sliced_positions = self.positions[frame_indices, :, :]
        sliced_velocities = self.velocities[frame_indices, :, :]

        return Trajectory(
            atom_types=self.atom_types,
            positions=sliced_positions,
            velocities=sliced_velocities,
            box_matrix=self.box_matrix,
            timestep=self.timestep,

        )

