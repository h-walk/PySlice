"""
Core trajectory data structure for molecular dynamics data.
"""
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Optional
from ase import Atoms

@dataclass
class Trajectory:
    """Molecular trajectory data used by PySlice simulations.

    Arrays use shape ``(n_frames, n_atoms, 3)`` for positions and velocities.
    ``box_matrix`` is a 3x3 cell matrix with lattice vectors stored by row, and
    ``timestep`` is measured in picoseconds.
    """

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

    @property
    def extent(self) -> np.ndarray:
        """Return Cartesian extent (x, y, z) of the cell in Angstroms.

        Computes the bounding box of the parallelepiped defined by the lattice vectors.
        Useful for determining sampling: sampling = extent / n_pixels.
        """
        a, b, c = self.box_matrix[0], self.box_matrix[1], self.box_matrix[2]
        corners = np.array([
            [0, 0, 0], a, b, c, a+b, a+c, b+c, a+b+c
        ])
        return corners.max(axis=0) - corners.min(axis=0)

    def fold_positions_to_orthogonal_box(
        self,
        axes: Tuple[int, ...] = (0, 1, 2),
        lengths: Optional[Tuple[float, float, float]] = None,
    ) -> 'Trajectory':
        """
        Fold periodic coordinates into a diagonal orthogonal simulation box.

        This is useful for trajectories loaded from tilted cells when the atom
        coordinates already describe the desired periodic sheet, but some atoms
        appear on negative Cartesian sides of the box. The returned trajectory
        keeps all atoms and frames, wraps selected Cartesian axes into
        ``[0, length)``, and replaces the cell with ``diag(lengths)``.

        Args:
            axes: Cartesian axes to wrap, using 0=x, 1=y, 2=z.
            lengths: Orthogonal box lengths. Defaults to ``diag(box_matrix)``.

        Returns:
            New Trajectory with folded positions and a diagonal box matrix.
        """
        lengths_array = (
            np.diag(self.box_matrix).astype(self.positions.dtype)
            if lengths is None
            else np.asarray(lengths, dtype=self.positions.dtype)
        )
        if lengths_array.shape != (3,):
            raise ValueError(f"lengths must contain three box lengths, got shape {lengths_array.shape}")
        if np.any(lengths_array <= 0):
            raise ValueError(f"Cannot build an orthogonal box from lengths {lengths_array}")

        axes_tuple = tuple(axes)
        invalid_axes = [axis for axis in axes_tuple if axis not in (0, 1, 2)]
        if invalid_axes:
            raise ValueError(f"axes must contain only 0, 1, or 2; got {invalid_axes}")

        positions = self.positions.copy()
        for axis in axes_tuple:
            positions[:, :, axis] = np.mod(positions[:, :, axis], lengths_array[axis])

        return Trajectory(
            atom_types=self.atom_types.copy(),
            positions=positions,
            velocities=self.velocities.copy(),
            box_matrix=np.diag(lengths_array),
            timestep=self.timestep
        )

    def get_mean_positions(self) -> np.ndarray: # TODO: THIS DOES NOT INCLUDE WRAPPING (what if an atom drifts out of the bbox and comes back in through PBC?)
        """Calculate the mean position for each atom over all frames."""
        if self.n_frames == 0:
            return np.empty((0, 3), dtype=self.positions.dtype)
        return np.mean(self.positions, axis=0)
 
    def get_distplacements(self) -> np.ndarray:
        """Return per-frame displacements from each atom's time-averaged position."""
        return self.positions[:,:,:]-self.get_mean_positions()[None,:,:]

    def tile_positions(self, repeats: Tuple[int, int, int], trajectories:list = None) -> 'Trajectory':
        """
        Tile one or more trajectories into a repeated supercell.

        Args:
            repeats: Tuple of (nx, ny, nz) repeats in x, y, z directions
            trajectories: Optional trajectory per tile. If omitted, this trajectory is
                repeated into every tile.

        Returns:
            New Trajectory with tiled positions
        """
        nx, ny, nz = repeats
        total_tiles = nx * ny * nz

        # Generate all tile offsets
        # Note: box_matrix uses row convention (rows are lattice vectors a, b, c)
        # so offset = [i, j, k] @ box_matrix = i*a + j*b + k*c
        offsets = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    offset = np.array([i, j, k]) @ self.box_matrix
                    offsets.append(offset)

        #print("self.box_matrix",self.box_matrix)

        # Apply offsets to create tiled positions and velocities
        tiled_positions = []
        tiled_velocities = []
        tiled_atom_types = []

        if trajectories is None or len(trajectories) != len(offsets):
            trajectories = [ self ]*len(offsets)

        for offset,traj in zip(offsets,trajectories):
            tiled_positions.append(traj.positions + offset)
            tiled_velocities.append(traj.velocities)
            tiled_atom_types.append(traj.atom_types)

        # Concatenate all tiles
        new_positions = np.concatenate(tiled_positions, axis=1)
        new_velocities = np.concatenate(tiled_velocities, axis=1)
        new_atom_types = np.concatenate(tiled_atom_types)

        # Scale box matrix (row convention: row 0 = a, row 1 = b, row 2 = c)
        new_box_matrix = self.box_matrix.copy()
        new_box_matrix[0, :] *= nx  # Scale a vector (row 0)
        new_box_matrix[1, :] *= ny  # Scale b vector (row 1)
        new_box_matrix[2, :] *= nz  # Scale c vector (row 2)

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

    def swap_axes(self,axes):
        """Return a trajectory with Cartesian axes permuted."""
        positions_swapped = self.positions[:,:,axes]
        velocities_swapped = self.velocities[:,:,axes]
        box_swapped = self.box_matrix[axes,:][:,axes]
        return Trajectory(
            atom_types=self.atom_types,
            positions=positions_swapped,
            velocities=velocities_swapped,
            box_matrix=box_swapped,
            timestep=self.timestep
        )

    def tilt_positions(self,alpha=0,beta=0) -> 'Trajectory':
        """Rotate positions and velocities by x-axis then y-axis tilt angles.

        Args:
            alpha: Rotation angle about x in radians.
            beta: Rotation angle about y in radians.
        """
        ca=np.cos(alpha) ; sa=np.sin(alpha)
        cb=np.cos(beta) ; sb=np.sin(beta)
        Ra=np.asarray([[1,0,0],[0,ca,-sa],[0,sa,ca]])
        Rb=np.asarray([[cb,0,-sb],[0,1,0],[sb,0,cb]])
        nt,na,nxyz = np.shape(self.positions)
        pos = np.reshape(self.positions,(nt*na,3)).T
        vel = np.reshape(self.velocities,(nt*na,3)).T
        positions_tilted = np.reshape( ( Rb @ Ra @ pos ).T , (nt,na,3) )
        velocities_tilted = np.reshape( ( Rb @ Ra @ vel ).T , (nt,na,3) )
        
        return Trajectory(
            atom_types=self.atom_types,
            positions=positions_tilted,
            velocities=velocities_tilted,
            box_matrix=self.box_matrix,
            timestep=self.timestep
        )

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

        # Apply filters. The box is shrunk to each range's width, so the kept
        # atoms must also be translated by the range lower bound; otherwise they
        # keep their original coordinates and fall outside the new [0, width) box.
        atom_mask = np.ones(self.n_atoms, dtype=bool)
        new_box = self.box_matrix.copy()
        origin_shift = np.zeros(3, dtype=self.positions.dtype)

        if x_range is not None:
            min_x, max_x = x_range
            atom_mask &= (mean_pos[:, 0] >= min_x) & (mean_pos[:, 0] <= max_x)
            new_box[0, 0] = max_x - min_x
            origin_shift[0] = min_x

        if y_range is not None:
            min_y, max_y = y_range
            atom_mask &= (mean_pos[:, 1] >= min_y) & (mean_pos[:, 1] <= max_y)
            new_box[1, 1] = max_y - min_y
            origin_shift[1] = min_y

        if z_range is not None:
            min_z, max_z = z_range
            atom_mask &= (mean_pos[:, 2] >= min_z) & (mean_pos[:, 2] <= max_z)
            new_box[2, 2] = max_z - min_z
            origin_shift[2] = min_z

        # Check results
        n_filtered = np.sum(atom_mask)
        if n_filtered == 0:
            ranges_desc = []
            if x_range: ranges_desc.append(f"X∈[{x_range[0]:.2f},{x_range[1]:.2f}]")
            if y_range: ranges_desc.append(f"Y∈[{y_range[0]:.2f},{y_range[1]:.2f}]")
            if z_range: ranges_desc.append(f"Z∈[{z_range[0]:.2f},{z_range[1]:.2f}]")
            raise ValueError(f"Filter {' AND '.join(ranges_desc)} resulted in 0 atoms")

        # Create the cropped trajectory even when every atom survives. A range
        # request changes the coordinate origin and box; returning self in that
        # case silently discarded the requested crop.
        return Trajectory(
            atom_types=self.atom_types[atom_mask],
            positions=self.positions[:, atom_mask, :] - origin_shift,
            velocities=self.velocities[:, atom_mask, :],
            box_matrix=new_box,
            timestep=self.timestep
        )

    def slice_timesteps(self, i1: int=0, i2: Optional[int]=None, ith: int=1) -> 'Trajectory':
        """
        Slice trajectory to include only specified timesteps.

        Args:
            i1: first frame index to keep (default 0, i.e., first),
                OR a list of specific indices to select
            i2: first frame index to ignore (defaults to include last)
            ith: used for selecting alternating frames. 2 = every-other, 3 = every 3rd etc

        Returns:
            New Trajectory with only the specified timesteps
        """
        # Handle explicit indices - delegate to select_timesteps.
        if isinstance(i1, (list, tuple, np.ndarray)):
            return self.select_timesteps(i1)

        if i2 is None:
            i2 = len(self.positions)

        return Trajectory(
            atom_types=self.atom_types,
            positions=self.positions[i1:i2:ith, :, :],
            velocities=self.velocities[i1:i2:ith, :, :],
            box_matrix=self.box_matrix,
            timestep=self.timestep*ith
        )

    def select_timesteps(self, indices) -> 'Trajectory':
        """
        Select specified timesteps.

        Args:
            indices:   indices of timesteps to be selected

        Returns:
            New Trajectory with only the specified timesteps
        """

        if indices is None:
            indices = slice(0, len(self.positions), 1)
        elif isinstance(indices, (int, np.integer)):
            indices = [indices]
        elif isinstance(indices, tuple):
            indices = list(indices)

        # Note in this case the timestep variable is meaningless,
        # maybe we can find a more elegant solution (?) 
        return Trajectory(
            atom_types=self.atom_types,
            positions=self.positions[indices, :, :],
            velocities=self.velocities[indices, :, :],
            box_matrix=self.box_matrix,
            timestep=0.0
        )

    def get_random_timesteps(self, N: int=1, seed: Optional[int] = None) -> 'Trajectory':
        """
        Filter trajectory down to random frames (useful for conventional frozen-phonon)

        Args:
            N: the number of frames to keep
            seed: randomization seed to supply to numpy random
        Returns:
            New Trajectory with randomized frames
        """

        return self.random_frames(N=N, seed=seed)

    def random_frames(self, N: int, seed: Optional[int] = None) -> 'Trajectory':
        """Return a new trajectory with ``N`` randomly selected frames."""
        rng = np.random.default_rng(seed)
        indices = rng.choice(self.n_frames, size=N, replace=False)
        return Trajectory(
            atom_types=self.atom_types,
            positions=self.positions[indices, :, :],
            velocities=self.velocities[indices, :, :],
            box_matrix=self.box_matrix,
            timestep=self.timestep
        )

    def generate_random_displacements(self, n_displacements, sigma, seed=None):
        """Generate frozen-phonon frames from Gaussian displacements.

        Args:
            n_displacements: Number of displaced frames to generate.
            sigma: RMS displacement magnitude in Angstroms.
            seed: Optional NumPy random seed.
        """
        rng = np.random.default_rng(seed)
        na = self.n_atoms
        dxyz = rng.normal(0, sigma / np.sqrt(3), size=(n_displacements, na, 3))
        positions = self.positions[0] + dxyz

        return Trajectory(
            atom_types=self.atom_types,
            positions=positions,
            velocities=np.broadcast_to(self.velocities[0], (n_displacements, na, 3)).copy(),
            box_matrix=self.box_matrix,
            timestep=self.timestep
        )

    def plot(self, timestep=0, view='3d', alpha=0.6, size=20):
        """
        Plot atomic positions.

        Args:
            timestep: Which frame to plot
            view: '3d' for 3D scatter, 'xy', 'xz', or 'yz' for 2D projections
            alpha: Transparency for overlapping atoms (0-1)
            size: Marker size
        """
        import matplotlib.pyplot as plt

        # Draw box outline
        box = self.box_matrix
        corners = np.array([
            [0, 0, 0], [box[0,0], 0, 0], [box[0,0], box[1,1], 0], [0, box[1,1], 0],
            [0, 0, box[2,2]], [box[0,0], 0, box[2,2]], [box[0,0], box[1,1], box[2,2]], [0, box[1,1], box[2,2]]
        ])
        edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]

        if view == '3d':
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(projection='3d')

            # Use different colors and sizes for different atom types
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            unique_types = sorted(set(self.atom_types))

            for at, c in zip(unique_types, colors):
                mask = self.atom_types == at
                xs = self.positions[timestep, mask, 0]
                ys = self.positions[timestep, mask, 1]
                zs = self.positions[timestep, mask, 2]
                ax.scatter(xs, ys, zs, c=c, s=size, alpha=alpha, label=str(at), edgecolors='none')

            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.legend()

            for edge in edges:
                pts = corners[list(edge)]
                ax.plot3D(pts[:,0], pts[:,1], pts[:,2], 'k-', alpha=0.3, linewidth=1)
        else:
            fig, ax = plt.subplots(figsize=(8, 8))

            for edge in edges:
                pts = corners[list(edge)]
                xyzs = [[pts[:,0], pts[:,1], pts[:,2]]["xyz".index(c)] for c in view ]
                ax.plot(xyzs[0], xyzs[1], 'k-', alpha=0.3, linewidth=1)

            # Select projection
            if view == 'xy':
                idx1, idx2 = 0, 1
                labels = ('X (Å)', 'Y (Å)')
            elif view == 'xz':
                idx1, idx2 = 0, 2
                labels = ('X (Å)', 'Z (Å)')
            elif view == 'yz':
                idx1, idx2 = 1, 2
                labels = ('Y (Å)', 'Z (Å)')
            else:
                raise ValueError(f"view must be '3d', 'xy', 'xz', or 'yz', got {view}")

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            unique_types = sorted(set(self.atom_types))

            for at, c in zip(unique_types, colors):
                mask = self.atom_types == at
                x = self.positions[timestep, mask, idx1]
                y = self.positions[timestep, mask, idx2]
                ax.scatter(x, y, c=c, s=size, alpha=alpha, label=str(at), edgecolors='none')

            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.legend()
            ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()


    def rotate_to(self, direction: Tuple[int, int, int]) -> 'Trajectory':
        """
        Rotate the structure so that a crystallographic direction aligns with the z-axis.

        Args:
            direction: Miller indices [h, k, l] defining the direction to align with z

        Returns:
            New Trajectory with rotated positions, velocities, and box_matrix
        """
        h, k, l = direction

        # Convert Miller indices to real-space Cartesian direction using box_matrix
        # The direction in reciprocal space becomes a real-space vector via box_matrix^T
        miller_vec = np.array([h, k, l], dtype=float)
        direction_cart = self.box_matrix.T @ miller_vec

        # Normalize the direction vector
        direction_cart = direction_cart / np.linalg.norm(direction_cart)

        # Target direction is z-axis
        z_axis = np.array([0.0, 0.0, 1.0])

        # Calculate rotation axis and angle using Rodrigues' rotation formula
        # The rotation axis is perpendicular to both vectors: k = v1 × v2
        # The rotation angle is: θ = arccos(v1 · v2)
        # The rotation matrix is: R = I + sin(θ)*K + (1-cos(θ))*K²
        # where K is the cross-product matrix of the rotation axis

        # If direction is already aligned (or anti-aligned) with z, handle specially
        dot_product = np.dot(direction_cart, z_axis)
        if np.abs(dot_product - 1.0) < 1e-10:
            # Already aligned, return copy
            return Trajectory(
                atom_types=self.atom_types,
                positions=self.positions.copy(),
                velocities=self.velocities.copy(),
                box_matrix=self.box_matrix.copy(),
                timestep=self.timestep
            )
        elif np.abs(dot_product + 1.0) < 1e-10:
            # Anti-aligned, rotate 180° around any perpendicular axis (use x)
            rotation_matrix = np.array([
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0]
            ])
        else:
            # General case: use Rodrigues' formula
            rotation_axis = np.cross(direction_cart, z_axis)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

            # Build the cross-product matrix K
            kx, ky, kz = rotation_axis
            K = np.array([
                [0.0, -kz, ky],
                [kz, 0.0, -kx],
                [-ky, kx, 0.0]
            ])

            # Rodrigues' rotation matrix: R = I + sin(θ)*K + (1-cos(θ))*K²
            I = np.eye(3)
            rotation_matrix = I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

        # Apply rotation to positions and velocities
        n_frames, n_atoms, _ = self.positions.shape
        positions_flat = self.positions.reshape(-1, 3)
        velocities_flat = self.velocities.reshape(-1, 3)

        rotated_positions_flat = positions_flat @ rotation_matrix.T
        rotated_velocities_flat = velocities_flat @ rotation_matrix.T

        rotated_positions = rotated_positions_flat.reshape(n_frames, n_atoms, 3)
        rotated_velocities = rotated_velocities_flat.reshape(n_frames, n_atoms, 3)

        # Rotate the box_matrix as well (row convention: rows are lattice vectors)
        rotated_box_matrix = self.box_matrix @ rotation_matrix.T

        return Trajectory(
            atom_types=self.atom_types,
            positions=rotated_positions,
            velocities=rotated_velocities,
            box_matrix=rotated_box_matrix,
            timestep=self.timestep
        )

    # returns an ase object
    def to_ase(self):
        """Convert the first frame to an ASE ``Atoms`` object."""
        return Atoms(''.join(self.atom_types), positions=self.positions[0], cell=self.box_matrix, pbc=True)
