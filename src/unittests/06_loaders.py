import sys,os
sys.path.insert(1,"../../")
from src.io.loader import TrajectoryLoader
from src.multislice.potentials import gridFromTrajectory,Potential
import matplotlib.pyplot as plt
import numpy as np

testFiles=["hBN.cif","hBN.xyz","hBN_truncated.lammpstrj"]

for filename in testFiles:
	trajectory=TrajectoryLoader(filename).load()
	trajectory = trajectory.generate_random_displacements(n_displacements=10,sigma=1)
	print(len(trajectory.positions))
	positions = trajectory.positions[0]
	atom_types=trajectory.atom_types
	xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)
	potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")

	ary=potential.to_cpu()
	fig, ax = plt.subplots()
	ax.imshow(np.sum(ary,axis=2), cmap="inferno")
	plt.show()
