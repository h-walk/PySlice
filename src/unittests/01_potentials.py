import sys,os
sys.path.insert(1,"../../")
from src.io.loader import TrajectoryLoader
from src.multislice.potentials import gridFromTrajectory,Potential
import numpy as np
#from ..src.tacaw.ms_calculator_npy import gridFromTrajectory
#from src.tacaw.multislice_npy import Probe,Propagate ; import numpy as xp
#from src.tacaw.multislice_torch import Probe,PropagateBatch,create_batched_probes ; import torch as xp
#from src.tacaw.potential import Potential

dump="hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}

# LOAD MD OUTPUT
trajectory=TrajectoryLoader(dump,timestep=dt,element_names=types).load()

# TEST GENERATION OF THE POTENTIAL
positions = trajectory.positions[0]
atom_types=trajectory.atom_types
xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)
potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
ary=potential.to_cpu()  # Convert to CPU numpy array properly

if not os.path.exists("potentials-test.npy"):
	np.save("potentials-test.npy",ary)
else:
	previous=np.load("potentials-test.npy")
	F , D = np.absolute(ary) , np.absolute(previous)
	dz=np.sum( (F-D)**2 ) / np.sum( F**2 ) # a scaling-resistant values-near-zero-resistance residual function
	if dz>1e-6:
		print("ERROR! POTENTIAL DOES NOT MATCH PREVIOUS RUN",dz*100,"%")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(np.sum(ary,axis=2), cmap="inferno")
plt.show()
