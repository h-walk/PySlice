import sys,os
sys.path.insert(1,"../../")
from src.io.loader import TrajectoryLoader
from src.multislice.multislice import Probe,Propagate
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
xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)

# GENERATE PROBE (ENSURE 00_PROBE.PY PASSES BEFORE RUNNING)
probe=Probe(xs,ys,mrad=5,eV=100e3)

# GENERATE THE POTENTIAL (ENSURE 01_POTENTIAL.PY PASSES BEFORE RUNNING)
positions = trajectory.positions[0]
atom_types=trajectory.atom_types
potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")

# TEST PROPAGATION
ary=np.asarray( Propagate(probe,potential) )

if not os.path.exists("propagate-test.npy"):
	np.save("propagate-test.npy",ary)
else:
	previous=np.load("propagate-test.npy")
	F , D = np.absolute(ary) , np.absolute(previous)
	dz=np.sum( (F-D)**2 ) / np.sum( F**2 ) # a scaling-resistant values-near-zero-resistance residual function
	if dz>1e-6:
		print("ERROR! EXIT WAVE DOES NOT MATCH PREVIOUS RUN",dz*100,"%")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
#ax.imshow(np.absolute(ary), cmap="inferno")
#plt.show()
ax.imshow(np.absolute(np.fft.fftshift(np.fft.fft2(ary)))**.1, cmap="inferno")
plt.show()
