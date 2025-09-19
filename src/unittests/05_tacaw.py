import sys,os
sys.path.insert(1,"../../")
from src.io.loader import TrajectoryLoader
from src.multislice.multislice import probe_grid
from src.multislice.calculators import MultisliceCalculator
from src.postprocessing.tacaw_data import TACAWData
import numpy as np
import matplotlib.pyplot as plt
import os,shutil

#if os.path.exists("psi_data"):
#	shutil.rmtree("psi_data")

dump="hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}
a,b=2.4907733333333337,2.1570729817355123

# LOAD TRAJECTORY
trajectory=TrajectoryLoader(dump,timestep=dt,atom_mapping=types).load()
# TRIM TO 10x10 UC
#trajectory=trajectory.slice_positions([0,10*a],[0,10*b])

# TACAW CALCULATION: ALL TIMESTEPS, LET'S DO PARALLEL BEAM
calculator=MultisliceCalculator()
calculator.setup(trajectory,aperture=0,voltage_eV=100e3,sampling=.1,slice_thickness=.5)
exitwaves = calculator.run()

tacaw = TACAWData(exitwaves)
print(tacaw.frequencies)
ary=np.asarray( tacaw.intensity[0,65,:,:]**.1 )

fig, ax = plt.subplots()
ax.imshow(ary.T, cmap="inferno")
plt.show()

if not os.path.exists("tacaw-test.npy"):
	np.save("tacaw-test.npy",ary)
else:
	previous=np.load("tacaw-test.npy")
	F , D = np.absolute(ary) , np.absolute(previous)
	dz=np.sum( (F-D)**2 ) / np.sum( F**2 ) # a scaling-resistant values-near-zero-resistance residual function
	if dz>1e-6:
		print("ERROR! TACAW SLICE DOES NOT MATCH PREVIOUS RUN",dz*100,"%")
