import sys,os
sys.path.insert(1,"../../")
from src.io.loader import TrajectoryLoader
from src.multislice.multislice import probe_grid
from src.multislice.calculators import MultisliceCalculator
from src.postprocessing.haadf_data import HAADFData
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
trajectory=trajectory.slice_positions([0,10*a],[0,10*b])
# SELECT 10 "RANDOM" TIMESTEPS (use seed for reproducibility)
slice_timesteps = np.arange(trajectory.n_frames)
np.random.seed(5) ; np.random.shuffle(slice_timesteps)
slice_timesteps = slice_timesteps[:3] # 3 random frames (test we can do multiple but don't bog down the test)
trajectory=trajectory.slice_timesteps( slice_timesteps )
# SET UP GRID OF HAADF SCAN POINTS
xy=probe_grid([a,3*a],[b,3*b],14,16)

calculator=MultisliceCalculator()
calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_positions=xy)
exitwaves = calculator.run()

#print(exitwaves.wavefunction_data.shape)
# exitwaves.wavefunction_data is reciprocal-space now! 
#ary=np.mean(np.absolute(exitwaves.wavefunction_data[:,:,:,:,-1]),axis=1)
#q=np.sqrt(exitwaves.kxs[:,None]**2+exitwaves.kys[None,:]**2)
#mask=np.zeros(q.shape) ; mask[q>2]=1
#fig, ax = plt.subplots()
#print(ary.shape,q.shape)
#ax.imshow(np.absolute(ary[0])**.1, cmap="inferno")
#plt.show()
#fig, ax = plt.subplots()
#HAADF=np.sum(np.absolute(ary*mask[None,:]),axis=(1,2)).reshape((len(x),len(y)))
#ax.imshow(HAADF, cmap="inferno")
#plt.show()

haadf=HAADFData(exitwaves)
ary=haadf.ADF(preview=False)
xs=haadf.xs ; ys=haadf.ys

fig, ax = plt.subplots()
ax.imshow(ary.T, cmap="inferno")
plt.show()

ary=np.asarray(ary)
if not os.path.exists("haadf-test.npy"):
	np.save("haadf-test.npy",ary)
else:
	previous=np.load("haadf-test.npy")
	F , D = np.absolute(ary) , np.absolute(previous)
	dz=np.sum( (F-D)**2 ) / np.sum( F**2 ) # a scaling-resistant values-near-zero-resistance residual function
	if dz>1e-6:
		print("ERROR! EXIT WAVE DOES NOT MATCH PREVIOUS RUN",dz*100,"%")
