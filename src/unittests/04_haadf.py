import sys,os
sys.path.insert(1,"../../")
from src.io.loader import TrajectoryLoader
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
trajectory=TrajectoryLoader(dump,timestep=dt,element_names=types).load()
# TRIM TO 10x10 UC
trajectory=trajectory.slice_positions([0,10*a],[0,10*b])
# SELECT 10 "RANDOM" TIMESTEPS (use seed for reproducibility)
slice_timesteps = np.arange(trajectory.n_frames)
np.random.seed(5) ; np.random.shuffle(slice_timesteps)
slice_timesteps = slice_timesteps[:10]
trajectory=trajectory.slice_timesteps( slice_timesteps )
# SET UP GRID OF HAADF SCAN POINTS
x,y=np.meshgrid(np.linspace(a,3*a,16),np.linspace(b,3*b,16))
xy=np.reshape([x,y],(2,len(x.flat))).T

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

haadf=HAADFData(exitwaves).ADF(preview=False)

fig, ax = plt.subplots()
ax.imshow(haadf.T, cmap="inferno")
plt.show()
