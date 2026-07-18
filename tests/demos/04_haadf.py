import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice import Loader,MultisliceCalculator,HAADFData,differ

import numpy as np
import matplotlib.pyplot as plt
import os,shutil

#if os.path.exists("psi_data"):
#	shutil.rmtree("psi_data")

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}
a,b=2.4907733333333337,2.1570729817355123

# LOAD TRAJECTORY
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
# TRIM TO 10x10 UC
trajectory=trajectory.slice_positions([0,10*a],[0,10*b])
# SELECT 10 "RANDOM" TIMESTEPS (use seed for reproducibility)
trajectory=trajectory.get_random_timesteps(3,seed=5)
# CREATE CALCULATOR OBJECT
calculator=MultisliceCalculator()
# SET UP GRID OF HAADF SCAN POINTS
#xy=probe_grid([a,3*a],[b,3*b],14,16)
#calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_positions=xy,cache_wavefunctions=False)
probe_xs = np.linspace(10*a-a,10*a-3*a,14)
probe_ys = np.linspace(10*b-b,10*b-3*b,16)
calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys)#,kth=4)#,cache_wavefunctions=False)
# RUN MULTISLICE
exitwaves = calculator.run()

exitwaves.plot_reciprocal(filename="outputs/figs/04_haadf_cbed.png")

# 4D-STEM EXAMPLE, OR, DIY-ADF, JUST TO DEMO HOW TO NAVIGATE THE AXES
#ary = np.asarray(exitwaves.reshaped) # .array would be p,t,kx,ky,l indices, but .reshaped is x,y,t,kx,ky,l
#xs = exitwaves.probe_xs ; ys = exitwaves.probe_ys # real-space dimensions
#kx = exitwaves.kxs ; ky = exitwaves.kys		# reciprocal-space dimensions
#print(ary.shape,len(xs),len(ys),len(kx),len(ky))
#mask = np.zeros((len(kx),len(ky)))				# for ADF we'll mask in reciprocal space
#kr = np.sqrt(kx[:,None]**2+ky[None,:]**2)
#mask[kr>3]=1
#ary*=mask[None,None,None,:,:,None]				# apply mask along x,y,t,[kx,ky],l axes
#ary=np.sum(np.absolute(ary),axis=(2,3,4,5))	# sume along x,y,[t,kx,ky,l] axes
#fig, ax = plt.subplots()
#ax.imshow(ary.T, cmap="inferno")
#plt.show()

haadf=HAADFData(exitwaves)
ary=haadf.calculateADF(preview=False) # use preview=True to view the collection angles of the ADF detector in reciprocal space
xs=haadf.xs ; ys=haadf.ys

#fig, ax = plt.subplots()
#ax.imshow(ary.T, cmap="inferno")
#plt.show()
haadf.plot("outputs/figs/04_haadf.png")

haadf.to_sea("04_haadf.sea")

ary=np.asarray(ary)
differ(ary[::4,::4],"outputs/haadf-test.npy","HAADF")
