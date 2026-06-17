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
#trajectory=trajectory.slice_positions([0,10*a],[0,10*b])					# TRIM TO 10x10 UC
trajectory=trajectory.get_random_timesteps(1,seed=5)						# SELECT "RANDOM" TIMESTEPS
for x,y,m in [[2*a,b*4/3,12],[2*a,b*4/3+b,14],[3.5*a,b*4/3,16]]:			# ADD DOPANTS (for testing scan lims)
    dxyz = trajectory.positions[0,:,:]-np.asarray([x,y,0])[None,:]
    distances = np.sqrt(np.sum((dxyz)**2,axis=1))
    i = np.argmin(distances) # which atom is closest to a,b?
    trajectory.atom_types[i] = m
# SELECT 10 "RANDOM" TIMESTEPS (use seed for reproducibility)
trajectory=trajectory.get_random_timesteps(2,seed=5)
# CREATE CALCULATOR OBJECT
calculator=MultisliceCalculator()
# SET UP GRID OF HAADF SCAN POINTS
#xy=probe_grid([a,3*a],[b,3*b],14,16)
#calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_positions=xy,cache_wavefunctions=False)
lx,ly,lz=np.diag(trajectory.box_matrix)
probe_xs = np.linspace(0,lx,1024) ; probe_ys = np.linspace(0,ly,1024)
calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,prism=50,loop_probes=7,use_memmap=False,kth=10,ADF=True,cache_wavefunctions=False,return_layers=None)
# RUN MULTISLICE
exitwaves,haadf = calculator.run()
#exitwaves.plot_realspace(whichProbe=500)

#fig, ax = plt.subplots()
#ary = calculator.base_probe.array[0,25,:,:]
#ax.imshow(ary.T, cmap="inferno")
#plt.show()

#haadf=HAADFData(exitwaves)
#ary=haadf.calculateADF(preview=False) # use preview=True to view the collection angles of the ADF detector in reciprocal space
#xs=haadf.xs ; ys=haadf.ys

#fig, ax = plt.subplots()
#ax.imshow(ary.T, cmap="inferno")
#plt.show()
haadf.plot("outputs/figs/22_prism.png")
