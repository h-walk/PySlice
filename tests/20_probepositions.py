import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice import Loader,MultisliceCalculator,HAADFData,differ,gridFromTrajectory,Potential

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
# SELECT 10 "RANDOM" TIMESTEPS (use seed for reproducibility)
trajectory=trajectory.get_random_timesteps(3,seed=5)
# REPLACE A SINGLE ATOM WITH SILICON
dxyz = trajectory.positions[0,:,:]-np.asarray([a,b*4/3,0])[None,:]
distances = np.sqrt(np.sum((dxyz)**2,axis=1))
i = np.argmin(distances) # which atom is closest to a,b?
trajectory.atom_types[i] = 28
# PREVIEW IT:
xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)
potential = Potential(xs, ys, zs, trajectory.positions[0], trajectory.atom_types, kind="kirkland")
potential.plot()
# TRIM TO 10x10 UC
trajectory=trajectory.slice_positions([0,10*a],[0,10*b])
# CHECK AGAIN, POST-CROP
xs,ys,zs,lx,ly,lz=gridFromTrajectory(trajectory,sampling=0.1,slice_thickness=0.5)
potential = Potential(xs, ys, zs, trajectory.positions[0], trajectory.atom_types, kind="kirkland")
potential.plot()
# CREATE CALCULATOR OBJECT
calculator=MultisliceCalculator()
# SET UP GRID OF HAADF SCAN POINTS
#xy=probe_grid([a,3*a],[b,3*b],14,16)
#calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_positions=xy,cache_wavefunctions=False)
probe_xs = np.linspace(a/2,2.5*a,14)
probe_ys = np.linspace(b/2,2.5*b,16)
calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,cache_wavefunctions=False)
# RUN MULTISLICE
exitwaves = calculator.run()

haadf=HAADFData(exitwaves)
ary=haadf.calculateADF(preview=False) # use preview=True to view the collection angles of the ADF detector in reciprocal space
xs=haadf.xs ; ys=haadf.ys

#fig, ax = plt.subplots()
#ax.imshow(ary.T, cmap="inferno")
#plt.show()
haadf.plot()#"outputs/figs/04_haadf.png")

#ary=np.asarray(ary)
#differ(ary[::4,::4],"outputs/haadf-test.npy","HAADF")
