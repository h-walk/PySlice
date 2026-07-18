import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice import Loader,Probe,Propagate,grid_from_trajectory,Potential,differ

import numpy as np

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}

# LOAD MD OUTPUT
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
xs,ys,zs,lx,ly,lz=grid_from_trajectory(trajectory,sampling=0.1,slice_thickness=0.5)

# GENERATE PROBE (ENSURE 00_PROBE.PY PASSES BEFORE RUNNING)
probe=Probe(xs,ys,mrad=5,eV=100e3)

# GENERATE THE POTENTIAL (ENSURE 01_POTENTIAL.PY PASSES BEFORE RUNNING)
positions = trajectory.positions[0]
atom_types=trajectory.atom_types
potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")

# TEST PROPAGATION
# Handle device conversion properly for PyTorch tensors
result = Propagate(probe,potential,onthefly=True)
# result may be a torch tensor (since Calculators and friends don't want the exit wave moved off-device yet, and we do not expect the end user to call Propagate directly)
if hasattr(result, 'cpu'):
    ary = result.cpu().numpy()  # Convert PyTorch tensor to numpy
else:
    ary = np.asarray(result)  # Already numpy array
ary=ary[0,:,:]

differ(ary[::10,::10],"outputs/propagate-test.npy","EXIT WAVE")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
#ax.imshow(np.absolute(ary), cmap="inferno")
#plt.show()
ax.imshow(np.absolute(np.fft.fftshift(np.fft.fft2(ary)))**.1, cmap="inferno")
plt.savefig("outputs/figs/02_propagate_otf=True.png")

#result.plot()
