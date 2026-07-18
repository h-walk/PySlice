import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice import Loader,grid_from_trajectory,Potential,differ

import numpy as np

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}

# LOAD MD OUTPUT
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()

# TEST GENERATION OF THE POTENTIAL
positions = trajectory.positions[0]
atom_types=trajectory.atom_types
xs,ys,zs,lx,ly,lz=grid_from_trajectory(trajectory,sampling=0.1,slice_thickness=0.5)
potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
potential.build()
ary=potential.array  # ".array" converts torch tensor to CPU numpy array automatically if required
print(ary.shape)
differ(ary[::20,::20,::2],"outputs/potentials-test.npy","POTENTIAL")

potential.plot("outputs/figs/01_potentials.png")

#import matplotlib.pyplot as plt
#fig, ax = plt.subplots()
#ax.imshow(np.sum(ary,axis=2), cmap="inferno")
#plt.show()
