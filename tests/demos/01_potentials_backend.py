import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice.io.loader import Loader
from pyslice.multislice.potentials import grid_from_trajectory,Potential
from pyslice.postprocessing.testtools import differ
import numpy as np
#from ..pyslice.tacaw.ms_calculator_npy import grid_from_trajectory
#from pyslice.tacaw.multislice_npy import Probe,Propagate ; import numpy as xp
#from pyslice.tacaw.multislice_torch import Probe,PropagateBatch,create_batched_probes ; import torch as xp
#from pyslice.tacaw.potential import Potential

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
potential_cpu = Potential(xs, ys, zs, positions, atom_types, kind="kirkland", device='cpu')

potential.build()
potential_cpu.build()

ary=potential.to_numpy()
ary_cpu=potential_cpu.to_numpy()

print(np.max(ary-ary_cpu, axis=(0,1)))

differ(ary_cpu[::20,::20,::2],"outputs/potentials-test.npy","POTENTIAL")

potential.plot("outputs/figs/01_potentials_backend.png")

#import matplotlib.pyplot as plt
#fig, ax = plt.subplots()
#ax.imshow(np.sum(ary,axis=2), cmap="inferno")
#plt.show()
