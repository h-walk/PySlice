import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice import Loader,Probe,Propagate,create_batched_probes,grid_from_trajectory,Potential,differ

import numpy as np
import matplotlib.pyplot as plt

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}
a,b=2.4907733333333337,2.1570729817355123

# LOAD MD OUTPUT
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
trajectory=trajectory.slice_positions([0,10*a],[0,10*b])
xs,ys,zs,lx,ly,lz=grid_from_trajectory(trajectory,sampling=0.1,slice_thickness=0.5)

# GENERATE PROBE (ENSURE 00_PROBE.PY PASSES BEFORE RUNNING)
#probe=Probe(xs,ys,mrad=30,eV=100e3)
xsp = np.linspace(10*a-a,10*a-3*a,16)
ysp = np.linspace(10*b-b,10*b-3*b,16)
#x,y=np.meshgrid(xs,ys)
#xy=np.reshape([x,y],(2,len(x.flat))).T
#print(xy)
#probes_many=create_batched_probes(probe,xy) # we stopped recommending create_batched_probes many commits ago
probes_many=Probe(xs,ys,mrad=30,eV=100e3,probe_xs=xsp,probe_ys=ysp) # creates a flattened list of probes, each shifted to center on each x,y point

# GENERATE THE POTENTIAL (ENSURE 01_POTENTIAL.PY PASSES BEFORE RUNNING)
positions = trajectory.positions[0]
atom_types=trajectory.atom_types
potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")

# SANITY CHECK THAT OUR CROPPED TRAJECTORY IS CORRECT
#ary=np.asarray(potential.array)
#fig, ax = plt.subplots()
#ax.imshow(np.sum(ary,axis=2), cmap="inferno")
#dx=potential.xs[1]-potential.xs[0] ; dy=potential.ys[1]-potential.ys[0]
#ax.plot(x/dx,y/dy)
#plt.show()
#pr=probes_many.array[5,:,:]
#po=np.sum(potential.array,axis=2)
#fig, ax = plt.subplots()
#ax.imshow(np.absolute(pr)*np.absolute(po), cmap="inferno")
#plt.show()

# TEST PROPAGATION
result = Propagate(probes_many,potential)
kxs = potential.kxs # also retrieve kx,ky since we may need to convert from Torch
kys = potential.kys
# result may be a torch tensor (since Calculators and friends don't want the exit wave moved off-device yet, and we do not expect the end user to call Propagate directly)
if hasattr(result, 'cpu'):
    ary = result.cpu().numpy()  # Convert PyTorch tensor to numpy
else:
    ary = np.asarray(result)  # Already numpy array
if hasattr(kxs,'cpu'):
    kxs = np.asarray(kxs.cpu())
    kys = np.asarray(kys.cpu())

print(ary.shape)
differ(ary[::5,::5,::5],"outputs/manyprobes-test.npy","EXIT WAVE")

# ASSEMBLE HAADF IMAGE
q=np.sqrt(kxs[:,None]**2+kys[None,:]**2)
fig, ax = plt.subplots()
fft=np.fft.fft2(ary,axes=(1,2)) ; fft[:,q<2]=0 # mask in reciprocal space (keep only high scattering angles)
#ax.imshow(np.absolute(np.fft.fftshift(fft[0]))**.1, cmap="inferno")
#plt.show()
HAADF=np.sum(np.absolute(fft),axis=(1,2)).reshape((len(xsp),len(ysp)))
ax.imshow(HAADF, cmap="inferno")
plt.savefig("outputs/figs/03_manyprobes.png")

