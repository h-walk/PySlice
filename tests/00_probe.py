import os
os.environ["MPLBACKEND"] = "Agg"

import sys,time
try:
    import pyslice
except ModuleNotFoundError:
    print("import failed, falling back to relative paths")
    sys.path.insert(0, '../src')
start=time.time()
from pyslice.multislice.multislice import Probe
print("probe import took",time.time()-start,"s")

start=time.time()
from pyslice import differ
print("differ import took",time.time()-start,"s")

import numpy as np

# Generate a few dummy probes
xs=np.linspace(0,50,501)
ys=np.linspace(0,49,491)

mrads=[1,3,5,15,30]
ary=np.zeros((5,501,491),dtype=complex)
for i,mrad in enumerate(mrads):
	probe=Probe(xs,ys,mrad=mrad,eV=100e3,preview=False,gaussianVOA=0)
	probe.plot("outputs/figs/00_probe_"+str(i)+".png")
	#if hasattr(probe, 'to_cpu'):
	#	ary[i] = probe.to_cpu()
	#else:
	#	ary[i] = np.asarray(probe.array)
	ary[i] = probe.array # ".array" converts torch tensor to CPU numpy array automatically if required

ary=np.reshape(ary,(501*5,491))

differ(ary[::10,::10],"outputs/probe-test.npy","PROBE")

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(np.absolute(ary.T)**.25, cmap="inferno")
plt.savefig("outputs/figs/00_probe.png")
#plt.show()
