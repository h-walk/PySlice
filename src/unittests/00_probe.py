import sys,os
sys.path.insert(1,"../../")
from src.multislice.multislice import Probe
import numpy as np

# Generate a few dummy probes
xs=np.linspace(0,50,501)
ys=np.linspace(0,49,491)

mrads=[1,3,5,15,30]
ary=np.zeros((5,501,491),dtype=complex)
for i,mrad in enumerate(mrads):
	probe=Probe(xs,ys,mrad=mrad,eV=100e3)
	probe.plot()
	if hasattr(probe, 'to_cpu'):
		ary[i] = probe.to_cpu()
	else:
		ary[i] = np.asarray(probe.array)
ary=np.reshape(ary,(501*5,491))

if not os.path.exists("probe-test.npy"):
	np.save("probe-test.npy",ary)
else:
	previous=np.load("probe-test.npy")
	F , D = np.absolute(ary) , np.absolute(previous)
	dz=np.sum( (F-D)**2 ) / np.sum( F**2 ) # a scaling-resistant values-near-zero-resistance residual function
	if dz>1e-6:
		print("ERROR! POTENTIAL DOES NOT MATCH PREVIOUS RUN",dz*100,"%")



import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(np.absolute(ary.T)**.25, cmap="inferno")
plt.show()
