import sys,os,time
try:
    import pyslice
except ModuleNotFoundError:
    print("import failed, falling back to relative paths")
    sys.path.insert(0, '../src')
from pyslice import Probe,Propagate,Potential,to_numpy
import matplotlib.pyplot as plt
import numpy as np

xs=np.linspace(0,11,111)
ys=np.linspace(0,10,101)

array = np.zeros((len(xs),len(ys),1))+1+.2*np.sin(10*xs[:,None,None]+13*ys[None,:,None])
array*=1000
#plt.imshow(array[:,:,0])
#plt.show()

O = Potential(xs, ys, [0], array=array)
P = probe=Probe(xs,ys,mrad=30,eV=100e3)
E = Propagate(P,O)

#E.plot()
#plt.imshow(np.absolute(E[0,:,:]))
#plt.show()
plt.imshow(np.absolute(np.fft.fftshift(np.fft.fft2(to_numpy(E[0,:,:])))))
plt.show()
