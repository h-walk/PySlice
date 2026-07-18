import sys,os,itertools
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice import Loader,MultisliceCalculator,TACAWData,differ

import numpy as np
import matplotlib.pyplot as plt
import os,shutil

#if os.path.exists("psi_data"):
#	shutil.rmtree("psi_data")

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}
a,b=2.4907733333333337,2.1570729817355123

skip = -1
if len(sys.argv)>1:
    skip = int(sys.argv[-1])

# each kwarg and it's options
options = { "loop_probes":[False,10], "use_memmap":[False,True], "chunkFFT":[False,True] , "min_dk":[0,.1] , "kth":[1,3] }
args = list(options.keys())
# all permutations: [0,0,0,0,0], [0,0,0,0,1], and so on
indices = list(itertools.product([0,1],repeat=len(args)))
kwargCombos = []
for ijklm in indices:
    dic = { k:options[k][n] for k,n in zip(args,ijklm) }
    kwargCombos.append( dic )
#print(kwargCombos)

for n,kwargs in enumerate(kwargCombos):
    os.system("rm -rf psi_data")
    for i in range(2):
        if n<=skip:
            continue
        print("RUNNING ITERATION",n,"/",len(kwargCombos),["a","b"][i],"TACAW WITH KWARGS:",kwargs)
        trajectory = Loader(dump,timestep=dt,atom_mapping=types).load()                   # LOAD TRAJECTORY
        trajectory = trajectory.slice_positions([0,10*a],[0,10*b])                        # TRIM TO 10x10 UC
        trajectory = trajectory.slice_timesteps(0,100,2)
        calculator=MultisliceCalculator()                                               # CREATE CALCULATOR OBJECT
        probe_xs = np.linspace(10*a-a,10*a-3*a,3)                                      # SET UP GRID OF HAADF SCAN POINTS
        probe_ys = np.linspace(10*b-b,10*b-3*b,4)
        calcKwargs = { k:v for k,v in kwargs.items() if k!="chunkFFT" }
        calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,**calcKwargs)
        exitwaves = calculator.run()
        tacaw=TACAWData(exitwaves,chunkFFT=kwargs["chunkFFT"])
        Z = tacaw.spectral_diffraction(30) #; print(Z.shape)
        if kwargs["kth"]==1 and kwargs["min_dk"]==0:
            differ(Z[:,:]**.1,"outputs/tacawotf-test.npy","TACAW SLICE")
        diff=tacaw.diffraction().T
        kx=np.asarray(tacaw.kxs) ; kx=kx[kx>=0] ; kx=kx[kx<=4/a] ; print("kx",kx.shape)
        dispersion = tacaw.dispersion( kx , np.zeros(len(kx))+2/b )

