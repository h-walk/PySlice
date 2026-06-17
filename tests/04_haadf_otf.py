import sys,os,itertools
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

start = 0 ; end = np.inf
if len(sys.argv)>1:
    start = int(sys.argv[1])
if len(sys.argv)>2:
    end = int(sys.argv[2])

# each kwarg and it's options
options = { "ADF":[False,True], "loop_probes":[False,10], "use_memmap":[False,True], "prism":[False,25], "keep_wavefunctions":[True,False], "min_dk":[0.0,.1], "kth":[1,3], "skip_vacuum":[False,True] }
args = list(options.keys())
# all permutations: [0,0,0,0,0], [0,0,0,0,1], and so on
indices = list(itertools.product([0,1],repeat=len(args)))
kwargCombos = []
for ijklm in indices:
    dic = { k:options[k][n] for k,n in zip(args,ijklm) }
    kwargCombos.append( dic )

#for n,kwargs in enumerate(kwargCombos):
#    print(str(n)+"/"+str(len(kwargCombos))+"\t"+"  ".join([k[:7]+":"+str(v)[:4] for k,v in kwargs.items()]))
#sys.exit()

for n,kwargs in enumerate(kwargCombos):
    os.system("rm -rf psi_data")
    for i in range(2):
        if not kwargs.get("ADF",False) and not kwargs.get("keep_wavefunctions",False): # skip nonsense combo (since ADF-in-post requires wavefunction_data returned)
            continue
        if n<start or n>end:
            continue
        #os.system("rm -rf psi_data")
        print("RUNNING ITERATION",n,"/",len(kwargCombos),["a","b"][i],"HAADF WITH KWARGS:",kwargs)
        trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()                   # LOAD TRAJECTORY
        trajectory=trajectory.slice_positions([0,10*a],[0,10*b])                        # TRIM TO 10x10 UC
        trajectory=trajectory.get_random_timesteps(3,seed=5)                            # SELECT 10 "RANDOM" TIMESTEPS (use seed for reproducibility)
        calculator=MultisliceCalculator()                                               # CREATE CALCULATOR OBJECT
        probe_xs = np.linspace(10*a-a,10*a-3*a,14)                                      # SET UP GRID OF HAADF SCAN POINTS
        probe_ys = np.linspace(10*b-b,10*b-3*b,16)
        calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,**kwargs)
        if kwargs.get("ADF",False):
            exitwaves,haadf = calculator.run()                                          # RUN MULTISLICE, RETURNS HAADF OBJECT SINCE WE USED ADF=True FLAG
            ary = haadf.array
        else:
            exitwaves = calculator.run()
            haadf=HAADFData(exitwaves)                                                  # NO NEED FOR HAADF CALCULATOR SINCE WE DID IT ON THE FLY
            ary=haadf.calculateADF()
        haadf.plot("outputs/figs/04_haadf_otf_"+str(n)+["a","b"][i]+".png")
        ary=np.asarray(ary)
        differ(ary[::4,::4],"outputs/haadf-test.npy","HAADF")
        if kwargs.get("prism",False):
            print("(but don't worry, we do not expect it to match)")
