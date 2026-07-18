import sys,os
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

# LOAD TRAJECTORY
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
# TRIM TO 10x10 UC
trajectory=trajectory.slice_positions([0,20*a],[0,20*b])
# SELECT 10 "RANDOM" TIMESTEPS (use seed for reproducibility)
trajectory=trajectory.get_random_timesteps(3,seed=5)

for counts in [1e3,1e4,1e5]:
    # CREATE CALCULATOR OBJECT
    calculator=MultisliceCalculator()
    # SET UP MULTISLICE FOR TEM DIFFRACTION
    calculator.setup(trajectory,aperture=3,voltage_eV=100e3,sampling=.1,slice_thickness=.5)
    # RUN MULTISLICE
    exitwaves = calculator.run()
    exitwaves.counts(counts)
    exitwaves.plot(filename="outputs/figs/23_counts_"+str(int(counts))+".png",powerscaling=.1,extent=(-2,2,-2,2),title=str(int(counts))+" counts")
