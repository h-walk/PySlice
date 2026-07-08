import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice import Loader,wavelength,MultisliceCalculator,TACAWData

import numpy as np
import matplotlib.pyplot as plt
import shutil

#if os.path.exists("psi_data"):
#	shutil.rmtree("psi_data")

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}
a,b=2.4907733333333337,2.1570729817355123

# LOAD TRAJECTORY
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
# TRIM TO 10x10 UC
#trajectory=trajectory.slice_positions([0,20*a],[0,20*b])
# SELECT 10 "RANDOM" TIMESTEPS (use seed for reproducibility)
trajectory=trajectory.get_random_timesteps(3,seed=5)
# CREATE CALCULATOR OBJECT
calculator=MultisliceCalculator()
# CONVERGENT BEAM
calculator.setup(trajectory,aperture=5,voltage_eV=100e3)

exitwaves = calculator.run()

exitwaves.recenter()
exitwaves.plot_realspace(filename="outputs/figs/26_lenses_0_orig.png")
exitwaves.pad_real_space(100,100)
exitwaves.plot_realspace(filename="outputs/figs/26_lenses_1_pad.png")

#exitwaves.plot_reciprocal(powerscaling=0.125)

exitwaves.propagate_free_space(1000)
exitwaves.plot_realspace(filename="outputs/figs/26_lenses_2_100nm.png")
exitwaves.propagate_through_lens(1000)
for n in range(30):
    exitwaves.propagate_free_space(50)
    exitwaves.plot_realspace(filename="outputs/figs/26_lenses_"+str(n+3)+"_"+str(100+5*(n+1))+"nm.png")

