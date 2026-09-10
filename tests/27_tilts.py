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
trajectory=trajectory.slice_positions([0,10*a],[0,10*b])
# SELECT 10 "RANDOM" TIMESTEPS (use seed for reproducibility)
trajectory=trajectory.get_random_timesteps(3,seed=5)
# CREATE CALCULATOR OBJECT
calculator=MultisliceCalculator()
# SET UP GRID OF HAADF SCAN POINTS
#xy=probe_grid([a,3*a],[b,3*b],14,16)
#calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_positions=xy,cache_wavefunctions=False)
probe_xs = np.linspace(3*a,6*a,14)
probe_ys = np.linspace(3*b,6*b,16)
calculator.setup(trajectory,aperture=5,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,probe_tilt=(30,5))#,kth=4)#,cache_wavefunctions=False)
# RUN MULTISLICE
calculator.base_probe.plot(filename="outputs/figs/27_tilts_probe.png")
exitwaves = calculator.run()

exitwaves.plot_reciprocal(filename="outputs/figs/27_tilts_cbed.png")
