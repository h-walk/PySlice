# This serves as a quick and dirty memory stress test, by tiling the hBN example system out to 5x5 and running Midgley diffraction

import sys,os
from pyslice.io.loader import Loader
from pyslice.multislice.multislice import wavelength
from pyslice.multislice.calculators import MultisliceCalculator
import numpy as np
import matplotlib.pyplot as plt
import os,shutil

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}
a,b=2.4907733333333337,2.1570729817355123

# WHAT IS MIDGELEY?
#  \  Defocused sample yields diffraction spots at
#   \  the original sample plane. Each spot is a
#    \___________/  real-space probe, and the
#    /\       .-'  magnification of the diffraction
#   /  \   .-'/  \  pattern depends on the distance
#  /    \-'  /    \  defocused. With Midgley, you're
# /  .-' \  / '-.  \  interested in this diffraction
#/.-'     \/     '-.\  pattern, LACBED builds on this
#        (__) with a selected-area aperture on spot
#        /  \  then going back to reciprocal space to
#       /    \  view the "texture" of a single disk


# QUICK MATH: WHERE WILL OUR REAL-SPACE BRAGG SPOTS END UP?
l=wavelength(100e3)
defocus = 10000
dx=defocus*l/a ; dy=defocus*l/b # # d sin(theta) = n lamda
print(dx,dy)


# LOAD TRAJECTORY
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
trajectory = trajectory.get_random_timesteps(1,seed=0)
trajectory = trajectory.tile_positions([5,5,1])
#trajectory.plot(view='xy')

# CONVERGENT BEAM, DEFOCUSED
calculator=MultisliceCalculator()
calculator.setup(trajectory,aperture=5,voltage_eV=100e3,sampling=.1,slice_thickness=.5,cache=False,cache_potentials=True)

calculator.base_probe.defocus(-defocus)

#calculator.base_probe.plot()
exitwaves = calculator.run()
calculator.base_probe.aberrate({"C30":50000})
exitwaves.propagate_free_space(defocus-calculator.lz)

# REAL SPACE EXIT WAVE SHOWS PROBES FORMING THE DIFFRACTION PATTERN
exitwaves.plot_realspace()

