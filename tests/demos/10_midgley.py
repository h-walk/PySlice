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
dx=1000*l/a ; dy=1000*l/b # # d sin(theta) = n lamda

# LOAD TRAJECTORY
trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
#trajectory = trajectory.slice_timesteps(0,200,2)

# CONVERGENT BEAM, DEFOCUSED
calculator=MultisliceCalculator()
calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5)
calculator.base_probe.defocus(-1000)
exitwaves = calculator.run()
exitwaves.propagate_free_space(1000-calculator.lz)

# REAL SPACE EXIT WAVE SHOWS PROBES FORMING THE DIFFRACTION PATTERN
exitwaves.plot_realspace(filename="outputs/figs/10_midgley_diff.png")

# ENERGY RESOLVED:
tacaw = TACAWData(exitwaves,keep_complex=True)

# DIFFRACTION FROM TACAW:
diff=tacaw.diffraction(space="real").T # # swap default x,y to imshow's y,x since tacaw.plot does not transpose
tacaw.plot(diff**.125,"x","y",filename="outputs/figs/10_midgley_diff2.png") 

# PLOT AN ENERGY SLICE:
Z = tacaw.spectral_diffraction(30,space="real").T
tacaw.plot(Z**.125,"x","y",filename="outputs/figs/10_midgley_30THz.png") 

# PLOT THE DISPERSION
xs=np.asarray(tacaw.xs) ; xm = np.mean(xs) ; xs=xs[xs>=xm] ; xs=xs[xs<=xm+4*dx]
ys=np.asarray(tacaw.ys) ; ym = np.mean(ys) ; ys = np.zeros(len(xs))+ym+dy
dispersion = tacaw.dispersion( xs , ys , space="real")
tacaw.plot(dispersion**.125,xs,"omega",filename="outputs/figs/10_midgley_disp.png")



