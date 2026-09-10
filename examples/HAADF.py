import sys,os
sys.path.insert(1,"../PySlice/staging/src")
from pyslice import Loader,MultisliceCalculator,HAADFData

import numpy as np
import matplotlib.pyplot as plt
import shutil

#dump="inputs/hBN_truncated.lammpstrj"
#dump = "/Volumes/Alexandria/ORNL/MD/projects/harrisonBN_ML/monolayer300k.lammpstrj"
#dt=.005
#types={1:"B",2:"N"}
#a,b=2.4907733333333337,2.1570729817355123

# LOAD TRAJECTORY
#trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
# TRIM TO 10x10 UC
#trajectory=trajectory.slice_positions([0,4*a],[0,4*b])
# SELECT 30 "RANDOM" TIMESTEPS (use seed for reproducibility)
#trajectory=trajectory.get_random_timesteps(30,seed=5)

# OR, DO FROZEN PHONON
cif = os.path.join(os.path.dirname(__file__), '..', 'tests', 'inputs', 'hBN_cif.cif')
trajectory=Loader(cif).load()
a,b,c = np.diag(trajectory.box_matrix)
trajectory = trajectory.tile_positions([5,5,1])
trajectory = trajectory.generate_random_displacements(50,sigma=.2,seed=5)
trajectory.plot(view='xy')

# SET UP GRID OF HAADF SCAN POINTS
#xy=probe_grid([a,3*a],[b,3*b],32,32)
xs = np.linspace(a,3*a,32,endpoint=False)
ys = np.linspace(b,3*b,32,endpoint=False)
# RUN MULTISLICE
calculator=MultisliceCalculator()
calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=xs,probe_ys=ys)
calculator.preview_probes()
exitwaves = calculator.run()
# CALCULATE ADF
haadf=HAADFData(exitwaves)
ary=haadf.calculateADF(preview=False,inner_mrad=45, outer_mrad=150) # use preview=True to view the collection angles of the ADF detector in reciprocal space
# PLOT IT
haadf.plot("HAADF.svg")
haadf.plot("HAADF.png")
