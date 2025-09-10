import sys,os
sys.path.insert(1,"../../")
from src.io.loader import TrajectoryLoader
from src.multislice.potentials import gridFromTrajectory,Potential
import numpy as np
#from ..src.tacaw.ms_calculator_npy import gridFromTrajectory
#from src.tacaw.multislice_npy import Probe,Propagate ; import numpy as xp
#from src.tacaw.multislice_torch import Probe,PropagateBatch,create_batched_probes ; import torch as xp
#from src.tacaw.potential import Potential

dump="hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}

# LOAD MD OUTPUT
trajectory=TrajectoryLoader(dump,timestep=dt,element_names=types).load()


