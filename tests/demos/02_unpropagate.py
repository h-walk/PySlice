import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice import Loader,Probe,Propagate,grid_from_trajectory,Potential,differ,to_numpy,calculateObject
import matplotlib.pyplot as plt
import numpy as np

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}

for flatten in [True,False]:

    # LOAD MD OUTPUT
    trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
    xs,ys,zs,lx,ly,lz=grid_from_trajectory(trajectory,sampling=0.1,slice_thickness=0.5)

    # GENERATE PROBE (ENSURE 00_PROBE.PY PASSES BEFORE RUNNING)
    xpr=[lx/2,lx/2+5] ; ypr=[ly/2]*2 ; Os=[]
    for x,y in zip(xpr,ypr):

        probe=Probe(xs,ys,mrad=5,eV=100e3,probe_xs=[x],probe_ys=[y])
        probe.defocus(200)
        #probe.plot()

        # GENERATE THE POTENTIAL (ENSURE 01_POTENTIAL.PY PASSES BEFORE RUNNING)
        positions = trajectory.positions[0]
        atom_types=trajectory.atom_types
        potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
        potential.plot()
        potential.build()
        if flatten:
            potential.flatten() # TECHNICALLY calculateObject ONLY RETURNS THE TRULY CORRECT SOLUTION FOR A SINGLE SLICE
        p_arry = np.absolute(np.sum(to_numpy(potential.array),axis=2))

        #fig, ax = plt.subplots()
        #ax.imshow(p_arry, cmap="inferno")
        #plt.title("Original Potential")
        #plt.show()

        # PROPAGATION
        # Handle device conversion properly for PyTorch tensors
        result = Propagate(probe,potential,onthefly=True)
        res = to_numpy(result[0,:,:])
        #fig, ax = plt.subplots()
        #ax.imshow(np.absolute(res), cmap="inferno")
        #plt.title("|Exit Wave|")
        #plt.show()
        #fig, ax = plt.subplots()
        #ax.imshow(np.angle(res), cmap="inferno")
        #plt.title("Phi Exit Wave")
        #plt.show()

        # RECALCULATE OBJECT FROM THE RESULT
        dO = calculateObject(probe,result[0,:,:],np.zeros((len(xs),len(ys))),weighting=1,dz=0.5)
        Os.append(dO)
        dO = to_numpy(dO)
        fig, ax = plt.subplots()
        ax.imshow(np.absolute(dO)**.1, cmap="inferno")
        plt.title("Reconstructed Potential")
        plt.show()

    O = np.sum([to_numpy(o) for o in Os],axis=0)
    fig, ax = plt.subplots()
    ax.imshow(np.absolute(O)**.1, cmap="inferno")
    plt.title("summed reconstructed")
    plt.show()


    fig, ax = plt.subplots()
    delta = np.absolute(dO-p_arry)
    ax.imshow(delta**.1, cmap="inferno")
    plt.title("|OP-RP|/|OP|="+str(np.amax(delta)/np.amax(p_arry)))
    plt.show()
