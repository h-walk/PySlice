import ase,sys,os
import numpy as np
sys.path.insert(1,"../abTEM/fixnumpy2")
import abtem


cif = os.path.join(os.path.dirname(__file__), '..', 'tests', 'inputs', 'hBN_cif.cif')
repetitions = (5, 5, 1)
atoms = ase.io.read(cif) * repetitions
frozen_phonons = abtem.FrozenPhonons(atoms, num_configs=50, sigmas=0.2, seed=5)
potential = abtem.Potential(frozen_phonons, sampling=0.1, parametrization='kirkland', slice_thickness=0.5)


probe = abtem.Probe(energy=100e3, semiangle_cutoff=30)
probe.grid.match(potential)

detector = abtem.AnnularDetector(inner=45, outer=150) # defaults from PySlice/src/postprocessing/haadf_data.py > HAADFData > calculateADF

a,b,c = np.load("abc.npy")
#xy=probe_grid([a,3*a],[b,3*b],32,32)

grid_scan = abtem.GridScan(
    start=[a, b],
    end=[3*a, 3*b],
    gpts=(32,32)
)
#dir(grid_scan)
#sys.exit()

result = probe.scan(potential, detectors=[detector], scan=grid_scan
).compute();

print(result.shape)
#print(result)
np.save("HAADF-abtem.npy",result.array)
#print(result)
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

(xi,yi),(xf,yf)=grid_scan.limits

extent = (xi,xf,yi,yf)
ax.imshow(result.array.T, cmap="inferno", extent=extent)
ax.set_xlabel("x ($\\AA$)")
ax.set_ylabel("y ($\\AA$)")

plt.savefig("HAADF-abtem.svg")
plt.savefig("HAADF-abtem.png")
