import sys,os
try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice import Loader,Probe,Propagate,grid_from_trajectory,Potential,MultisliceCalculator,differ

import numpy as np
import matplotlib.pyplot as plt
import os,shutil
from tqdm import tqdm

# PROBE
xs=np.linspace(0,70,701)
ys=np.linspace(0,69,691)
eVs = np.linspace(80,120,50)
weighting = np.exp(-(eVs-100)**2/6**2)
plt.plot(eVs,weighting) ; plt.savefig("outputs/figs/12_aberrations_espread.png")
probes = [ Probe(xs,ys,mrad=30,eV=eV*1e3) for eV in eVs ]
# does it matter whether you aberrate the focused or defocused probe? i don't think so??

# Andy has: r = np.sqrt(x*x + y*y) ; theta = np.arctan2(x,y)
# rand += -r*r*75 + 5*r**4 					# if r^2 n=1 if r^4 n=3, if no cos then m=0
# rand += r*r*2.0*np.cos(2*(theta+0.07))	# r^2 means n=1, cos2 means m=2
# 2.5*np.cos(2*(theta+0.1))“*1*r**4			# r^4 means n=3, cos2 means m=2
# rand +=  -13*np.cos((theta+0.2))*1*r**3	# r^3 means n=2, cos1 means m=1
#  -0*np.cos(3*(theta-0.1))*1*r**3			# r^3 means n=2, cos2 means m=3
# And recall: χ(k,ϕ) = π/2/λ 1/(n+1) C ( k λ )^(n+1) cos(m*(ϕ-ϕa)) for Cnm
dummy = {"C30":1000000, # spherical for caustics
	"C23":(500,np.pi/3)}

#probe.plot()
for probe in probes:
	s=.02 ; l=probe.wavelength
	jellyfish = { "C10":-75*2*s/l**2, "C30":5*4*s/l**4, 
		"C12":(2*2*s/l**2,.07), "C32":(2.5*4*s/l**4,.1), 
		"C21":(-13*3*s/l**3,.2), "C23":(0*3*s/l**3,.1) }

	probe.aberrate(jellyfish)
#probe.plot()



# FOCUS TABLEAU
nx,ny = 7,6
z1=1600 ; zf=2600
dz = (zf-z1)/(nx*ny-1)
#zs = np.linspace(-1000,100,nx*ny) ; dz=zs[1]-zs[0]

plt.rc('axes', titlesize=2)  # Adjusts title font size
plt.rc('axes', titlepad=1)  # Adjusts title font size
plt.rc('axes', labelsize=2)  # Adjusts axis label font size
plt.rc('xtick', labelsize=2) # Adjusts x-tick label font size
plt.rc('ytick', labelsize=2) # Adjusts y-tick label font size
plt.rc('xtick.major',pad=1,size=1)
plt.rc('ytick.major',pad=1,size=1)

fig,axs = plt.subplots(ny,nx,dpi=3000)


ct=0 ; z=0
for j in range(ny):
	for i in tqdm(range(nx)):
		if ct==0:
			for probe in probes:
				probe.defocus(z1)
			z=z1

		else:
			for probe in probes:
				probe.defocus(dz)
			z+=dz
		axs[j,i].set_title("z="+str(np.round(z))+r"$\AA$")
		probe = np.stack([ w*p.array[0,0,:,:] for w,p in zip(weighting,probes) ])
		probe = np.sum(np.abs(probe),axis=0)
		axs[j,i].imshow(probe**2)
		ct+=1

plt.savefig("outputs/figs/12_aberrations_tableau.png")
