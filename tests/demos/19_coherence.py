import sys,os,time
try:
    import pyslice
except ModuleNotFoundError:
    print("import failed, falling back to relative paths")
    sys.path.insert(0, '../src')
start=time.time()
from pyslice import Probe,Loader,MultisliceCalculator,HAADFData
import numpy as np
import matplotlib.pyplot as plt

run = "TEM"

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}
a,b=2.4907733333333337,2.1570729817355123

# CHANGES SO FAR:
# probe.array is now fixed at summable,positional,x,y indices. previously, it was positional,x,y with positional optional
# calling Probe creates a base_probe
# probe_xs,probe_ys or probe_positions now passable to Probe (instead of this being set up inside MultisliceCalculator)
# Probe.__init__ calls applyShifts (which is like create_batched_probes from before) which expands the positional axis
# additional Probe internal variables Probe.eVs and Probe.wavelengths (plural) are arrays, with length matching the summable axis
# new function addTemporalDecoherence: expands summable axis by creating fresh base_probes of varying eV and varying wavelength
# Propagate is updated to handle varying eV and varying wavelength across multiple types of probes (summable axis)
# new function addSpatialDecoherence: expands summable axis (N*M for N samples spatial and M samples temporal) to defocus each existing probe
# Propagate also collapses probe._array from summable,positions,x,y --> summable*positions,x,y so all the same math, caching, etc, still works
# p.s., i decided to make eV NOT passable as a list. the user should use addTemporalDecoherence. this simplifies the logic, since addTemporalDecoherence is creating fresh probes and would then need to re-call addSpatialDecoherence if it was called previously
# TODO:
# caching excludes decoherence effects (since the calculator cache is calculated based on base_probe's n_probes)

if run == "probes":
	# Generate a few dummy probes
	xs=np.linspace(0,50,501)
	ys=np.linspace(0,49,491)

	probe = Probe(xs,ys,mrad=30,eV=100e3,gaussianVOA=.1,preview=True)
	probe.plot(title="gauss, 30mrad")

	probe = Probe(xs,ys,mrad=30,eV=100e3)

	# temporal decoherence: a range of energies
	eVs = np.linspace(80e3,120e3,25) ; amplitudes = np.exp(-(100e3-eVs)**2/10e3**2)
	#plt.plot(eVs,amplitudes) ; plt.show()
	# manually stack a list of probes' arrays
	probes = [ Probe(xs,ys,mrad=30,eV=eV) for eV in eVs ]
	probe._array = np.mean([ np.absolute(a*p._array) for a,p in zip(amplitudes,probes)],axis=0)[None,:,:,:]
	probe.plot(title="manual stack eV")
	# or, do it automatically:
	probe = Probe(xs,ys,mrad=30,eV=100e3)
	probe.addTemporalDecoherence(10e3,25) ; print(probe.array.shape)
	probe.plot(title="auto stack eV")

	# spatial decoherence: a range of defocuses?
	dZ = np.linspace(-400,400,27) ; amplitudes = np.exp(-(dZ)**2/200**2)
	#plt.plot(dZ,amplitudes) ; plt.show()
	probes = [ Probe(xs,ys,mrad=30,eV=100e3) for i in range(50) ]
	#[ p.aberrate({"C10":d}) for p,d in zip(probes,dZ) ]
	[ p.defocus(d) for p,d in zip(probes,dZ) ]
	probe._array = np.mean([np.absolute(a*p._array) for a,p in zip(amplitudes,probes)],axis=0)
	probe.plot(title="manual stack dZ")
	# or, do it automatically:
	probe = Probe(xs,ys,mrad=30,eV=100e3)
	probe.addSpatialDecoherence(200,10) ; print(probe.array.shape)
	probe.plot(title="auto stack dZ")

	# Or both, and let's check that order doesn't matter
	probe = Probe(xs,ys,mrad=30,eV=100e3)
	probe.addSpatialDecoherence(100,11)
	probe.addTemporalDecoherence(10e3,9)
	probe.plot(title="auto decohere eV and dZ")

if run == "STEM":
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
	# TRIM TO 10x10 UC
	trajectory=trajectory.slice_positions([0,10*a],[0,10*b])
	# SELECT 10 "RANDOM" TIMESTEPS (use seed for reproducibility)
	trajectory=trajectory.get_random_timesteps(3,seed=5)
	# CREATE CALCULATOR OBJECT
	calculator=MultisliceCalculator()
	# SET UP GRID OF HAADF SCAN POINTS
	probe_xs = np.linspace(a,3*a,10)
	probe_ys = np.linspace(b,3*b,9)
	calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys)
	#calculator.base_probe.addTemporalDecoherence(10e3,9)
	calculator.base_probe.addSpatialDecoherence(200,10)
	# RUN MULTISLICE
	exitwaves = calculator.run()
	haadf=HAADFData(exitwaves)
	ary=haadf.calculateADF(preview=False) # use preview=True to view the collection angles of the ADF detector in reciprocal space
	haadf.plot()

if run == "TEM":
	# LOAD TRAJECTORY
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()
	# TRIM TO 10x10 UC
	trajectory=trajectory.slice_positions([0,10*a],[0,10*b])
	# SELECT 10 "RANDOM" TIMESTEPS (use seed for reproducibility)
	trajectory=trajectory.get_random_timesteps(3,seed=5)
	# CREATE CALCULATOR OBJECT
	calculator=MultisliceCalculator()
	calculator.setup(trajectory,aperture=0,voltage_eV=100e3,sampling=.1,slice_thickness=.5)
	#calculator.base_probe.addTemporalDecoherence(30e3,25)
	print(calculator.base_probe.eV)
	#calculator.base_probe.addSpatialDecoherence(100,27)
	# RUN MULTISLICE
	exitwaves = calculator.run()
	print(exitwaves.array.shape)
	#exitwaves.propagate_free_space(10)
	exitwaves.addSpatialDecoherence(10,27)
	print(exitwaves.array.shape)

	exitwaves.plot_realspace()#filename="outputs/figs/04_haadf_cbed.png")

