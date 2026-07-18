import sys,os
try:
	import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice import Loader,MultisliceCalculator,HAADFData,TACAWData,differ,grid_from_trajectory,Potential

import numpy as np
import matplotlib.pyplot as plt
import os,shutil

#if os.path.exists("psi_data"):
#	shutil.rmtree("psi_data")

dump="inputs/hBN_truncated.lammpstrj"
dt=.005
types={1:"B",2:"N"}
a,b=2.4907733333333337,2.1570729817355123

run = "all"

#run = "reference"
#run = "memmap"
#run = "probeloop1"
#run = "probeloop10"
#run = "mindk"
#run = "mindkloop"
#run = "bigref"
#run = "bigmemmap"
#run = "bigmemloop"
#run = "mongo"
run = "memtacaw"

def clean():
	#if os.path.exists("psi_data"):
	#	shutil.rmtree("psi_data")
	os.system("rm -rf psi_data")

if run in [ "reference", "all" ]: # same as 04_haadf.py, no modifications
	clean() ; print("running reference")
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()				# LOAD TRAJECTORY
	trajectory=trajectory.slice_positions([0,10*a],[0,10*b])					# TRIM TO 10x10 UC
	trajectory=trajectory.get_random_timesteps(2,seed=5)						# SELECT "RANDOM" TIMESTEPS
	for x,y,m in [[2*a,b*4/3,12],[2*a,b*4/3+b,14],[3.5*a,b*4/3,16]]:			# ADD DOPANTS (for testing scan lims)
		dxyz = trajectory.positions[0,:,:]-np.asarray([x,y,0])[None,:]
		distances = np.sqrt(np.sum((dxyz)**2,axis=1))
		i = np.argmin(distances) # which atom is closest to a,b?
		trajectory.atom_types[i] = m
	#positions = trajectory.positions[0]											# PREVIEW POTENTIAL
	#atom_types=trajectory.atom_types
	#xs,ys,zs,lx,ly,lz=grid_from_trajectory(trajectory,sampling=0.1,slice_thickness=0.5)
	#potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
	#potential.plot()
	calculator=MultisliceCalculator()											# CREATE CALCULATOR OBJECT
	probe_xs = np.linspace(10*a-a,10*a-3*a,14) ; probe_ys = np.linspace(10*b-b,10*b-3*b,16)
	calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys)
	exitwaves = calculator.run()												# RUN MULTISLICE
	haadf=HAADFData(exitwaves)													# CALCULATE HAADF
	haadf.calculateADF(preview=False)
	haadf.plot("outputs/figs/21_memorytests_reference.png")


if run in [ "memmap", "all" ]: # same as 04_haadf.py, but using memmap
	clean() ; print("running memmap")
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()				# LOAD TRAJECTORY
	trajectory=trajectory.slice_positions([0,10*a],[0,10*b])					# TRIM TO 10x10 UC
	trajectory=trajectory.get_random_timesteps(2,seed=5)						# SELECT "RANDOM" TIMESTEPS
	for x,y,m in [[2*a,b*4/3,12],[2*a,b*4/3+b,14],[3.5*a,b*4/3,16]]:			# ADD DOPANTS (for testing scan lims)
		dxyz = trajectory.positions[0,:,:]-np.asarray([x,y,0])[None,:]
		distances = np.sqrt(np.sum((dxyz)**2,axis=1))
		i = np.argmin(distances) # which atom is closest to a,b?
		trajectory.atom_types[i] = m
	calculator=MultisliceCalculator()											# CREATE CALCULATOR OBJECT
	probe_xs = np.linspace(10*a-a,10*a-3*a,14) ; probe_ys = np.linspace(10*b-b,10*b-3*b,16)
	calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,use_memmap=True)
	exitwaves = calculator.run()												# RUN MULTISLICE
	haadf=HAADFData(exitwaves)													# CALCULATE HAADF
	haadf.calculateADF(preview=False)
	haadf.plot("outputs/figs/21_memorytests_memmap.png")


if run in [ "probeloop1", "all" ]: # same as 04_haadf.py, but with chunked looped probes
	clean() ; print("running probeloop1")
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()				# LOAD TRAJECTORY
	trajectory=trajectory.slice_positions([0,10*a],[0,10*b])					# TRIM TO 10x10 UC
	trajectory=trajectory.get_random_timesteps(2,seed=5)						# SELECT "RANDOM" TIMESTEPS
	for x,y,m in [[2*a,b*4/3,12],[2*a,b*4/3+b,14],[3.5*a,b*4/3,16]]:			# ADD DOPANTS (for testing scan lims)
		dxyz = trajectory.positions[0,:,:]-np.asarray([x,y,0])[None,:]
		distances = np.sqrt(np.sum((dxyz)**2,axis=1))
		i = np.argmin(distances) # which atom is closest to a,b?
		trajectory.atom_types[i] = m
	calculator=MultisliceCalculator()											# CREATE CALCULATOR OBJECT
	probe_xs = np.linspace(10*a-a,10*a-3*a,14) ; probe_ys = np.linspace(10*b-b,10*b-3*b,16)
	calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,loop_probes=True)
	exitwaves = calculator.run()												# RUN MULTISLICE
	haadf=HAADFData(exitwaves)													# CALCULATE HAADF
	haadf.calculateADF(preview=False)
	haadf.plot("outputs/figs/21_memorytests_probeloop1.png")


if run in [ "probeloop10", "all" ]: # same as 04_haadf.py, but with chunked looped probes
	clean() ; print("running probeloop10")
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()				# LOAD TRAJECTORY
	trajectory=trajectory.slice_positions([0,10*a],[0,10*b])					# TRIM TO 10x10 UC
	trajectory=trajectory.get_random_timesteps(2,seed=5)						# SELECT "RANDOM" TIMESTEPS
	for x,y,m in [[2*a,b*4/3,12],[2*a,b*4/3+b,14],[3.5*a,b*4/3,16]]:			# ADD DOPANTS (for testing scan lims)
		dxyz = trajectory.positions[0,:,:]-np.asarray([x,y,0])[None,:]
		distances = np.sqrt(np.sum((dxyz)**2,axis=1))
		i = np.argmin(distances) # which atom is closest to a,b?
		trajectory.atom_types[i] = m
	calculator=MultisliceCalculator()											# CREATE CALCULATOR OBJECT
	probe_xs = np.linspace(10*a-a,10*a-3*a,14) ; probe_ys = np.linspace(10*b-b,10*b-3*b,16)
	calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,loop_probes=10)
	exitwaves = calculator.run()												# RUN MULTISLICE
	haadf=HAADFData(exitwaves)													# CALCULATE HAADF
	haadf.calculateADF(preview=False)
	haadf.plot("outputs/figs/21_memorytests_probeloop10.png")


if run in [ "mindk", "all" ]: # same as 04_haadf.py, but with chunked looped probes
	clean() ; print("running mindk")
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()				# LOAD TRAJECTORY
	trajectory=trajectory.slice_positions([0,10*a],[0,10*b])					# TRIM TO 10x10 UC
	trajectory=trajectory.get_random_timesteps(2,seed=5)						# SELECT "RANDOM" TIMESTEPS
	for x,y,m in [[2*a,b*4/3,12],[2*a,b*4/3+b,14],[3.5*a,b*4/3,16]]:			# ADD DOPANTS (for testing scan lims)
		dxyz = trajectory.positions[0,:,:]-np.asarray([x,y,0])[None,:]
		distances = np.sqrt(np.sum((dxyz)**2,axis=1))
		i = np.argmin(distances) # which atom is closest to a,b?
		trajectory.atom_types[i] = m
	calculator=MultisliceCalculator()											# CREATE CALCULATOR OBJECT
	probe_xs = np.linspace(10*a-a,10*a-3*a,32) ; probe_ys = np.linspace(10*b-b,10*b-3*b,32)
	calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,min_dk=0.1)
	exitwaves = calculator.run()												# RUN MULTISLICE
	haadf=HAADFData(exitwaves)													# CALCULATE HAADF
	haadf.calculateADF(preview=False)
	haadf.plot("outputs/figs/21_memorytests_mindk.png")


if run in [ "mindkloop", "all" ]: # same as 04_haadf.py, but with chunked looped probes
	clean() ; print("running mindkloop")
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()				# LOAD TRAJECTORY
	trajectory=trajectory.slice_positions([0,10*a],[0,10*b])					# TRIM TO 10x10 UC
	trajectory=trajectory.get_random_timesteps(2,seed=5)						# SELECT "RANDOM" TIMESTEPS
	for x,y,m in [[2*a,b*4/3,12],[2*a,b*4/3+b,14],[3.5*a,b*4/3,16]]:			# ADD DOPANTS (for testing scan lims)
		dxyz = trajectory.positions[0,:,:]-np.asarray([x,y,0])[None,:]
		distances = np.sqrt(np.sum((dxyz)**2,axis=1))
		i = np.argmin(distances) # which atom is closest to a,b?
		trajectory.atom_types[i] = m
	calculator=MultisliceCalculator()											# CREATE CALCULATOR OBJECT
	probe_xs = np.linspace(10*a-a,10*a-3*a,67) ; probe_ys = np.linspace(10*b-b,10*b-3*b,69)
	calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,min_dk=0.1,loop_probes=50)
	exitwaves = calculator.run()												# RUN MULTISLICE
	haadf=HAADFData(exitwaves)													# CALCULATE HAADF
	haadf.calculateADF(preview=False)
	haadf.plot("outputs/figs/21_memorytests_mindkloop.png")


# exclude bigref from "all" since it will likely OOM (that's kinda the point)
if run in [ "bigref" ]: # same as 04_haadf.py, but bigger FOV. immediate OOM-kill on multislice (250x216 kpts, 50x50 probe positions, probecube is 250*216*50*50*128/8/1024^3=2GB, frame_data is the same, wavefunction_data is 3x, calculator's intermediate variable exit_waves_k is 2GB too, and Propagate has intermediate variables too)
	clean() ; print("running bigref")
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()				# LOAD TRAJECTORY
	trajectory=trajectory.slice_positions([0,10*a],[0,10*b])					# TRIM TO 10x10 UC
	trajectory=trajectory.get_random_timesteps(2,seed=5)						# SELECT "RANDOM" TIMESTEPS
	for x,y,m in [[2*a,b*4/3,12],[2*a,b*4/3+b,14],[3.5*a,b*4/3,16]]:			# ADD DOPANTS (for testing scan lims)
		dxyz = trajectory.positions[0,:,:]-np.asarray([x,y,0])[None,:]
		distances = np.sqrt(np.sum((dxyz)**2,axis=1))
		i = np.argmin(distances) # which atom is closest to a,b?
		trajectory.atom_types[i] = m
	positions = trajectory.positions[0]											# PREVIEW POTENTIAL
	atom_types=trajectory.atom_types
	xs,ys,zs,lx,ly,lz=grid_from_trajectory(trajectory,sampling=0.1,slice_thickness=0.5)
	potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
	potential.plot()
	calculator=MultisliceCalculator()											# CREATE CALCULATOR OBJECT
	probe_xs = np.linspace(a,4*a,32) ; probe_ys = np.linspace(b,4*b,32)
	calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys)
	exitwaves = calculator.run()												# RUN MULTISLICE
	haadf=HAADFData(exitwaves)													# CALCULATE HAADF
	haadf.calculateADF(preview=False)
	haadf.plot("outputs/figs/21_memorytests_bigref.png")


# (250x216 kpts, 50x50 probe positions, probecube is 250*216*50*50*128/8/1024^3=2GB, frame_data is the same, wavefunction_data is 3x, calculator's intermediate variable exit_waves_k is 2GB too, and Propagate has intermediate variables too)
# memmapping should at least remove wavefunction_data from memory, tops out at 18GB during first frame multislice, OOM-kill after. dropping to 40x40 runs
if run in [ "bigmemmap", "all" ]:
	clean() ; print("running bigmemmap")
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()				# LOAD TRAJECTORY
	trajectory=trajectory.slice_positions([0,10*a],[0,10*b])					# TRIM TO 10x10 UC
	trajectory=trajectory.get_random_timesteps(2,seed=5)						# SELECT "RANDOM" TIMESTEPS
	for x,y,m in [[2*a,b*4/3,12],[2*a,b*4/3+b,14],[3.5*a,b*4/3,16]]:			# ADD DOPANTS (for testing scan lims)
		dxyz = trajectory.positions[0,:,:]-np.asarray([x,y,0])[None,:]
		distances = np.sqrt(np.sum((dxyz)**2,axis=1))
		i = np.argmin(distances) # which atom is closest to a,b?
		trajectory.atom_types[i] = m
	#positions = trajectory.positions[0]											# PREVIEW POTENTIAL
	#atom_types=trajectory.atom_types
	#xs,ys,zs,lx,ly,lz=grid_from_trajectory(trajectory,sampling=0.1,slice_thickness=0.5)
	#potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
	#potential.plot()
	calculator=MultisliceCalculator()											# CREATE CALCULATOR OBJECT
	probe_xs = np.linspace(a,4*a,35) ; probe_ys = np.linspace(b,4*b,35)
	calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,use_memmap=True)
	exitwaves = calculator.run()												# RUN MULTISLICE
	haadf=HAADFData(exitwaves)													# CALCULATE HAADF
	haadf.calculateADF(preview=False)
	haadf.plot("outputs/figs/21_memorytests_bigmemmap.png")


# (250x216 kpts, 50x50 probe positions, probecube is 250*216*50*50*128/8/1024^3=2GB, frame_data is the same, wavefunction_data is 3x, calculator's intermediate variable exit_waves_k is 2GB too, and Propagate has intermediate variables too)
# memmapping should at least remove wavefunction_data from memory, probe_loop removes the probe cube plus however many intermediate variables are of the same size. 50x50 now runs in 2.7 GB ram
if run in [ "bigmemloop", "all" ]:
	clean() ; print("running bigmemloop")
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()				# LOAD TRAJECTORY
	trajectory=trajectory.slice_positions([0,10*a],[0,10*b])					# TRIM TO 10x10 UC
	trajectory=trajectory.get_random_timesteps(2,seed=5)						# SELECT "RANDOM" TIMESTEPS
	for x,y,m in [[2*a,b*4/3,12],[2*a,b*4/3+b,14],[3.5*a,b*4/3,16]]:			# ADD DOPANTS (for testing scan lims)
		dxyz = trajectory.positions[0,:,:]-np.asarray([x,y,0])[None,:]
		distances = np.sqrt(np.sum((dxyz)**2,axis=1))
		i = np.argmin(distances) # which atom is closest to a,b?
		trajectory.atom_types[i] = m
	#positions = trajectory.positions[0]											# PREVIEW POTENTIAL
	#atom_types=trajectory.atom_types
	#xs,ys,zs,lx,ly,lz=grid_from_trajectory(trajectory,sampling=0.1,slice_thickness=0.5)
	#potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
	#potential.plot()
	calculator=MultisliceCalculator()											# CREATE CALCULATOR OBJECT
	probe_xs = np.linspace(a,4*a,50) ; probe_ys = np.linspace(b,4*b,50)
	calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,use_memmap=True,loop_probes=100)
	exitwaves = calculator.run()												# RUN MULTISLICE
	haadf=HAADFData(exitwaves)													# CALCULATE HAADF
	haadf.calculateADF(preview=False)
	haadf.plot("outputs/figs/21_memorytests_bigmemloop.png")


# STRESS TEST: 1000x1000 probe positions (would be a 160 GB probe cube alone!) on uncropped trajectory (huge potential!). chunking is required to avoid the probe loop, memmaping is a good idea, and we're introducing autocropping to propagate a cropped proba through a cropped potential, which also means we limit our number of k-points. min_dk = 0.1 iA for a 0.1 A sampling means nkx,nky are 100x100 even for the full uncropped system.
# OPE: frame_data is 41 GB (100 kx 100 ky 512 x 512 y)
# do not include mongo in all
if run in [ "mongo" ]:
	clean() ; print("running mongo")
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()				# LOAD TRAJECTORY
	#trajectory=trajectory.slice_positions([0,10*a],[0,10*b])					# TRIM TO 10x10 UC
	trajectory=trajectory.get_random_timesteps(2,seed=5)						# SELECT "RANDOM" TIMESTEPS
	for x,y,m in [[2*a,b*4/3,12],[2*a,b*4/3+b,14],[3.5*a,b*4/3,16]]:			# ADD DOPANTS (for testing scan lims)
		dxyz = trajectory.positions[0,:,:]-np.asarray([x,y,0])[None,:]
		distances = np.sqrt(np.sum((dxyz)**2,axis=1))
		i = np.argmin(distances) # which atom is closest to a,b?
		trajectory.atom_types[i] = m
	#positions = trajectory.positions[0]											# PREVIEW POTENTIAL
	#atom_types=trajectory.atom_types
	#xs,ys,zs,lx,ly,lz=grid_from_trajectory(trajectory,sampling=0.1,slice_thickness=0.5)
	#potential = Potential(xs, ys, zs, positions, atom_types, kind="kirkland")
	#potential.plot()
	calculator=MultisliceCalculator()											# CREATE CALCULATOR OBJECT
	probe_xs = np.linspace(a,10*a,512) ; probe_ys = np.linspace(b,10*b,512)
	calculator.setup(trajectory,aperture=30,voltage_eV=100e3,sampling=.1,slice_thickness=.5,probe_xs=probe_xs,probe_ys=probe_ys,use_memmap=True,loop_probes=100,min_dk=0.1)
	exitwaves = calculator.run()												# RUN MULTISLICE
	haadf=HAADFData(exitwaves)													# CALCULATE HAADF
	haadf.calculateADF(preview=False)
	haadf.plot("outputs/figs/21_memorytests_mongo.png")

if run in [ "memtacaw", "all" ]:
	clean() ; print("running memtacaw")
	trajectory=Loader(dump,timestep=dt,atom_mapping=types).load()				# LOAD TRAJECTORY
	calculator=MultisliceCalculator()											# TACAW CALCULATION: ALL TIMESTEPS, PARALLEL BEAM
	calculator.setup(trajectory,aperture=0,voltage_eV=100e3,sampling=.1,slice_thickness=.5,use_memmap=True)
	exitwaves = calculator.run()
	tacaw = TACAWData(exitwaves)												# CALCULATE TACAW, TEMPORAL FFT
	Z = tacaw.spectral_diffraction(30) ; print(Z.shape)
	tacaw.plot(Z**.1,"kx","ky",filename="outputs/figs/21_memtacaw.png")			# OR PLOT USING BUILT IN TOOLS: AN ENERGY SLICE:

