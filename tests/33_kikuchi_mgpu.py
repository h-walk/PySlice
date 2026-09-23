import multiprocessing as mp
import os,pickle,sys,tempfile
from pathlib import Path

import numpy as np
from ase.io import read,write
sys.path.insert(1,"../src")
sys.path.insert(1,"/emmadrive/Shared_Code/PySlice/PySlice-20260923/src")
from pyslice import Loader,MultisliceCalculator,Trajectory,fetch_cif,to_numpy

# SETTINGS
n_probes = 10
n_configs = 10 ; sigma = .2
n_tilts = 50
workers_per_gpu = 3
diameter = 100.0
Area = "auto"
mrad = 30
voltage_eV = 100e3
dk = .025
cache_wavefunctions = False
dxy = .2 ; dz = .5

_trajectory = None
_area = None
_device = None

def spherical_to_tilts(theta,phi):
	"""Convert spherical beam angles to sequential PySlice tilts."""
	x,y,z = np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)
	return np.arctan2(y,z),np.arcsin(np.clip(x,-1,1))

def simulation_area():
	"""Return the configured lateral simulation area.

	Returns
	-------
	numpy.ndarray
		The x and y dimensions in Angstroms.
	"""
	if Area != "auto":
		return np.asarray(Area,dtype=float)
	energy_J = voltage_eV*1.602176634e-19
	wavelength = 6.62607015e-34*299792458/np.sqrt(
		energy_J**2+2*energy_J*9.1093837e-31*299792458**2)*1e10
	probe_diameter = 0 if mrad == 0 else .8224615*wavelength/(mrad*1e-3)
	return np.asarray([max(probe_diameter,1/dk)]*2)

def prepare_source_trajectory(dump,source_diameter):
	"""Load, tile, and spherical-trim the common source structure.

	Parameters
	----------
	dump : path-like
		CIF or cached structure input accepted by ``Loader``.
	source_diameter : float
		Diameter of the spherical source structure in Angstroms.

	Returns
	-------
	Trajectory
		Prepared source shared by all orientations.
	"""
	trajectory = Loader(dump).load()
	cube = np.linalg.norm(trajectory.box_matrix,axis=1)
	trajectory = trajectory.tile_positions([int(np.ceil(source_diameter/length)) for length in cube])
	center = np.sum(trajectory.box_matrix,axis=0)/2
	mask = np.linalg.norm(trajectory.positions[0]-center,axis=1) <= source_diameter/2
	return Trajectory(trajectory.atom_types[mask],trajectory.positions[:,mask]-center+source_diameter/2,
		trajectory.velocities[:,mask],np.eye(3)*source_diameter,trajectory.timestep)

def pristine_orientation(trajectory,area,alpha,beta):
	"""Build the pristine cropped structure for one PySlice tilt.

	Parameters
	----------
	trajectory : Trajectory
		Prepared spherical source structure.
	area : sequence
		Lateral x and y dimensions in Angstroms.
	alpha, beta : float
		Sequential PySlice tilt angles in radians.

	Returns
	-------
	Trajectory
		Oriented and laterally cropped pristine structure.
	"""
	traj = trajectory.tilt_positions(alpha,beta)
	p = traj.positions[0]
	center = (p.min(axis=0)+p.max(axis=0))/2
	keep = (np.abs(p[:,0]-center[0]) <= area[0]/2) & (np.abs(p[:,1]-center[1]) <= area[1]/2)
	z0,z1 = p[:,2].min(),p[:,2].max()
	shift = np.array([center[0]-area[0]/2,center[1]-area[1]/2,z0])
	return Trajectory(traj.atom_types[keep].copy(),traj.positions[:,keep]-shift,traj.velocities[:,keep].copy(),
		np.diag([area[0],area[1],z1-z0]),traj.timestep)

def initialize_worker(trajectory,area,device,n_threads):
	"""Bind one spawned worker to one CUDA device.

	Parameters
	----------
	trajectory : Trajectory
		Prepared spherical source structure.
	area : sequence
		Lateral simulation dimensions in Angstroms.
	device : str
		CUDA device owned by this worker.
	n_threads : int
		CPU threads assigned to this worker.
	"""
	global _trajectory,_area,_device
	_trajectory,_area = trajectory,area
	_device = device
	os.environ.pop("PYSLICE_DEVICE",None)
	import torch
	torch.cuda.set_device(_device)
	torch.set_num_threads(n_threads)
	torch.set_num_interop_threads(1)
	print(f"worker {os.getpid()} bound to {_device}",flush=True)

def run_worker(device,indexed_tasks,trajectory,area,n_threads,result_path):
	"""Run one fixed task shard on one explicitly assigned CUDA device.

	Parameters
	----------
	device : str
		CUDA device assigned to this process.
	indexed_tasks : sequence
		Original task indices paired with orientation arguments.
	trajectory : Trajectory
		Prepared spherical source structure.
	area : sequence
		Lateral simulation dimensions in Angstroms.
	n_threads : int
		CPU threads assigned to this process.
	result_path : pathlib.Path
		Temporary pickle receiving indexed results.
	"""
	initialize_worker(trajectory,area,device,n_threads)
	results = [(index,simulate_tilt(task)) for index,task in indexed_tasks]
	with open(result_path,"wb") as f:
		pickle.dump(results,f)
	print(f"worker {os.getpid()} completed {len(results)} orientations",flush=True)

def simulate_tilt(args):
	"""Simulate one orientation on the CUDA device owned by this worker."""
	theta,phi,alpha,beta,grid_position,probe_positions = args
	pristine = pristine_orientation(_trajectory,_area,alpha,beta)
	traj = pristine.generate_random_displacements(n_configs,sigma,seed=0)
	calculator = MultisliceCalculator(device=_device)
	calculator.setup(traj,aperture=mrad,voltage_eV=voltage_eV,probe_positions=probe_positions,
		cache_wavefunctions=cache_wavefunctions,sampling=dxy,slice_thickness=dz)
	exitwaves = calculator.run()
	structure_path = exitwaves.cache_dir/"structure.xyz"
	write(structure_path,pristine.to_ase(),format="extxyz")
	pattern = np.mean(np.abs(to_numpy(exitwaves.array[:,:,:,:,-1]))**2,axis=(0,1)).T**.5
	extent = (exitwaves.kxs[0],exitwaves.kxs[-1],exitwaves.kys[0],exitwaves.kys[-1])
	return (theta,phi),(alpha,beta),grid_position,pattern,extent,structure_path

def display_results(mp_id,theta_max,phi_max,results,structure_fallback=None):
	"""Display orientation, diffraction, and structure panels interactively."""
	import tkinter as tk
	from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
	from matplotlib.figure import Figure
	angles = np.asarray([result[0] for result in results])
	tilts = np.asarray([result[1] for result in results])
	grid_positions = np.asarray([result[2] for result in results])
	patterns = [result[3] for result in results]
	extent = results[0][4]
	structure_paths = [result[5] for result in results]
	root = tk.Tk()
	root.title(f"Kikuchi patterns: {mp_id}")
	fig = Figure(figsize=(18,6))
	grid = fig.add_gridspec(1,3,width_ratios=(1,1,1.6))
	ax_tilts = fig.add_subplot(grid[0,0])
	ax_pattern = fig.add_subplot(grid[0,1])
	ax_structure = fig.add_subplot(grid[0,2],projection="3d")
	ax_structure.mouse_init()
	ax_tilts.fill([.5,0,1],[1,0,0],color="C0",alpha=.25)
	ax_tilts.scatter(grid_positions[:,0],grid_positions[:,1],s=8)
	corner_angles = [(0,0),(theta_max,0),(theta_max,phi_max)]
	corner_specs = [((.5,1),(0,-4),"center","top"),((0,0),(4,4),"left","bottom"),((1,0),(-4,4),"right","bottom")]
	for (xy,offset,ha,va),(theta,phi) in zip(corner_specs,corner_angles):
		alpha,beta = spherical_to_tilts(theta,phi)
		phi_label = "arbitrary" if theta == 0 else f"{np.rad2deg(phi):.1f}°"
		label = f"θ={np.rad2deg(theta):.1f}°, φ={phi_label}\nα={np.rad2deg(alpha):.1f}°, β={np.rad2deg(beta):.1f}°"
		ax_tilts.annotate(label,xy,xytext=offset,textcoords="offset points",fontsize=8,ha=ha,va=va)
	ax_tilts.set(xlabel=f"orientation between φ=0° and φ={np.rad2deg(phi_max):.1f}° edges",
		ylabel="weight toward θ=0° ([001])",title="Click a simulated orientation")
	image = ax_pattern.imshow(patterns[0]**.1,cmap="inferno",extent=extent,origin="lower")
	ax_pattern.set(xlabel="kx (1/Å)",ylabel="ky (1/Å)",title="θ=0.0°, φ=arbitrary; α=0.0°, β=0.0°")
	canvas = FigureCanvasTkAgg(fig,master=root)
	toolbar = NavigationToolbar2Tk(canvas,root,pack_toolbar=False)
	toolbar.update()
	toolbar.pack(side=tk.BOTTOM,fill=tk.X)
	canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=True)

	def display_structure(path,index):
		"""Display a cached structure or reconstruct it from the source CIF."""
		ax_structure.clear()
		reconstructed = False
		if Path(path).is_file():
			atoms = read(path)
		elif structure_fallback is not None:
			atoms = structure_fallback(index)
			reconstructed = atoms is not None
		else:
			atoms = None
		if atoms is None:
			ax_structure.text2D(.5,.5,"Structure unavailable\nPlace the original CIF beside the NPZ file",
				transform=ax_structure.transAxes,ha="center",va="center")
			ax_structure.set_title("Structure XYZ and source CIF unavailable")
			return
		positions = atoms.get_positions()
		ax_structure.scatter(*positions.T,s=3,c=atoms.get_atomic_numbers(),cmap="tab20",depthshade=True)
		ax_structure.set(xlabel="x (Å)",ylabel="y (Å)",zlabel="z (Å)")
		ax_structure.set_box_aspect(np.maximum(np.ptp(positions,axis=0),1),zoom=1.35)
		ax_structure.view_init(elev=90,azim=-90)
		ax_structure.set_proj_type("ortho")
		source = "reconstructed from CIF" if reconstructed else "saved XYZ"
		ax_structure.set_title(f"Structure ({source}): beam view down z\nleft drag: rotate; toolbar: pan/zoom")

	def select_tilt(event):
		"""Display the simulated orientation nearest a click."""
		if event.inaxes is not ax_tilts:
			return
		k = np.argmin(np.sum((grid_positions-[event.xdata,event.ydata])**2,axis=1))
		image.set_data(patterns[k]**.1)
		theta,phi = np.rad2deg(angles[k]); alpha,beta = np.rad2deg(tilts[k])
		ax_pattern.set_title(f"θ={theta:.1f}°, φ={phi:.1f}°; α={alpha:.1f}°, β={beta:.1f}°")
		display_structure(structure_paths[k],k)
		canvas.draw_idle()

	canvas.mpl_connect("button_press_event",select_tilt)
	display_structure(structure_paths[0],0)
	canvas.draw()
	root.mainloop()

def view_npz(result_path):
	"""Open a saved Kikuchi result archive in the interactive viewer.

	Parameters
	----------
	result_path : path-like
		NPZ archive written by this script.
	"""
	result_path = Path(result_path).resolve()
	with np.load(result_path) as data:
		angles = np.asarray(data["angles"])
		tilts = np.asarray(data["tilts"])
		grid_positions = np.asarray(data["grid_positions"])
		patterns = np.asarray(data["patterns"])
		extent = np.asarray(data["extent"])
		stored_paths = [Path(str(path)) for path in data["structure_paths"]]
		theta_max = float(data["theta_max"]) if "theta_max" in data else float(angles[:,0].max())
		phi_max = float(data["phi_max"]) if "phi_max" in data else float(angles[:,1].max())
		saved_area = np.asarray(data["area"]) if "area" in data else simulation_area()
		saved_diameter = float(data["diameter"]) if "diameter" in data else diameter
		source_cif = str(data["source_cif"].item()) if "source_cif" in data else ""
	structure_paths = []
	for stored_path in stored_paths:
		local_path = result_path.parent/"psi_data"/stored_path.parent.name/"structure.xyz"
		structure_paths.append(stored_path if stored_path.is_file() else local_path)
	results = [(angles[i],tilts[i],grid_positions[i],patterns[i],extent,structure_paths[i])
		for i in range(len(angles))]
	mp_id = result_path.stem.removeprefix("kikuchi_").removesuffix("_results")
	cif_candidates = [result_path.parent/source_cif,result_path.parent/f"mp_{mp_id}.cif"]
	cif_path = next((path for path in cif_candidates if path.name and path.is_file()),None)
	local_cifs = sorted(result_path.parent.glob("*.cif"))
	if cif_path is None and len(local_cifs) == 1:
		cif_path = local_cifs[0]
	prepared = None
	last_index,last_atoms = None,None

	def reconstruct_structure(index):
		"""Reconstruct one missing oriented structure lazily.

		Parameters
		----------
		index : int
			Orientation index in the saved result.

		Returns
		-------
		ase.Atoms or None
			Reconstructed structure, or ``None`` when no source CIF is available.
		"""
		nonlocal prepared,last_index,last_atoms
		if cif_path is None:
			return None
		if index == last_index:
			return last_atoms
		if prepared is None:
			prepared = prepare_source_trajectory(cif_path,saved_diameter)
		alpha,beta = tilts[index]
		last_atoms = pristine_orientation(prepared,saved_area,alpha,beta).to_ase()
		last_index = index
		return last_atoms

	display_results(mp_id,theta_max,phi_max,results,reconstruct_structure)

def main():
	"""Prepare the orientation grid, distribute it over GPUs, and report results."""
	global n_tilts,workers_per_gpu
	if len(sys.argv) == 3 and sys.argv[1] == "--view":
		view_npz(sys.argv[2])
		return
	if len(sys.argv) < 2:
		raise SystemExit("Usage: python 33_kikuchi_mgpu.py MP_ID [theta=90] [phi=90] [tilts=30] "
			"[workers_per_gpu=1] [--no-gui]\n       python 33_kikuchi_mgpu.py --view RESULTS.npz")
	mp_id = sys.argv[1]
	theta_max,phi_max = 90.,90.
	for arg in sys.argv[2:]:
		if arg.startswith("theta="): theta_max = float(arg.split("=",1)[1])
		if arg.startswith("phi="): phi_max = float(arg.split("=",1)[1])
		if arg.startswith("tilts="): n_tilts = int(arg.split("=",1)[1])
		if arg.startswith("workers_per_gpu="): workers_per_gpu = int(arg.split("=",1)[1])
	if workers_per_gpu < 1:
		raise ValueError("workers_per_gpu must be at least 1.")
	theta_max,phi_max = np.deg2rad([theta_max,phi_max])
	print("fetch cif",flush=True)
	cif = Path(f"mp_{mp_id}.cif")
	dump = cif if cif.exists() else fetch_cif("mp",mp_id)
	print("tile and spherical trim",flush=True)
	trajectory = prepare_source_trajectory(dump,diameter)
	area = simulation_area()
	corners = np.asarray([[0,0,1],[np.sin(theta_max),0,np.cos(theta_max)],
		[np.sin(theta_max)*np.cos(phi_max),np.sin(theta_max)*np.sin(phi_max),np.cos(theta_max)]])
	rng = np.random.default_rng(0)
	tasks = []
	for i in range(n_tilts):
		for j in range(n_tilts-i):
			weights = np.asarray([n_tilts-1-i-j,i,j])/(n_tilts-1)
			direction = weights@corners
			direction /= np.linalg.norm(direction)
			theta = np.arccos(np.clip(direction[2],-1,1))
			phi = 0 if theta < 1e-12 else np.mod(np.arctan2(direction[1],direction[0]),2*np.pi)
			alpha,beta = spherical_to_tilts(theta,phi)
			grid_position = (weights[2]+weights[0]/2,weights[0])
			tasks.append((theta,phi,alpha,beta,grid_position,rng.random((n_probes,2))*area))
	import torch
	if not torch.cuda.is_available():
		raise RuntimeError("Multi-GPU Kikuchi search requires CUDA-enabled PyTorch and visible GPUs.")
	n_gpus = torch.cuda.device_count()
	n_workers = min(n_gpus*workers_per_gpu,len(tasks))
	devices = [f"cuda:{i % n_gpus}" for i in range(n_workers)]
	ctx = mp.get_context("spawn")
	n_cpus = len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else (os.cpu_count() or 1)
	n_threads = max(1,n_cpus//n_workers)
	print(f"dispatching {len(tasks)} orientations to {n_workers} workers on {n_gpus} GPUs "
		f"({workers_per_gpu} requested workers/GPU): {devices}",flush=True)
	indexed_tasks = list(enumerate(tasks))
	with tempfile.TemporaryDirectory(prefix="pyslice-mgpu-") as d:
		paths = [Path(d)/f"worker_{i}.pkl" for i in range(n_workers)]
		processes = [ctx.Process(target=run_worker,args=(devices[i],indexed_tasks[i::n_workers],
			trajectory,area,n_threads,paths[i])) for i in range(n_workers)]
		for process in processes: process.start()
		for process in processes: process.join()
		failures = [(process.pid,process.exitcode) for process in processes if process.exitcode]
		if failures:
			raise RuntimeError(f"CUDA worker failures (pid, exit code): {failures}")
		indexed_results = []
		for path in paths:
			with open(path,"rb") as f: indexed_results.extend(pickle.load(f))
	results = [result for _,result in sorted(indexed_results)]
	angles = np.asarray([result[0] for result in results])
	tilts = np.asarray([result[1] for result in results])
	grid_positions = np.asarray([result[2] for result in results])
	patterns = np.asarray([result[3] for result in results])
	extent = np.asarray(results[0][4])
	structure_paths = np.asarray([str(result[5]) for result in results])
	result_path = Path(f"kikuchi_{mp_id}_results.npz")
	np.savez(result_path,angles=angles,tilts=tilts,grid_positions=grid_positions,patterns=patterns,
		extent=extent,structure_paths=structure_paths,theta_max=theta_max,phi_max=phi_max,
		area=area,diameter=diameter,source_cif=cif.name)
	print(f"saved {len(results)} orientations to {result_path}",flush=True)
	if "--no-gui" not in sys.argv and os.environ.get("DISPLAY"):
		display_results(mp_id,theta_max,phi_max,results)

if __name__ == "__main__":
	main()
