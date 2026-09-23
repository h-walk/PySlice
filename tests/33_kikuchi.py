import sys,os,multiprocessing as mp
import tkinter as tk
import numpy as np
from ase.io import read,write
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib.figure import Figure
sys.path.insert(1,"../src")
from pyslice import Loader, MultisliceCalculator, Trajectory, fetch_cif, to_numpy

# SETTINGS
n_probes = 3
n_configs = 3 ; sigma = .2
n_tilts = 30
n_workers = 2
diameter = 100.0 	# Angstrom
Area = "auto"		# "auto" to infer from probe 1/e^2 diameter, OR, a tuple with units of angstrom
mrad = 30			# convergence semi-angle
voltage_eV = 60e3
dk = .025			# maximum reciprocal-space pixel spacing in 1/Angstrom
cache_wavefunctions = True

dxy = .2 ; dz = .5	# can we be sloppy with these and still get tasty kikuchi bands?

# COMMAND LINE ARGS
mp_id = sys.argv[1]
theta_max,phi_max = 90,90
for arg in sys.argv:
	if "theta=" in arg:
		theta_max = float(arg.split("=")[-1])
	if "phi=" in arg:
		phi_max = float(arg.split("=")[-1])
theta_max *= np.pi/180 ; phi_max *= np.pi/180

# FETCH AND LOAD CIF FROM MATERIALS PROJECT
print("fetch cif")
if os.path.exists("mp_"+mp_id+".cif"):
	dump = "mp_"+mp_id+".cif"
else:
	dump = fetch_cif("mp", mp_id)
trajectory = Loader(dump).load()

# TILE OUT TO FILL CUBE OF SIZE LENGTH "diameter"
print("tile")
cube = np.linalg.norm(trajectory.box_matrix,axis=1)
n_cells = [int(np.ceil(diameter/cube[i])) for i in range(3)]
trajectory = trajectory.tile_positions(n_cells)

# DELETE ATOMS OUTSIDE OF SPHERE
print("spherical trim")
center = np.sum(trajectory.box_matrix,axis=0)/2
distances = np.sum((trajectory.positions-center[None,None,:])**2,axis=2)**.5
mask = distances[0] <= diameter/2
trajectory = Trajectory(trajectory.atom_types[mask],trajectory.positions[:,mask]-center+diameter/2,
	trajectory.velocities[:,mask],np.eye(3)*diameter,trajectory.timestep)

# INFER AREA FROM PROBE: WEIRD BUG: We previous calculated area by constructing a Probe object and then measuring the 1/e^2 radius. This worked, and worked when using numpy only, but hangs indefinitely when using multiprocessing and torch (dead lock on forked subprocesses, all aware of the lock on the main thread's torch object)
if Area == "auto":
	#xs = np.linspace(0,100,int(100/.05)+1)
	#probe = Probe(xs, xs, mrad, voltage_eV)
	#ary = np.absolute(to_numpy(probe.array[0,0]))**2
	#probe_radius = np.max(np.abs(xs[ary[:,len(xs)//2] >= ary.max()/np.e**2]-xs[len(xs)//2]))
	#Area = [max(probe_radius*2,1/dk)]*2
	#A = [probe_radius*2]*2
	energy_J = voltage_eV*1.602176634e-19
	wavelength = 6.62607015e-34*299792458/np.sqrt(energy_J**2+2*energy_J*9.1093837e-31*299792458**2)*1e10
	probe_diameter = 0 if mrad == 0 else .8224615*wavelength/(mrad*1e-3)
	Area = [max(probe_diameter,1/dk)]*2

# CONVERT TILT SPHERICAL COORDINATES (https://en.wikipedia.org/wiki/Spherical_coordinate_system) TO TILT ANGLES. THETA IS ANGLE FROM VERTICAL, PHI IS ANGLE ABOUT VERTICAL
def spherical_to_tilts(theta,phi):
	x,y,z = np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)
	beta = np.arcsin(np.clip(x,-1,1))
	alpha = np.arctan2(y,z)
	return alpha,beta

# ASSEMBLE A SPHERICAL TRIANGLE BETWEEN THE POLE AND TWO THETA_MAX DIRECTIONS
print("assemble tasks for workers")
corners = np.asarray([[0,0,1],[np.sin(theta_max),0,np.cos(theta_max)],
	[np.sin(theta_max)*np.cos(phi_max),np.sin(theta_max)*np.sin(phi_max),np.cos(theta_max)]])
rng = np.random.default_rng(0)
tasks = []
for i in range(n_tilts):
	for j in range(n_tilts - i):
		weights = np.asarray([n_tilts-1-i-j,i,j])/(n_tilts-1)
		direction = weights@corners
		direction /= np.linalg.norm(direction)
		theta = np.arccos(np.clip(direction[2],-1,1))
		phi = 0 if theta < 1e-12 else np.mod(np.arctan2(direction[1],direction[0]),2*np.pi)
		alpha,beta = spherical_to_tilts(theta,phi)
		probe_positions = rng.random((n_probes, 2)) * Area
		grid_position = (weights[2]+weights[0]/2,weights[0])
		tasks.append((theta,phi,alpha,beta,grid_position,probe_positions))

# WORKER FUNCTION FOR MULTIPROCESSING: processes a single alpha/beta tilt
def simulate_tilt(args):
	theta,phi,alpha,beta,grid_position,probe_positions = args
	traj = trajectory.tilt_positions(alpha,beta)
	p = traj.positions[0]
	center = (p.min(axis=0)+p.max(axis=0))/2
	keep = (np.abs(p[:,0]-center[0]) <= Area[0]/2) & (np.abs(p[:,1]-center[1]) <= Area[1]/2)
	z0,z1 = p[:,2].min(),p[:,2].max()
	shift = np.array([center[0]-Area[0]/2,center[1]-Area[1]/2,z0])
	pristine = Trajectory(traj.atom_types[keep].copy(),traj.positions[:,keep]-shift,traj.velocities[:,keep].copy(),
		np.diag([Area[0],Area[1],z1-z0]),traj.timestep)
	traj = pristine.generate_random_displacements(n_configs,sigma,seed=0)
	calculator = MultisliceCalculator()
	calculator.setup(traj,aperture=mrad,voltage_eV=voltage_eV,
		probe_positions=probe_positions,cache_wavefunctions=cache_wavefunctions,sampling=dxy,slice_thickness=dz)
	exitwaves = calculator.run()
	structure_path = exitwaves.cache_dir/"structure.xyz"
	write(structure_path,pristine.to_ase(),format="extxyz")
	pattern = np.mean(np.abs(to_numpy(exitwaves.array[:,:,:,:,-1]))**2,axis=(0,1)).T**.5
	extent = (exitwaves.kxs[0],exitwaves.kxs[-1],exitwaves.kys[0],exitwaves.kys[-1])
	return (theta,phi),(alpha,beta),grid_position,pattern,extent,structure_path

# SPAWN
print("spawn workers")
if n_workers == 1:
	results = list(map(simulate_tilt,tasks))
else:
	with mp.get_context("fork").Pool(n_workers) as pool:
		results = pool.map(simulate_tilt,tasks)
angles = np.asarray([result[0] for result in results])
tilts = np.asarray([result[1] for result in results])
grid_positions = np.asarray([result[2] for result in results])
patterns = [result[3] for result in results]
extent = results[0][4]
structure_paths = [result[5] for result in results]

# DISPLAY NEAREST SIMULATED TILT AND ITS STRUCTURE
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
image = ax_pattern.imshow(patterns[0]**.1, cmap="inferno", extent=extent, origin="lower")
ax_pattern.set(xlabel="kx (1/Å)", ylabel="ky (1/Å)", title="θ=0.0°, φ=arbitrary; α=0.0°, β=0.0°")
canvas = FigureCanvasTkAgg(fig, master=root)
toolbar = NavigationToolbar2Tk(canvas,root,pack_toolbar=False)
toolbar.update()
toolbar.pack(side=tk.BOTTOM,fill=tk.X)
canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=True)

def display_structure(path):
	atoms = read(path)
	positions = atoms.get_positions()
	ax_structure.clear()
	ax_structure.scatter(*positions.T,s=3,c=atoms.get_atomic_numbers(),cmap="tab20",depthshade=True)
	ax_structure.set(xlabel="x (Å)",ylabel="y (Å)",zlabel="z (Å)")
	ax_structure.set_box_aspect(np.maximum(np.ptp(positions,axis=0),1),zoom=1.35)
	ax_structure.view_init(elev=90,azim=-90)
	ax_structure.set_proj_type('ortho')
	ax_structure.set_title("Structure: beam view down z\nleft drag: rotate; toolbar: pan/zoom")

def select_tilt(event):
	if event.inaxes is not ax_tilts:
		return
	k = np.argmin(np.sum((grid_positions - [event.xdata,event.ydata]) ** 2,axis=1))
	image.set_data(patterns[k]**.1)
	theta,phi = np.rad2deg(angles[k])
	alpha,beta = np.rad2deg(tilts[k])
	ax_pattern.set_title(f"θ={theta:.1f}°, φ={phi:.1f}°; α={alpha:.1f}°, β={beta:.1f}°")
	display_structure(structure_paths[k])
	canvas.draw_idle()

canvas.mpl_connect("button_press_event", select_tilt)
display_structure(structure_paths[0])
canvas.draw()
root.mainloop()
