"""Spectral energy density helpers for phonon-dispersion analysis.

The implementation follows the pySED-style formulation, projecting atomic
displacements onto reciprocal-space phase factors and Fourier transforming the
time axis.
"""
# Spectral Energy Density: phonon dispersions: borrowed from pySED: https://github.com/tpchuckles/pySED
# avg - average positions [a,xyz]
# displacements - time-dependent atom displacements [t,a,xyz]
# kvec - an NxM grid of length-3 vectors for each k point
# v_xyz - 0,1,2 indicating if we'll track displacements in x,y or z (this is your vibration polarization direction). vector also allowed: e.g. [1,1,0] for diagonal
# bs - optional: should be a list of atom indices to include. this allows the caller to sum over crystal cell coordinates (see discussion on Σb below)

import numpy as np

def SED(avg,displacements,kvec,v_xyz=0,bs=''):
	"""Calculate spectral energy density on a grid of reciprocal vectors.

	Args:
		avg: Mean atomic positions with shape ``(n_atoms, 3)``.
		displacements: Time-dependent displacements with shape
			``(n_frames, n_atoms, 3)``.
		kvec: Reciprocal-vector grid with shape ``(..., 3)``.
		v_xyz: Polarization axis as 0/1/2 or an arbitrary 3-vector.
		bs: Optional atom indices to include.

	Returns:
		``(intensity, frequencies)`` where intensity is ``|FFT|^2`` over time.
	"""

	# THE MATH: 
	#    Φ(k,ω) = Σb | ∫ Σn u°(n,b,t) exp( i k r̄(n,0) - i ω t ) dt |²
	# https://github.com/tyst3273/phonon-sed/blob/master/manual.pdf
	# b is index *within* a unit cell, n is index *of* unit cell. pairs of 
	# n,b are indices pointing to a specific atom. u° is the velocity of 
	# each atom as a function of time. r̄(n,b=0) is the equilibrium position 
	# of the unit cell. (if atom b=0 in a unit cell is at the origin, then 
	# we can use x̄(n,b=0), i.e., the atom's average position). 
	#    looping over n inside the integral, but not b, means there is 
	# coherent summing over "every bth atom". the maximum k will then depend
	# on the spacing between these atoms (unit cell lattice constant), since 
	# higher-k (or lower wavelength) modes will see aliasing. short-
	# wavelength optical modes will register as their BZ-folded longer-
	# wavelength selves*. 
	#    using r̄(n,b=0) rather than x̄(n,b) means a phase-shift is applied
	# for b≠0 atoms. this means if we ignore b,n ("...Σb | ∫ Σn...") and sum 
	# across all atoms instead ("... | ∫ Σi..."), we lose BZ folding. if we
	# don't care (fine, so long as a is small enough / k is large is small
	# enough to "unfold" the dispersion), we simplify summing, and can use x̄
	# instead. we thus no longer require a perfect crystal to be analyzed.
	#    the above equation thus simplified to:
	# Φ(k,ω) = | ∫ Σn u°(n,t) exp( i k x̄(n) - i ω t ) dt |²
	# and noting the property: e^(A+B)=e^(A)*e^(B)
	# Φ(k,ω) = | ∫ Σn u°(n,t) exp( i k x̄(n) ) * exp( - i ω t ) dt |²
	# and noting that the definition of a fourier transform:
	# F(w) = ∫ f(t) * exp( -i 2 π ω t ) dt
	# we can reformulate the above eqaution as:
	# f(t) = u°(n,t) * exp( i k x )
	# or:
	# Φ(k,ω) = | FFT{ Σn f(t) } |² (this is much faster)
	#    of course, this code still *can* analyze crystals with folding: the 
	# user should simply call this function multiple times, passing the "bs"
	# argument with a list of atom indices for us to use
	#    to process waves along an arbitrary vector, you you can *either* 
	# project positions onto the vector (then ks are just reciprocal 
	# distances along that vector), *or*, simply multiply position vectors 
	# (x,y,z positions) by k (as a vector, kx,ky,kz) and sum. when plugged
	# into exp(ikx), you get the same result! 

	# PRE-PROCESSING
	nt,na,nax=np.shape(displacements)	# Infer system shape

	if len(bs)==0:						# if user didn't specify specific atom indices, we take all atoms
		bs=np.arange(na)
	else:
		na=len(bs)

	# user might pass an index for xyz, or a vector
	if isinstance(v_xyz,(int,float,np.integer)):
		v_xyz=np.roll([1,0,0],v_xyz)					# 0,1,2 --> x,y,z
	else:
		v_xyz=np.asarray(v_xyz,dtype=float)
		v_xyz/=np.sqrt(np.sum(v_xyz**2))	# normalize magnitude of passed vector

	# time axis: infer frequencies, calculate length so we can trim off negative frequencies, set up empty 
	nt2=int(nt/2) ; ws=np.fft.fftfreq(nt)[:nt2]

	# project displacements onto vector
	if isinstance(v_xyz,(int,float,np.integer)):
		us=displacements[:,bs,v_xyz] # t,a,xyz --> t,a
	else: 
		us=np.einsum("tax,x->ta",displacements[:,bs,:],v_xyz)	# indices: (t)ime, (a)tom index, (x)/y/z direction
	
	# real spectral *energy* density needs to use masses ( Ek = ½ m v² )
	#if masses is not None:
	#	us=us[:,:]*masses[None,:] # t,a indices for velocities

	# damping out beginning and end of time series may help with noise
	#if hannFilter:
	#	ts=np.linspace(0,np.pi,len(vs)) ; hann=np.sin(ts)**2
	#	vs*=hann[:,None]

	# FINALLY TIME FOR MATH! and how does einsum work?
	# Suppose I have the two matrices: the first has axis indices of 
	# a,k,xyz (which (a)tom, which (k) point, which direction of (x)/y/z). I 
	# want to multiply with the second matrix,  with axis indices of t,a,xyz 
	# (which (t)ime point, which (a)tom, which direction of (x)/y/z). then I
	# want to sum across a and xyz. I want to do: 
	# "v[:,:,None,:]*expo[None,:,:,:]"
	# which creates a giganormous matrix of indices t,a,k,xyz, then we can
	# sum via:
	# np.sum(that,axis=(1,3))
	# to obtain a result with indices of t,k
	# *or* we can use np.einsum:
	# np.eigsum("tax,akx->tk",v,expo)
	# using einsum means we don't need to populate the super huge 4D matrix,
	# but we still do all the same math, super fast, using underlying C
	# code, without needing to do (slow) python loops. 

	# Φ(k,ω) = Σb | ∫ Σn u°(n,b,t) exp( i k r̄(n,0) ) exp( i ω t ) dt |² or | ℱ[ Σn u°(n,b,t) exp( i k r̄(n,0) ) |²
	# exp( i k r̄(n,0) ) term:
	expo=np.exp(1j*np.einsum('aj,xyj->axy',avg[bs,:],kvec[:,:,:])) # indices: (a)tom, (x)/y/z, (k) point
	# u°(n,b,t) exp( i k r̄(n,0) ) term:
	integrands=np.einsum('ta,axy->txy',us,expo,optimize=True) # indices: (t)ime, (a)toms, (k) point
	Zs=np.fft.fft(integrands,axis=0)[:nt2,:,:]
	#if not keepComplex:
	Zs=np.absolute(Zs)**2

	return Zs,ws
