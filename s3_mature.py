# Known issues
# - height above flotation interpolator not working if haf_desired = 0
# - also numerical issues if ice less than 10 m thick or so
# - add lateral shear stress; allow for high accumulation rates?
# - possible to add variable width?
# - interpolate (last two points with xarray?)
# - first two points swapped with depth average
# - degree two or degree 4 function space?
# - some issue with transition from land-terminating to tidewater

import firedrake
from firedrake import sqrt, inner, ln, exp
from firedrake import assemble, dx # for computing integrals

import matplotlib.pyplot as plt
import numpy as np


import icepack
import icepack.models.hybrid

from firedrake import max_value, min_value
from firedrake import conditional, eq, ne, le, ge, lt, gt

import tqdm

import func
from func import schoof_approx_friction, constants, params
constant = constants()
param = params()

import glob

#%% 


sed = func.sedModel(param.L,param.sedDepth)

files = sorted(glob.glob('./results/spinup/*'))

with firedrake.CheckpointFile(files[-1], "r") as checkpoint:
    mesh = checkpoint.load_mesh(name="mesh")
    h = checkpoint.load_function(mesh, name="thickness")
    s = checkpoint.load_function(mesh, name="surface")
    u = checkpoint.load_function(mesh, name="velocity")

# Set up function spaces for the scalars (Q) and vectors (V) for the 2D mesh.
Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2, name='velocity')

x_sc, z_sc = firedrake.SpatialCoordinate(mesh)
x = firedrake.interpolate(x_sc, Q)
z = firedrake.interpolate(z_sc, Q)

L = np.max(x.dat.data)

# create initial geometry
b, tideLine = func.bedrock(x, Q) # firedrake bed function

fig, axes = plt.subplots(2, 2)
axes[0,0].set_xlabel('Longitudinal Coordinate [m]')
axes[0,0].set_ylabel('Speed [m/yr]')
axes[1,0].set_xlabel('Longitudinal Coordinate [m]')
axes[1,0].set_ylabel('Elevation [m]')
axes[0,1].set_xlabel('Longitudinal Coordinate [m]')
axes[0,1].set_ylabel('Sediment thickness [m]')
axes[1,1].set_xlabel('Longitudinal coordinate [m]')
axes[1,1].set_ylabel('Erosion or deposition rate [m a$^{-1}$]')
firedrake.plot(icepack.depth_average(b), edgecolor='k', axes=axes[1,0]);
# axes[1].plot(x.dat.data, b.dat.data, 'k')
plt.tight_layout();



# Set up hybrid model solver with custom friction function and initialize the velocity field.
model = icepack.models.HybridModel(friction = schoof_approx_friction)
opts = {
    "dirichlet_ids": [1],
    #"diagnostic_solver_type": "petsc",
    #"diagnostic_solver_parameters": {"snes_type": "newtontr"},
}

solver = icepack.solvers.FlowSolver(model, **opts)


years = 50
timesteps_per_year = 20
snapshot_location = [0, 50, 100, 150, 200]
snapshots = []

dt = 1/timesteps_per_year
num_timesteps = years * timesteps_per_year

color_id = np.linspace(0,1,num_timesteps)

param.ELA = 500 # lower the ELA

length = np.zeros(num_timesteps+1)
length[0] = L

for step in tqdm.trange(num_timesteps):
    u = solver.diagnostic_solve(
        velocity = u,
        thickness = h,
        surface = s,
        fluidity = constant.A,
        friction = constant.C,
    )
    
    a = func.massBalance(s, Q, param)
    
    h = solver.prognostic_solve(
        dt,
        thickness = h,
        velocity = u,
        accumulation = a)
    
    # erode the bed; what is the right order of doing these updates?
    b = sed.sedTransportImplicit(x, h, a, Q, dt)
    
    s = icepack.compute_surface(thickness = h, bed = b)
    
    L_new = np.max([func.find_endpoint_massflux(L, x, a, u, h, dt), tideLine])
    
    # L_new = np.max([func.find_endpoint_haf(L, h, s, Q), tideLine]) # find new terminus position
    
    if L_new > tideLine: # if tidewater glacier, always need to regrid
        Q, V, h, u, b, s, mesh, x = func.regrid(param.n, L, L_new, h, u, sed) # regrid velocity and thickness
                
    elif (L_new==tideLine) and (L_new<L): # just became land-terminating, need to regrid
        Q, V, h, u, b, s, mesh, x = func.regrid(param.n, L, L_new, h, u, sed) # regrid velocity and thickness
        h.interpolate(max_value(h, constant.hmin))
        
    else: # land-terminating and was previously land-terminating, only need to ensure minimum thickness
        h.interpolate(max_value(h, constant.hmin))
        s = icepack.compute_surface(thickness = h, bed = b)
        
    L = L_new # reset the length
    length[step] = L
    
    # update model after every time step?
    model = icepack.models.HybridModel(friction = schoof_approx_friction)
    opts = {
        "dirichlet_ids": [1],
        #"diagnostic_solver_type": "petsc",
        #"diagnostic_solver_parameters": {"snes_type": "newtontr"},
    }
     
    solver = icepack.solvers.FlowSolver(model, **opts)



    # z_b = firedrake.interpolate(s - h, Q) # glacier bottom
    
    
    #print(h.dat.data[-1] + z_b.dat.data[-1]*rho_water/rho_ice)
    # print(h.dat.data[0])
    
    if step%10==0:
        firedrake.plot(icepack.depth_average(u), edgecolor=plt.cm.viridis(color_id[step]), axes=axes[0,0]);
        firedrake.plot(icepack.depth_average(s), edgecolor=plt.cm.viridis(color_id[step]), axes=axes[1,0]);
        firedrake.plot(icepack.depth_average(b), edgecolor=plt.cm.viridis(color_id[step]), axes=axes[1,0]);
        axes[1,0].plot(np.array([L_new, L_new]), np.array([b.dat.data[-1],s.dat.data[-1]]), color=plt.cm.viridis(color_id[step]))

        axes[0,1].plot(sed.x, sed.H, color=plt.cm.viridis(color_id[step]))
        axes[1,1].plot(sed.x, sed.erosionRate, color=plt.cm.viridis(color_id[step]), label='Erosion rate')
        # axes[1,1].plot(sed.x, sed.depositionRate, color=plt.cm.viridis(color_id[step]), linestyle='--', label='Deposition rate')
        # axes[1,1].plot(sed.x, sed.hillslope, color=plt.cm.viridis(color_id[step]), linestyle=':', label='Deposition rate')
        
   
    filename = './results/mature/mature_' + "{:03}".format(step) + '.h5'
    with firedrake.CheckpointFile(filename, "w") as checkpoint:
        checkpoint.save_mesh(mesh)
        checkpoint.save_function(h, name="thickness")
        checkpoint.save_function(s, name="surface")
        checkpoint.save_function(u, name="velocity")
    

# plt.figure()
# plt.plot(sed.x, sed.erosionRate, sed.x, sed.depositionRate, sed.x, sed.hillslope)
# plt.plot(sed.x, sed.H+sed.zBedrock)
# plt.plot(x.dat.data, b.dat.data)





