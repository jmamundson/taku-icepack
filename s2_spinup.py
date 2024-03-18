# Known issues
# - height above flotation interpolator not working if haf_desired = 0

import firedrake

import matplotlib.pyplot as plt
import numpy as np


import icepack
import icepack.models.hybrid

from firedrake import max_value

import tqdm

import func
from func import schoof_approx_friction, side_drag, constants, params
constant = constants()
param = params()

#%% 
# initialize mesh
L = param.L
mesh1d = firedrake.IntervalMesh(param.n, L)
mesh = firedrake.ExtrudedMesh(mesh1d, layers=1, name="mesh")

# set up function spaces for the scalars (Q) and vectors (V) for the 2D mesh
Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2, name='velocity')

# create scalar functions for the spatial coordinates
x_sc, z_sc = firedrake.SpatialCoordinate(mesh)
x = firedrake.interpolate(x_sc, Q)
z = firedrake.interpolate(z_sc, Q)

# create initial geometry
b, tideLine = func.bedrock(x, Q=Q) 
h = func.initial_thickness(x, Q)                  
s = icepack.compute_surface(thickness = h, bed = b) 
u = func.initial_velocity(x, V)
w = func.width(x, Q=Q)

# initialize sediment model with sediment thickness of 0
sed = func.sedModel(param.L, param.sedDepth)
sed.H = 0*sed.H


# set up hybrid model solver with custom friction function
model = icepack.models.HybridModel(friction = schoof_approx_friction)
opts = {
    "dirichlet_ids": [1],
    #"diagnostic_solver_type": "petsc",
    #"diagnostic_solver_parameters": {"snes_type": "newtontr"},
}

solver = icepack.solvers.FlowSolver(model, **opts)


years = 50
dt = param.dt
num_timesteps = int(years/dt)

# set up basic figure
fig, axes = plt.subplots(2, 1)
axes[0].set_xlabel('Longitudinal Coordinate [m]')
axes[0].set_ylabel('Speed [m/yr]')
axes[0].set_xlim([0,40e3])
axes[1].set_xlabel('Longitudinal Coordinate [m]')
axes[1].set_ylabel('Elevation [m]')
axes[1].set_xlim([0,40e3])
firedrake.plot(icepack.depth_average(b), edgecolor='k', axes=axes[1]);
plt.tight_layout();

color_id = np.linspace(0,1,num_timesteps)

# create length and time arrays for storing changes in glacier length
length = np.zeros(num_timesteps+1)
length[0] = L

time = np.linspace(0, num_timesteps*dt, num_timesteps, endpoint=True)


for step in tqdm.trange(num_timesteps):
    # solve for velocity
    u = solver.diagnostic_solve(
        velocity = u,
        thickness = h,
        surface = s,
        fluidity = constant.A,
        friction = constant.C,
        side_friction = side_drag(w)
    )
    
    u_bar = icepack.interpolate(4/5*u, V) # width-averaged velocity
    
    # determine mass balance rate and adjusted mass balance profile
    a, a_mod = func.massBalance(x, s, u_bar, h, w, param)
    
    # solve for new thickness
    h = solver.prognostic_solve(
        dt,
        thickness = h,
        velocity = u_bar,
        accumulation = a_mod)
    
    # determine surface elevation
    s = icepack.compute_surface(thickness = h, bed = b)
    
    # find new terminus position
    # L_new = np.max([func.find_endpoint_massflux(L, x, a, u_bar, h, w, dt), tideLine])
    L_new = np.max([func.find_endpoint_haf(L, h, s), tideLine])
    
    # regrid, if necessary
    if L_new > tideLine: # if tidewater glacier, always need to regrid
        Q, V, h, u, b, s, w, mesh, x = func.regrid(param.n, x, L, L_new, h, u, sed) # regrid velocity and thickness
                
    elif (L_new==tideLine) and (L_new<L): # just became land-terminating, need to regrid
        Q, V, h, u, b, s, w, mesh, x = func.regrid(param.n, x, L, L_new, h, u, sed) # regrid velocity and thickness
        h.interpolate(max_value(h, constant.hmin))
        
    else: # land-terminating and was previously land-terminating, only need to ensure minimum thickness
        h.interpolate(max_value(h, constant.hmin))
        s = icepack.compute_surface(thickness = h, bed = b)
        
    L = L_new # reset the length
    length[step+1] = L

    zb = firedrake.interpolate(s - h, Q) # glacier bottom; not the same as the bedrock function if floating

    # update model with new friction coefficients
    model = icepack.models.HybridModel(friction = schoof_approx_friction)
    opts = {
        "dirichlet_ids": [1],
        #"diagnostic_solver_type": "petsc",
        #"diagnostic_solver_parameters": {"snes_type": "newtontr"},
    }
     
    solver = icepack.solvers.FlowSolver(model, **opts)

    if step%10==0:
        firedrake.plot(icepack.depth_average(u), edgecolor=plt.cm.viridis(color_id[step]), axes=axes[0]);
        firedrake.plot(icepack.depth_average(s), edgecolor=plt.cm.viridis(color_id[step]), axes=axes[1]);
        firedrake.plot(icepack.depth_average(zb), edgecolor=plt.cm.viridis(color_id[step]), axes=axes[1]);
        plt.plot(np.array([L_new, L_new]), np.array([zb.dat.data[-1], s.dat.data[-1]]), color=plt.cm.viridis(color_id[step]))
    
   
    filename = './results/spinup/spinup_' + "{:03}".format(step) + '.h5'
    with firedrake.CheckpointFile(filename, "w") as checkpoint:
        checkpoint.save_mesh(mesh)
        checkpoint.save_function(x, name="position")
        checkpoint.save_function(h, name="thickness")
        checkpoint.save_function(s, name="surface")
        checkpoint.save_function(u, name="velocity")
        checkpoint.save_function(b, name="bed")
        checkpoint.save_function(w, name="width")
