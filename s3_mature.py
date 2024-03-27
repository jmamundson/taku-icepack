# Known issues
# - height above flotation interpolator not working if haf_desired = 0
# - hillslope processes if no erosion/deposition; there is a kind in the bed at the bedrock/sediment transition
#   need to fix somehow. also affects the surface geometry and velocity.

import firedrake

import matplotlib.pyplot as plt
import numpy as np


import icepack
import icepack.models.hybrid

from firedrake import max_value

import tqdm

import func
from func import schoof_approx_friction, side_drag, constants, params
from func import sediment

constant = constants()
param = params()

import glob
import pickle

#%% 
files = sorted(glob.glob('./results/spinup/*h5'))


with firedrake.CheckpointFile(files[-1], "r") as checkpoint:
    mesh = checkpoint.load_mesh(name="mesh")
    x = checkpoint.load_function(mesh, name="position")
    h = checkpoint.load_function(mesh, name="thickness")
    s = checkpoint.load_function(mesh, name="surface")
    u = checkpoint.load_function(mesh, name="velocity")
    b = checkpoint.load_function(mesh, name="bed")
    w = checkpoint.load_function(mesh, name="width")

# set up function spaces for the scalars (Q) and vectors (V) for the 2D mesh
Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2, name='velocity')

# create scalar functions for the spatial coordinates
x_sc, z_sc = firedrake.SpatialCoordinate(mesh)
x = firedrake.interpolate(x_sc, Q)
z = firedrake.interpolate(z_sc, Q)


# find initial glacier length
L = np.max(x.dat.data)

# find tideLine; currently this will only work if the glacier is terminating in the ocean
_, tideLine = func.bedrock(x, Q=Q) # 

# initialize sediment model to fill fjord at level equal to the base of the terminus
# (also requires that terminus be in the water) 
# sed = func.sedModel(param.Lsed, -b.dat.data[-1]-10)
# sed.H[sed.H<2] = 2
sed = sediment(-b.dat.data[-1]-10) # initialize the sediment model


# set up hybrid model solver with custom friction function
model = icepack.models.HybridModel(friction = schoof_approx_friction)
opts = {
    "dirichlet_ids": [1],
    #"diagnostic_solver_type": "petsc",
    #"diagnostic_solver_parameters": {"snes_type": "newtontr"},
}

solver = icepack.solvers.FlowSolver(model, **opts)


years = 400
dt = param.dt
num_timesteps = int(years/dt)

# create length and time arrays for storing changes in glacier length
length = np.zeros(num_timesteps+1)
length[0] = L

time = np.linspace(0, num_timesteps*dt, num_timesteps+1, endpoint=True)

num_timesteps = 100

#%%
for step in tqdm.trange(num_timesteps):
    # solve for velocity
    u = solver.diagnostic_solve(
        velocity = u,
        thickness = h,
        surface = s,
        fluidity = constant.A,
        friction = constant.C,
        U0 = constant.U0,
        side_friction = side_drag(w)
    )
    
    u_bar = icepack.interpolate(4/5*u, u.function_space()) # width-averaged velocity
    
    # determine mass balance rate and adjusted mass balance profile
    a, a_mod = func.massBalance(x, s, u_bar, h, w, param)
    
    # solve for new thickness
    h = solver.prognostic_solve(
        dt,
        thickness = h,
        velocity = u_bar,
        accumulation = a_mod)
    
    # erode the bed
    # b = sed.sedTransportImplicit(x, h, a, b, u, Q, dt)
    b = sed.transport(x, b, a, w, h, dt)
    
    
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
    #length[step+1] = L

    print(' L: ' + '{:.2f}'.format(L*1e-3) + ' km')

    zb = firedrake.interpolate(s - h, Q) # glacier bottom; not the same as the bedrock function if floating

    # update model with new friction coefficients
    model = icepack.models.HybridModel(friction = schoof_approx_friction)
    opts = {
        "dirichlet_ids": [1],
        #"diagnostic_solver_type": "petsc",
        #"diagnostic_solver_parameters": {"snes_type": "newtontr"},
    }
     
    solver = icepack.solvers.FlowSolver(model, **opts)
    
    basename = './results/mature/mature_' + "{:04}".format(step)
    # filename = './results/mature/mature_' + "{:03}".format(step) + '.h5'
    with firedrake.CheckpointFile(basename + '.h5', "w") as checkpoint:
        checkpoint.save_mesh(mesh)
        checkpoint.save_function(x, name="position")
        checkpoint.save_function(h, name="thickness")
        checkpoint.save_function(s, name="surface")
        checkpoint.save_function(u, name="velocity")
        checkpoint.save_function(b, name="bed")
        checkpoint.save_function(w, name="width")
        
    # with open(basename + '_sed.pickle', 'wb') as file:
    #     pickle.dump(sed, file)
    #     file.close()    

    func.basicPlot(x, h, s, u, b, w, sed, basename, time[step])