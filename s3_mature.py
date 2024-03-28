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
from func import sediment, glacier

constant = constants()
param = params()

import glob
import pickle

#%% 
files = sorted(glob.glob('./results/spinup/*h5'))

glac = glacier()
with firedrake.CheckpointFile(files[-1], "r") as checkpoint:
    glac.mesh = checkpoint.load_mesh(name="mesh")
    glac.x = checkpoint.load_function(glac.mesh, name="position")
    glac.h = checkpoint.load_function(glac.mesh, name="thickness")
    glac.s = checkpoint.load_function(glac.mesh, name="surface")
    glac.u = checkpoint.load_function(glac.mesh, name="velocity")
    glac.b = checkpoint.load_function(glac.mesh, name="bed")
    glac.w = checkpoint.load_function(glac.mesh, name="width")
    
# set up function spaces for the scalars (Q) and vectors (V) for the 2D mesh
glac.Q = firedrake.FunctionSpace(glac.mesh, "CG", 2, vfamily="R", vdegree=0)
glac.V = firedrake.FunctionSpace(glac.mesh, "CG", 2, vfamily="GL", vdegree=2, name='velocity')

# create scalar functions for the spatial coordinates
x_sc, z_sc = firedrake.SpatialCoordinate(glac.mesh)
glac.x = firedrake.interpolate(x_sc, glac.Q)
glac.z = firedrake.interpolate(z_sc, glac.Q)


# find initial glacier length
L = np.max(glac.x.dat.data)

# find tideLine; currently this will only work if the glacier is terminating in the ocean
_, glac.tideLine = func.bedrock(glac.x, Q=glac.Q) # 

# initialize sediment model to fill fjord at level equal to the base of the terminus
# (also requires that terminus be in the water) 
# sed = func.sedModel(param.Lsed, -b.dat.data[-1]-10)
# sed.H[sed.H<2] = 2
sed = sediment(-glac.b.dat.data[-1]-10) # initialize the sediment model


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


#%%
for step in tqdm.trange(num_timesteps):
    # solve for velocity
    glac.u = solver.diagnostic_solve(
        velocity = glac.u,
        thickness = glac.h,
        surface = glac.s,
        fluidity = constant.A,
        friction = constant.C,
        U0 = constant.U0,
        side_friction = side_drag(glac.w)
    )
    
    glac.u_bar = icepack.interpolate(4/5*glac.u, glac.V) # width-averaged velocity
    
    # determine mass balance rate and adjusted mass balance profile
    glac.massBalance()
    
    # solve for new thickness
    glac.h = solver.prognostic_solve(
        dt,
        thickness = glac.h,
        velocity = glac.u_bar,
        accumulation = glac.a_mod)
    
    # erode the bed
    # b = sed.sedTransportImplicit(x, h, a, b, u, Q, dt)
    sed.transport(glac, dt)
    
    
    # determine surface elevation
    glac.s = icepack.compute_surface(thickness = glac.h, bed = glac.b)
    
    # find new terminus position
    # L_new = np.max([func.find_endpoint_massflux(L, x, a, u_bar, h, w, dt), tideLine])
    L_new = np.max([func.find_endpoint_haf(L, glac.h, glac.s), glac.tideLine])
    
    # regrid, if necessary
    if L_new > glac.tideLine: # if tidewater glacier, always need to regrid
        glac.regrid(param.n, L, L_new, sed) # regrid velocity and thickness
                
    elif (L_new==glac.tideLine) and (L_new<L): # just became land-terminating, need to regrid
        glac.regrid(param.n, L, L_new, sed) # regrid velocity and thickness
        glac.h.interpolate(max_value(glac.h, constant.hmin))
        
    else: # land-terminating and was previously land-terminating, only need to ensure minimum thickness
        glac.h.interpolate(max_value(glac.h, constant.hmin))
        glac.s = icepack.compute_surface(thickness = glac.h, bed = glac.b)
        
    L = L_new # reset the length
    #length[step+1] = L

    print(' L: ' + '{:.2f}'.format(L*1e-3) + ' km')

    zb = firedrake.interpolate(glac.s - glac.h, glac.Q) # glacier bottom; not the same as the bedrock function if floating

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
        checkpoint.save_mesh(glac.mesh)
        checkpoint.save_function(glac.x, name="position")
        checkpoint.save_function(glac.h, name="thickness")
        checkpoint.save_function(glac.s, name="surface")
        checkpoint.save_function(glac.u, name="velocity")
        checkpoint.save_function(glac.b, name="bed")
        checkpoint.save_function(glac.w, name="width")
    
    basenameSed = './results/mature/matureSed_' + "{:04}".format(step)
    with firedrake.CheckpointFile(basename + '.h5', "w") as checkpoint:
        checkpoint.save_mesh(sed.mesh)
        checkpoint.save_function(sed.x, name="position")
        checkpoint.save_function(sed.H, name="thickness")
        checkpoint.save_function(sed.bedrock, name="bedrock")
        checkpoint.save_function(sed.erosionRate, name="erosionRate")
        checkpoint.save_function(sed.depositionRate, name="depositionRate")
        checkpoint.save_function(sed.dHdt, name="dHdt")
        checkpoint.save_function(sed.Qw, name="runoff")
        checkpoint.save_function(sed.Qs, name="sedimentFlux")
        

    func.basicPlot(glac, sed, basename, time[step])