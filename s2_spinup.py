import firedrake

import numpy as np

import icepack
import icepack.models.hybrid

from firedrake import max_value

import tqdm

import func
from func import schoof_approx_friction, side_drag, constants, params, glacier, sedimentFD

constant = constants()
param = params()

#%% 
# initialize glacier model
glac = glacier()

# initialize sediment model with sediment thickness of 0
# don't actually use it during model spin up but needed to pass into basic
# plotting function
sed = sedimentFD(-50)
sed.H[sed.H>0] = 0

# set up hybrid model solver with custom friction function
model = icepack.models.HybridModel(friction = schoof_approx_friction)
opts = {
    "dirichlet_ids": [1],
}

solver = icepack.solvers.FlowSolver(model, **opts)


years = 100
dt = param.dt
num_timesteps = int(years/dt)


# create length and time arrays for storing changes in glacier length
L = np.max(glac.x.dat.data)
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
        side_friction = side_drag(glac.w),
    )
    
    # determine mass balance rate
    glac.massBalance()
    
    # solve for new thickness
    # use the prognostic solver to determine the new cross-sectional area, 
    # then divide by width to get the ice thickness
    
    glac.u_bar = icepack.interpolate(4/5*glac.u, glac.V) # width-averaged velocity
    
    area = solver.prognostic_solve(
        dt,
        thickness = icepack.interpolate(glac.h*glac.w, glac.Q),
        velocity = glac.u_bar,
        accumulation = icepack.interpolate(glac.a*glac.w, glac.Q)
        )
    
    glac.h = icepack.interpolate(area/glac.w, glac.Q)
    
    
    # determine surface elevation
    glac.s = icepack.compute_surface(thickness = glac.h, bed = glac.bed)

    # find new terminus position
    # L_new = np.max([glac.massFlux(), glac.tideLine])
    # L_new = np.max([glac.HAF(), glac.tideLine])
    L_new = np.max([glac.HAFmodified(), glac.tideLine])
    # L_new = np.max([glac.crevasseDepth(), glac.tideLine])
    # L_new = np.max([glac.eigencalving(), glac.tideLine])
    # L_new = np.max([glac.vonMises(), glac.tideLine])
    
    # regrid, if necessary
    if L_new > glac.tideLine: # if tidewater glacier, always need to regrid
        glac.regrid(param.n, L, L_new, sed) # regrid velocity and thickness
                
    elif (L_new==glac.tideLine) and (L_new<L): # just became land-terminating, need to regrid
        glac.regrid(param.n, L, L_new, sed) # regrid velocity and thickness
        glac.h.interpolate(max_value(glac.h, constant.hmin))
        
    else: # land-terminating and was previously land-terminating, only need to ensure minimum thickness
        glac.h.interpolate(max_value(glac.h, constant.hmin))
        glac.s = icepack.compute_surface(thickness = glac.h, bed = glac.bed)
    
    glac.balanceFlux()
    
    L = L_new # reset the length
    length[step+1] = L


    # update model with new friction coefficients
    model = icepack.models.HybridModel(friction = schoof_approx_friction)
    opts = {
        "dirichlet_ids": [1],
        #"diagnostic_solver_type": "petsc",
        #"diagnostic_solver_parameters": {"snes_type": "newtontr"},
    }
     
    solver = icepack.solvers.FlowSolver(model, **opts)

    basename = './results/spinup/spinup_' + "{:04}".format(step)
    with firedrake.CheckpointFile(basename + '.h5', "w") as checkpoint:
        checkpoint.save_mesh(glac.mesh)
        checkpoint.save_function(glac.x, name="position")
        checkpoint.save_function(glac.h, name="thickness")
        checkpoint.save_function(glac.s, name="surface")
        checkpoint.save_function(glac.u, name="velocity")
        checkpoint.save_function(glac.bed, name="bedrock")
        checkpoint.save_function(glac.b, name="bed")
        checkpoint.save_function(glac.w, name="width")

    func.basicPlotFD(glac, sed, basename, time[step])
