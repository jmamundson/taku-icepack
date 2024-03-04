# Known issues
# - height above flotation interpolator not work if haf_desired = 0
# - turn off h_inflow to stablize upper boundary?
# - interpolating velocity introduces noise


import firedrake
from firedrake import sqrt, inner
from firedrake import ln, exp

import matplotlib.pyplot as plt
import numpy as np


import icepack
import icepack.models.hybrid

# Import some constants from icepack [NEED TO CHECK UNITS]
from icepack.constants import(
    ice_density as rho_ice,           # ρ_I | 9.21e-19
    water_density as rho_water,       # ρ_W | 1.03e-18
    weertman_sliding_law as weertman, # m   | 3.0
    gravity as g,                     # g   | 9.77e15
)

Temperature = firedrake.Constant(273.0) # -1 deg C
A = icepack.rate_factor(Temperature) # flow rate factor
C = firedrake.Constant(0.1) # friction coefficient

from firedrake import max_value, min_value
from firedrake import conditional, eq, ne, le, ge, lt, gt


from scipy.interpolate import interp1d
import xarray

import tqdm

#%% functions to specify bedrock elevation, initial thickness, and initial velocity
# later will add an initial sediment thickness
def bedrock(x, Q):
    
    zb = 1500*exp(-(x+5000)**2 / (2*15000**2)) - 400
    
    b = firedrake.interpolate(zb, Q)
    
    return(b)    
    

def initial_thickness(x, Q):
    
    h_divide, h_terminus = 500, 200 # thickness at the divide and the terminus [m]
    h = h_divide - (h_divide-h_terminus) * x/x.dat.data[-1] # thickness profile [m]
    h0 = firedrake.interpolate(h, Q)

    return(h0)


def initial_velocity(x, V):

    u_divide, u_terminus = 0, 500 # velocity at the divide and the terminus [m/yr]
    u = u_divide + (u_terminus - u_divide) * x/x.dat.data[-1] # velocity profile
    u0 = firedrake.interpolate(u, V) # Vector function space

    return(u0)


#%% miscellaneous functions

def schoof_approx_friction(**kwargs):
    u = kwargs['velocity']
    h = kwargs['thickness']
    s = kwargs['surface']
    C = kwargs['friction']

    U_0 = firedrake.Constant(50)
    
    U = sqrt(inner(u,u)) # Supposed to be u0, does this work?
    tau_0 = firedrake.interpolate(
        C*( U_0**(1/weertman+1) + U**(1/weertman+1) )**(1/(weertman+1)), u.function_space()
    )
    
    # Update pressures
    p_water = rho_water * g * firedrake.max_value(0, h - s) # assuming phreatic surface at sea level
    p_ice = rho_ice * g * h
    phi = 1 - p_water/p_ice

    U = sqrt(inner(u, u))
    return tau_0 * phi * (
        (U_0**(1/weertman+1) + U**(1/weertman+1))**(weertman/(weertman+1))-U_0
    )

#%%


def find_endpoint_haf(L, src_nx, h, s, Q):
    # at flotation
    
    zb = firedrake.interpolate(s-h, Q) # glacier bed elevation [m]
    
    h_flotation = firedrake.interpolate(-rho_water/rho_ice*zb, Q) # thickness flotation (thickness if at flotation) [m]
    haf = firedrake.interpolate(h - h_flotation, Q) # height above flotation [m]

    haf_dat = haf.dat.data # height above flotation as a numpy array
    x_dat = np.linspace(0, L, len(h.dat.data), endpoint=True) # grid points as a numpy array [m]
    haf_interpolator = interp1d(haf_dat-10, x_dat, fill_value='extrapolate')
    
    haf_desired = 10 # desired height above flotation at the terminus
    L_new = haf_interpolator(haf_desired)

    return L_new
    



def mass_balance(mb_h, mb_Q):
    '''
    Mass balance profile. Returns function space for the mass balance.

    Parameters
    ----------
    mb_h : TYPE
        DESCRIPTION.
    mb_Q : TYPE
        DESCRIPTION.

    Returns
    -------
    mb_a : TYPE
        DESCRIPTION.

    '''
    
    b_max = 5 # maximum mass balance rate [m a^{-1}]
    b_sea = -12 # mass balance rate at sea level [m a^{-1}]
    ELA = 1000 # equilbrium line altitude [m]
    k = 0.005 # smoothing factor
    b_gradient = -b_sea/ELA # mass balance gradient
    z_threshold = b_max/b_gradient + ELA # location where unsmoothed balance rate switches from nonzero to zero
    
    mb_a = firedrake.interpolate(b_gradient*((mb_h-ELA) - 1/(2*k) * ln(1+exp(2*k*(mb_h - z_threshold)))), mb_Q)
    
    return mb_a

#%% functions to re-mesh the model


def adjust_function(func_src, L, L_new, funcspace_dest):
    '''
    Shrinks function spaces.

    Parameters
    ----------
    func_src : source function
    nx : number of grid points
    Lx_src : Length of source function [m]
    Lx_dest : Length of destination function [m]
    funcspace_dest : destination function

    Returns
    -------
    func_dest : destination function

    '''
    
    
    # adjust_function(u, L, L_new, V_new)
    
    # Pull out data from func_src
    data = func_src.dat.data
    
    if funcspace_dest.topological.name=='velocity':
        data_len = int(len(data)/3)
    else:
        data_len = len(data)
    
    x = np.linspace(0, L, data_len, endpoint=True)
    x_new = np.linspace(0, L_new, data_len, endpoint=True)
    
    
    # Interpolate data from x to x_new
    if funcspace_dest.topological.name=='velocity':
        # velocity is defined at three vertical grid points, and the data is ordered
        # [bottom, middle, top, bottom, middle, top, ...]
        #
        # Not sure how to use icepack.interpolate to create this 2D grid.
        # Instead this hack just uses the mean velocity
        
        u_ = icepack.depth_average(func_src).dat.data
        u_[0], u_[1] = u_[1], u_[0] # swap first two values; seems to be an issue here with the depth averaging?
        
        data_interpolator = interp1d(x, u_, kind='linear', fill_value='extrapolate')
        data_new = data_interpolator(x_new)
        
        
    else:
        data_interpolator = interp1d(x, data, kind='linear', fill_value='extrapolate')
        data_new = data_interpolator(x_new)
    
    
    # Convert f_data_dest into a firedrake function on destination function space funcspace_dest
    temp_dest = xarray.DataArray(data_new, [x_new], 'x')
    func_dest = icepack.interpolate(temp_dest, funcspace_dest)

    func_dest.dat.data[-10:] = temp_dest[-10:] # sometimes last grid points aren't correctly interpolated [WHY?]
    
    return func_dest



def regrid(n, L, L_new, h, u):
    '''
    Now that the new function spaces have been created, interpolate on the new grid
    
    Parameters
    ----------
    n : number of grid points
    L : length of source function [m]
    L_new : length of destination function [m]
    h_src : source thickness [m]
    u_src : source velocity [m a^{-1}]
    a_src : TYPE
        DESCRIPTION.
    h0_src : TYPE
        DESCRIPTION.
    b_src : source bed [m]
    s_src : source surface [m]
        
    Returns
    -------
    Q_dest : destination scalar function space
    V_dest : destination vector function space
    h_dest : destination thickness [m]
    u_dest : destination velocity [m a^{-1}]
    a_dest : TYPE
        DESCRIPTION.
    h0_dest : TYPE
        DESCRIPTION.
    b_dest : destination bed [m]
    s_dest : destination surface [m]
    
    '''
    
    # Initialize mesh and mesh spaces
    mesh1d_new = firedrake.IntervalMesh(n, L_new)
    mesh_new = firedrake.ExtrudedMesh(mesh1d_new, layers=1)
    Q_new = firedrake.FunctionSpace(mesh_new, "CG", 2, vfamily="R", vdegree=0)
    V_new = firedrake.FunctionSpace(mesh_new, "CG", 2, vfamily="GL", vdegree=2, name='velocity')
    
    
    # Set up function spaces for the scalars (Q) and vectors (V) for the 2D mesh.
    # mesh1d = firedrake.IntervalMesh(n, L)
    # mesh = firedrake.ExtrudedMesh(mesh1d, layers=1)
    # Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
    # V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2, name='velocity')

    x_sc, z_sc = firedrake.SpatialCoordinate(mesh_new)
    x = firedrake.interpolate(x_sc, Q_new)
    z = firedrake.interpolate(z_sc, Q_new)
    
    b_new = bedrock(x, Q_new)
    
    # s_new = icepack.compute_surface(thickness = h_new, bed = b_new)
    # x_sc_new, _ = firedrake.SpatialCoordinate(mesh_new)
    # x_new = firedrake.interpolate(x_sc_new, Q_new)

    # Remesh thickness and velocity
    h_new =  adjust_function(h, L, L_new, Q_new)
    u_new =  adjust_function(u, L, L_new, V_new)

        
    s_new = icepack.compute_surface(thickness = h_new, bed = b_new) 
    
    return Q_new, V_new, u_new, h_new, b_new, s_new





#%% 
L = 40e3 # domain length [m]
n = 72 # number of grid points 

# initialize mesh
mesh1d = firedrake.IntervalMesh(n, L)
mesh = firedrake.ExtrudedMesh(mesh1d, layers=1)

# Set up function spaces for the scalars (Q) and vectors (V) for the 2D mesh.
Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2, name='velocity')

x_sc, z_sc = firedrake.SpatialCoordinate(mesh)
x = firedrake.interpolate(x_sc, Q)
z = firedrake.interpolate(z_sc, Q)

# It looks like there is an issue in Firedrake or Icepack where the first two 
# values of a `SpatialCoordinate` get swapped. I don't see a way to manipulate 
# the resulting variable directly. To fix this, we put the project the spatial 
# coordinates onto the function space, and swap the values there.

# x_sc_temp = x.dat.data[0]
# x.dat.data[0] = x.dat.data[1]
# x.dat.data[1] = x_sc_temp

# z_sc_temp = z.dat.data[0]
# z.dat.data[0] = z.dat.data[1]
# z.dat.data[1] = z_sc_temp


# create initial geometry [NOT YET INCLUDING SEDIMENT]
b = bedrock(x, Q) # use bedrock function to determine initial bedrock geometry
h = initial_thickness(x, Q) # use constant gradient for initial thickness                 
s = icepack.compute_surface(thickness = h, bed = b) # initial surface elevation
u = initial_velocity(x, V)


fig, axes = plt.subplots(2, 1)
axes[0].set_xlabel('Longitudinal Coordinate [m]')
axes[0].set_ylabel('Speed [m/yr]')
axes[1].set_xlabel('Longitudinal Coordinate [m]')
axes[1].set_ylabel('Elevation [m]')
axes[1].plot(x.dat.data, b.dat.data, 'k')
plt.tight_layout();



# Set up hybrid model solver with custom friction function and initialize the velocity field.
model = icepack.models.HybridModel(friction = schoof_approx_friction)
opts = {
    "dirichlet_ids": [1],
    #"diagnostic_solver_type": "petsc",
    #"diagnostic_solver_parameters": {"snes_type": "newtontr"},
}

solver = icepack.solvers.FlowSolver(model, **opts)


years = 40
timesteps_per_year = 10
snapshot_location = [0, 50, 100, 150, 200]
snapshots = []

dt = 1/timesteps_per_year
num_timesteps = years * timesteps_per_year

color_id = np.linspace(0,1,num_timesteps)

for step in tqdm.trange(num_timesteps):
    u = solver.diagnostic_solve(
        velocity = u,
        thickness = h,
        surface = s,
        fluidity = A,
        friction = C,
    )
    
    a = mass_balance(s, Q)
    
    h = solver.prognostic_solve(
        dt,
        thickness = h,
        velocity = u,
        accumulation = a)
    
    # h = firedrake.interpolate(conditional(h<10, 10, h), Q) # don't allow the ice to thin to less than 10 m
    
    s = icepack.compute_surface(thickness = h, bed = b)
    
    L_new = find_endpoint_haf(L, n, h, s, Q) # find new terminus position
    Q, V, u, h, b, s = regrid(n, L, L_new, h, u) # regrid velocity and thickness
    L = L_new # reset the length
    
    

    
    
    z_b = firedrake.interpolate(s - h, Q) # glacier bottom
    
    firedrake.plot(icepack.depth_average(u), edgecolor=plt.cm.viridis(color_id[step]), axes=axes[0]);
    firedrake.plot(icepack.depth_average(s), edgecolor=plt.cm.viridis(color_id[step]), axes=axes[1]);
    firedrake.plot(icepack.depth_average(z_b), edgecolor=plt.cm.viridis(color_id[step]), axes=axes[1]);
    plt.plot(np.array([L_new, L_new]), np.array([z_b.dat.data[-1],s.dat.data[-1]]), color=plt.cm.viridis(color_id[step]))


    # update model after every time step?
    
    
    model = icepack.models.HybridModel(friction = schoof_approx_friction)
    opts = {
        "dirichlet_ids": [1],
        #"diagnostic_solver_type": "petsc",
        #"diagnostic_solver_parameters": {"snes_type": "newtontr"},
    }
    
    solver = icepack.solvers.FlowSolver(model, **opts)





