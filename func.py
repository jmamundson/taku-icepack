#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:05:29 2024

@author: jason
"""

import numpy as np

import firedrake
from firedrake import exp, ln, inner, sqrt

from scipy.interpolate import interp1d

import icepack

import xarray


#%% basic constants parameters needed for the model

# Import some constants from icepack [NEED TO CHECK UNITS]
from icepack.constants import(
    ice_density as rho_ice,           # ρ_I | 9.21e-19
    water_density as rho_water,       # ρ_W | 1.03e-18
    weertman_sliding_law as weertman, # m   | 3.0
    gravity as g,                     # g   | 9.77e15
)


class constants:
    '''
    Global constants that should never need to be changed.
    '''
    
    def __init__(self):
        self.Temperature = firedrake.Constant(273.0) # -1 deg C
        self.A = icepack.rate_factor(self.Temperature) # flow rate factor
        self.C = firedrake.Constant(0.1) # friction coefficient
        self.hmin = 10 # minimum thickness [m]


class params:
    '''
    Scalar parameters.
    '''
    
    def __init__(self):
        self.n = 72 # number of grid points
        self.b_max = 4 # maximum mass balance rate [m a^{-1}]
        self.b_gradient = 0.01 # mass balance gradient
        # self.b_sea = -10 # mass balance rate at sea level [m a^{-1}]
        self.ELA = 800 # equilbrium line altitude [m]


#%% functions to specify bedrock elevation, initial thickness, and initial velocity

def bedrock(x, Q):
    '''
    Produces a longitudinal bedrock profile.

    Parameters
    ----------
    x : firedrake function that contains the longitudinal coordinates of the grid points
    Q : firedrake function space used for scalars

    Returns
    -------
    b : firedrake function that contains the bed elevation at the grid points
    tideLine : location where the bedrock intersects sea level

    '''
    zb = 1500*exp(-(x+5000)**2 / (2*15000**2)) - 400
    
    b = firedrake.interpolate(zb, Q)
    
    # determine where the bedrock intersects sea level
    bedInterpolator = interp1d(b.dat.data, x.dat.data)
    tideLine = bedInterpolator(0) 
    
    return(b, tideLine)    
    

def initial_thickness(x, Q):
    '''
    Prescribe a linear initial thickness profile.

    Parameters
    ----------
    x : firedrake function that contains the longitudinal coordinates of the grid points
    Q : firedrake function space used for scalars

    Returns
    -------
    h0 : firedrake function that contains the initial thickness at the grid points

    '''
    
    h_divide, h_terminus = 500, 200 # thickness at the divide and the terminus [m]
    h = h_divide - (h_divide-h_terminus) * x/x.dat.data[-1] # thickness profile [m]
    h0 = firedrake.interpolate(h, Q)

    return(h0)


def initial_velocity(x, V):
    '''
    Prescribe a linear initial velocity profile.

    Parameters
    ----------
    x : firedrake function that contains the longitudinal coordinates of the grid points
    Q : firedrake function space used for scalars

    Returns
    -------
    u0 : firedrake function that contains the initial velocity at the grid points

    '''
    
    u_divide, u_terminus = 0, 500 # velocity at the divide and the terminus [m/yr]
    u = u_divide + (u_terminus - u_divide) * x/x.dat.data[-1] # velocity profile
    u0 = firedrake.interpolate(u, V) # Vector function space

    return(u0)


def massBalance(s, Q, param):
    '''
    Mass balance profile. Returns function space for the mass balance.

    Parameters
    ----------
    s : firedrake function for the surface elevation profile
    Q : firedrake function space for the surface elevation profile
    param : class instance that contains: 
        param.b_max : maximum mass balance rate (at high elevations) [m a^{-1}]
        param.b_sea : mass balance rate at sea level [m a^{-1}]
        param.ELA : equilbrium line altitude [m]

    Returns
    -------
    a : firedrake function for the mass balance rate

    '''
    
    b_max = param.b_max
    b_gradient = param.b_gradient
    ELA = param.ELA
    
    k = 0.005 # smoothing factor
    # b_sea = -b_gradient*ELA
    
    z_threshold = b_max/b_gradient + ELA # location where unsmoothed balance rate switches from nonzero to zero
    
    a = firedrake.interpolate(b_gradient*((s-ELA) - 1/(2*k) * ln(1+exp(2*k*(s - z_threshold)))), Q)
    
    return a

#%% functions for re-meshing the model
def find_endpoint_haf(L, h, s, Q):
    '''
    Finds new glacier length using the height above flotation calving criteria.
    For some reason this only works if desired height above flotation is more 
    than about 5 m.

    Parameters
    ----------
    L : domain length [m]
    h : firedrake function for the thickness profile
    s : firedrake function for the surface elevation profile
    Q : firedrake scalar function space

    Returns
    -------
    L_new : new domain length [m]

    '''
    
    zb = firedrake.interpolate(s-h, Q) # glacier bed elevation [m]
    
    h_flotation = firedrake.interpolate(-rho_water/rho_ice*zb, Q) # thickness flotation (thickness if at flotation) [m]
    haf = firedrake.interpolate(h - h_flotation, Q) # height above flotation [m]

    haf_dat = haf.dat.data # height above flotation as a numpy array
    x_dat = np.linspace(0, L, len(h.dat.data), endpoint=True) # grid points as a numpy array [m]
    haf_interpolator = interp1d(haf_dat, x_dat, fill_value='extrapolate')
    
    haf_desired = 10 # desired height above flotation at the terminus
    L_new = haf_interpolator(haf_desired) # new domain length

    return L_new


def regrid(n, L, L_new, h, u):
    '''
    Create new mesh and interpolate functions onto the new mesh
    
    Parameters
    ----------
    n : number of grid points
    L : length of source function [m]
    L_new : length of destination function [m]
    h : thickness profile [m]
    u : velocity profile [m a^{-1}]
        
    Returns
    -------
    Q_new : destination scalar function space
    V_new : destination vector function space
    h_new : destination thickness profile [m]
    u_new : destination velocity profile [m a^{-1}]
    b_dest : destination bed profile [m]
    s_dest : destination surface profile [m]
    
    '''
    
    # Initialize mesh and mesh spaces
    mesh1d_new = firedrake.IntervalMesh(n, L_new)
    mesh_new = firedrake.ExtrudedMesh(mesh1d_new, layers=1, name="mesh")
    Q_new = firedrake.FunctionSpace(mesh_new, "CG", 2, vfamily="R", vdegree=0)
    V_new = firedrake.FunctionSpace(mesh_new, "CG", 2, vfamily="GL", vdegree=2, name='velocity')
    

    x_sc, z_sc = firedrake.SpatialCoordinate(mesh_new)
    x = firedrake.interpolate(x_sc, Q_new)
    z = firedrake.interpolate(z_sc, Q_new)
    
    b_new, _ = bedrock(x, Q_new)

    # Remesh thickness and velocity
    h_new =  adjust_function(h, L, L_new, Q_new)
    u_new =  adjust_function(u, L, L_new, V_new)

        
    s_new = icepack.compute_surface(thickness = h_new, bed = b_new) 
    
    return Q_new, V_new, h_new, u_new, b_new, s_new, mesh_new



def adjust_function(func_src, L, L_new, funcspace_dest):
    '''
    Shrinks function spaces.

    Parameters
    ----------
    func_src : source function
    L : length of source function [m]
    L_new : length of destination function [m]
    funcspace_dest : destination function

    Returns
    -------
    func_dest : destination function

    '''
    
    # extract data from func_src
    data = func_src.dat.data
    
    # need to do this differently if the function space is a scalar or vector
    if funcspace_dest.topological.name=='velocity':
        data_len = int(len(data)/3)
    else:
        data_len = len(data)
    
    x = np.linspace(0, L, data_len, endpoint=True) # old grid points
    x_new = np.linspace(0, L_new, data_len, endpoint=True) # new grid points
    
    # Interpolate data from x to x_new
    if funcspace_dest.topological.name=='velocity':
        # velocity is defined at three vertical grid points, and the data is ordered
        # [bottom, middle, top, bottom, middle, top, ...]
        #
        # Not sure how to use icepack.interpolate to create this 2D grid.
        # Instead this hack just uses the mean velocity. This is fine since
        # for plotting purposes I only want the the mean velocity, and the mean
        # velocity is a good starting point for the next iteration.
        
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

    func_dest.dat.data[-2:] = temp_dest[-2:] # sometimes last grid points aren't correctly interpolated [WHY?]
    
    return func_dest


#%%
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


