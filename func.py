# sediment model not working with hillslope diffusion if grid size too small?
# hillslope diffusion not allowing for erosion?
# need to increase water flux?

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:05:29 2024

@author: jason
"""

import numpy as np

import firedrake
from firedrake import exp, ln, inner, sqrt, conditional

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy import sparse

import icepack

import xarray
from scipy.optimize import root

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
        self.Temperature = firedrake.Constant(273.15) # -1 deg C
        self.A = icepack.rate_factor(self.Temperature) # flow rate factor
        self.C = firedrake.Constant(0.1) # friction coefficient
        self.hmin = 10 # minimum thickness [m]

constant = constants()

class params:
    '''
    Scalar parameters.
    '''
    
    def __init__(self):
        self.n = 200 # number of grid points
        self.L = 40e3 # initial domain length [m]
        self.sedDepth = 50 # depth to sediment, from sea level, prior to advance scenario [m]
        self.b_max = 4 # maximum mass balance rate [m a^{-1}]
        self.b_gradient = 0.01 # mass balance gradient
        # self.b_sea = -10 # mass balance rate at sea level [m a^{-1}]
        self.ELA = 600 # equilbrium line altitude [m]

param = params()

class paramsSed:
    '''
    Parameters for the sediment transport model.
    '''
    
    def __init__(self):
        self.h_eff = 0.1 # effective water thickness [m]
        self.c = 2e-12
        self.w = 500 # settling velocity [m a^{-1}]
        self.k = 5000 # sediment diffusivity

paramSed = paramsSed()


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
def find_endpoint_massflux(L, x, a, u, h, dt):
    
    alpha = 1.15
    
    index = np.argsort(x.dat.data)
    xGlacier = x.dat.data[index]
    balanceRate = a.dat.data[index]
    
    hGlacier = h.dat.data[index]
    uGlacier = icepack.depth_average(u).dat.data[index]
    Ut = uGlacier[-1]
    
    Qb = cumtrapz(balanceRate, dx=xGlacier[1], initial=0)
    Ub = Qb[-1]/hGlacier[-1]
    
    dLdt = (alpha-1)*(Ub-Ut)
    
    L_new = L + dLdt*dt
    
    return L_new


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


def regrid(n, L, L_new, h, u, sed):
    '''
    Create new mesh and interpolate functions onto the new mesh
    
    Parameters
    ----------
    n : number of grid points
    L : length of source function [m]
    L_new : length of destination function [m]
    h : thickness profile [m]
    u : velocity profile [m a^{-1}]
    sed : instance of the sediment model class
        
    Returns
    -------
    Q : destination scalar function space
    V : destination vector function space
    h : destination thickness profile [m]
    u : destination velocity profile [m a^{-1}]
    b : destination bed profile [m]
    s : destination surface profile [m]
    mesh : destination mesh
    x : destination x function
    '''
    
    # Initialize mesh and mesh spaces
    mesh1d = firedrake.IntervalMesh(n, L_new)
    mesh = firedrake.ExtrudedMesh(mesh1d, layers=1, name="mesh")
    Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
    V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2, name='velocity')
    

    x_sc, z_sc = firedrake.SpatialCoordinate(mesh)
    x = firedrake.interpolate(x_sc, Q)
    # z = firedrake.interpolate(z_sc, Q)
    
    # b, _ = bedrock(x, Q)

    
    zBed_interpolator = interp1d(sed.x, sed.H+sed.zBedrock) # create interpolator to put bed elevation into a firedrake function
    zBed_xarray = xarray.DataArray(zBed_interpolator(x.dat.data), [x.dat.data], 'x')
    b = icepack.interpolate(zBed_xarray, Q)

    # Remesh thickness and velocity
    h =  adjust_function(h, L, L_new, Q)
    u =  adjust_function(u, L, L_new, V)

        
    s = icepack.compute_surface(thickness = h, bed = b) 
    
    return Q, V, h, u, b, s, mesh, x



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
    
    # index = np.argsort(x.dat.data)
    # x = x.dat.data[index]
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
        # u_ = u_[index]
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
    
    U = sqrt(inner(u,u))
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


#%% sediment transport model
# be careful with CFL condition?
# bed function not updating correctly?

class sedModel:

    def __init__(self, L, sedDepth):
        '''
        Create grid and initial values of sediment model
    
        Parameters
        ----------
        L : domain length [m]
        sedDepth = depth to sediment from sea level [m]
    
        '''
        self.x = np.linspace(0, L, 101, endpoint=True)
        self.zBedrock = 1500*np.exp(-(self.x+5000)**2 / (2*15000**2)) - 400 # make the same as bedrock
    
        # determine x-location of start of sediment
        bedInterpolator = interp1d(self.zBedrock, self.x)
        sedLevelInit = bedInterpolator(-sedDepth) # sediment fills fjord to 50 m depth
    
        # initial sediment thickness
        self.H = np.zeros(len(self.zBedrock)) # sediment thickness
        self.H[self.x>sedLevelInit] = -sedDepth-self.zBedrock[self.x>sedLevelInit]
    
        # smooth_length = 5
        # box = np.ones(smooth_length)/smooth_length
        # tmp = np.convolve(box, self.H, mode='valid')
        # self.H = np.concatenate((self.H[:2], tmp, self.H[-2:])) # need to be careful with length
    
        self.erosionRate = np.zeros(len(self.x))
        self.depositionRate = np.zeros(len(self.x))


    def sedTransportExplicit(self, x, h, a, Q, dt):
        '''
        Calculate new bed elevation. 
    
        Parameters
        ----------
        x : function containing longitudinal position of glacier grid coordinates
        h : firedrake function for the glacier thickness
        a : firedrake function for the mass balance rate
        xBed : longitudinal coordinates of bed model [m]
        zBedrock : bedrock elevation of bed model [m]
        Hsediment : sediment thickness [m]
        dt : time step size [a]
    
        Returns
        -------
        b : firedrak function for the bed elevation
        Hsediment : new sediment thickness in the bed model
        '''
    
        # create numpy array of glacier x-coordinates; make sure they are properly 
        # sorted
        index = np.argsort(x.dat.data)
        xGlacier = x.dat.data[index]
    
        # mask out areas where the ice is less than 10 m thick, then extract melt rate
        # at glacier x-coordinates
        iceMask = icepack.interpolate(conditional(h<=constant.hmin+1e-4, 0, 1), Q)
        # meltRate = icepack.interpolate(conditional(a>0, 0, -a), Q)
        meltRate = icepack.interpolate(param.b_max-a, Q)
        meltRate = meltRate.dat.data[index]
    
        meltInterpolator = interp1d(xGlacier, meltRate, fill_value=np.array([0]), bounds_error=False)
        meltBed = meltInterpolator(self.x)
    
        # determine subglacial charge on the sediment model grid points
        dx = self.x[1] 
        self.Qw = cumtrapz(meltBed, dx=dx, initial=0) # subglacial discharge, calculated from balance rate

        
        delta_s = [np.min((x, 1)) for x in self.H] # erosion goes to 0 if sediment thickness is 0
        self.erosionRate = paramSed.c * self.Qw**2/paramSed.h_eff**3 * delta_s
        self.erosionRate[self.x > xGlacier[-1]] = 0 # no erosion in front of the glacier; only works for tidewater, but okay because no erosion when land-terminating
   

        # solve for sediment transport with left-sided difference
        d = self.Qw + dx*paramSed.w
        d[0] = 1
        d[-1] = 1
        
        d_left = -self.Qw[1:]
        d_left[-1] = 0
        
        diagonals = [d_left, d]
        D = sparse.diags(diagonals, [-1,0]).toarray()
        
        f = dx*self.Qw*self.erosionRate
        f[0] = 0
        f[1] = 0
        
        Qs = np.linalg.solve(D,f)
        self.Qs = Qs
    
        self.depositionRate = np.zeros(len(Qs))
        self.depositionRate[self.Qw>0] = paramSed.w*Qs[self.Qw>0]/self.Qw[self.Qw>0]
    
        zBed = self.H+self.zBedrock # elevation of glacier bed from previous time step
    
        # calculate bed curvature for hillslope diffusion; dz^2/dx^2(zBed)
        zBed_curvature = (zBed[2:]-2*zBed[1:-1]+zBed[:-2])/dx**2
        zBed_left = (zBed[2]-2*zBed[1]+zBed[0])/dx**2
        zBed_right = (zBed[-1]-2*zBed[-2]+zBed[-3])/dx**2
    
        zBed_curvature = np.concatenate(([zBed_left], zBed_curvature, [zBed_right]))
    
        # self.hillslope = paramSed.k*(1-np.exp(-self.H/10))*zBed_curvature*np.sign(self.H)
        self.hillslope = 0*paramSed.k*zBed_curvature*delta_s
        
        
        self.dHdt = self.depositionRate - self.erosionRate + self.hillslope
        self.H = self.H + self.dHdt*dt # + paramSed.k*zBed_curvature)*dt # new sediment thickness
    
        self.H = self.H + self.hillslope
        self.H[self.H<0] = 0 # can't erode bedrock
    
        zBed = self.H+self.zBedrock # new bed elevation at sediment model grid points
    
        zBed_interpolator = interp1d(self.x, zBed) # create interpolator to put bed elevation into a firedrake function
    
        zBed_xarray = xarray.DataArray(zBed_interpolator(xGlacier), [xGlacier], 'x')
        b = icepack.interpolate(zBed_xarray, Q)
        
        return b
    

    def sedTransportImplicit(self, x, h, a, Q, dt):
        result = root(self.__sedTransportImplicit, self.H, (x, h, a, Q, dt), method='hybr', options={'maxfev':int(1e6)})#, 'xtol':1e-12})
        self.H = result.x
        
        
        index = np.argsort(x.dat.data)
        xGlacier = x.dat.data[index]
        
        zBed = self.H+self.zBedrock # new bed elevation at sediment model grid points
        zBed_interpolator = interp1d(self.x, zBed) # create interpolator to put bed elevation into a firedrake function
    
        zBed_xarray = xarray.DataArray(zBed_interpolator(xGlacier), [xGlacier], 'x')
        b = icepack.interpolate(zBed_xarray, Q)
        
        return b
    

    def __sedTransportImplicit(self, H_guess, x, h, a, Q, dt):
        # use Crank-Nicolson method; need to use minimization because right hand side depends on H (sediment thickness)
        
        # extract and temporarily keep the old values
        H_old = self.H
        
        
        # clunky way of dealing with first time step, for now
        if hasattr(self,'hillslope')==True:
            edot_old = self.erosionRate
            ddot_old = self.depositionRate
            hill_old = self.hillslope
        else:
            edot_old = np.zeros(len(self.x))
            ddot_old = np.zeros(len(self.x))
            hill_old = np.zeros(len(self.x))
    
        RHS_old = ddot_old - edot_old + hill_old
        
        
        # compute new values
        # create numpy array of glacier x-coordinates; make sure they are properly 
        # sorted
        index = np.argsort(x.dat.data)
        xGlacier = x.dat.data[index]
    
        # mask out areas where the ice is less than 10 m thick, then extract melt rate
        # at glacier x-coordinates
        iceMask = icepack.interpolate(conditional(h<=constant.hmin+1e-4, 0, 1), Q)
        # meltRate = icepack.interpolate(conditional(a>0, 0, -a), Q)
        meltRate = icepack.interpolate(param.b_max-a, Q)
        meltRate = icepack.interpolate(iceMask*meltRate, Q)
        meltRate = meltRate.dat.data[index]
    
        meltInterpolator = interp1d(xGlacier, meltRate, fill_value=np.array([0]), bounds_error=False)
        meltBed = meltInterpolator(self.x)
    
        # determine subglacial charge on the sediment model grid points
        dx = self.x[1] 
        self.Qw = cumtrapz(meltBed, dx=dx, initial=0) # subglacial discharge, calculated from balance rate
        
        delta_s = [np.min((x, 1)) for x in self.H] # erosion goes to 0 if sediment thickness is 0
        self.erosionRate = paramSed.c * self.Qw**2/paramSed.h_eff**3 * delta_s
        self.erosionRate[self.x > xGlacier[-1]] = 0 # no erosion in front of the glacier; only works for tidewater, but okay because no erosion when land-terminating
    
    
        # solve for sediment transport with left-sided difference
        d = self.Qw + dx*paramSed.w
        d[0] = 1
        d[-1] = 1
        
        d_left = -self.Qw[1:]
        d_left[-1] = 0
        
        diagonals = [d_left, d]
        D = sparse.diags(diagonals, [-1,0]).toarray()
        
        f = dx*self.Qw*self.erosionRate
        f[0] = 0
        f[1] = 0
        
        Qs = np.linalg.solve(D,f)
        self.Qs = Qs
    
        self.depositionRate = np.zeros(len(Qs))
        self.depositionRate[self.Qw>0] = paramSed.w*Qs[self.Qw>0]/self.Qw[self.Qw>0]
    
        zBed = self.H+self.zBedrock # elevation of glacier bed from previous time step
    
        # calculate bed curvature for hillslope diffusion; dz^2/dx^2(zBed)
        zBed_curvature = (zBed[2:]-2*zBed[1:-1]+zBed[:-2])/dx**2
        zBed_left = (zBed[2]-2*zBed[1]+zBed[0])/dx**2
        zBed_right = (zBed[-1]-2*zBed[-2]+zBed[-3])/dx**2
    
        zBed_curvature = np.concatenate(([zBed_left], zBed_curvature, [zBed_right]))
    
        # self.hillslope = paramSed.k*(1-np.exp(-self.H/10))*zBed_curvature*np.sign(self.H)
        self.hillslope = paramSed.k*zBed_curvature*delta_s
        
        
        self.dHdt = self.depositionRate - self.erosionRate + self.hillslope
        RHS_new = self.dHdt
    
        
        res = H_guess - H_old - dt/2*(RHS_new + RHS_old)
        
        return(res)
