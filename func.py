# sediment model not working with hillslope diffusion if grid size too small?
# hillslope diffusion not allowing for erosion?
# need to increase water flux?
# model behaves better with height above flotation criterion (why?) or just use floatation criteria for initial spin up?
# if time step is too large get negative sediment thickness
# main issues with sediment model:
#   1. kink in bed slope at bedrock-sediment transition
#   2. too much deposition at the terminus
#   3. "oscillating" behavior if grid size too small    


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

# Import some constants from icepack
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
        self.Temperature = firedrake.Constant(273.15) # temperate ice = 0 deg C
        self.A = icepack.rate_factor(self.Temperature) # flow rate factor
        self.C = firedrake.Constant(0.2) # friction coefficient [good value here?]
        self.hmin = 10 # minimum thickness [m]

constant = constants()

class params:
    '''
    Scalar parameters.
    '''
    
    def __init__(self):
        self.n = 100 # number of grid points
        self.dt = 0.1 # time step size [a]
        self.L = 40e3 # initial domain length [m]
        self.sedDepth = 50 # depth to sediment, from sea level, prior to advance scenario [m]
        self.b_max = 4 # maximum mass balance rate [m a^{-1}]
        self.b_gradient = 0.01 # mass balance gradient
        self.ELA = 1000 # equilbrium line altitude [m]

param = params()

class paramsSed:
    '''
    Parameters for the sediment transport model.
    '''
    
    def __init__(self):
        self.h_eff = 0.1 # effective water thickness [m]
        self.c = 1.5e-12 # Brinkerhoff used 2e-12
        self.w = 500 # settling velocity [m a^{-1}]; Brinkerhoff used 500
        self.k = 5000 # sediment diffusivity; Brinkerhoff used 5000

paramSed = paramsSed()


#%% functions to specify bedrock elevation, initial thickness, and initial velocity
def bedrock(x, **kwargs):
    '''
    Returns a longitudinal bedrock profile.

    Parameters
    ----------
    x : firedrake function that contains the longitudinal coordinates of the grid points
    Q : firedrake function space used for scalars

    Returns
    -------
    b : firedrake function that contains the bed elevation at the grid points
    tideLine : location where the bedrock intersects sea level
    '''
    
    if "Q" in kwargs:
        Q = kwargs["Q"]
        b = 2500*exp(-(x+5000)**2 / (2*15000**2)) - 400
        b = firedrake.interpolate(b, Q)
    
        # determine where the bedrock intersects sea level
        bedInterpolator = interp1d(b.dat.data, x.dat.data, kind='linear')
    
    else:
        b = 2500*np.exp(-(x+5000)**2 / (2*15000**2)) - 400
        bedInterpolator = interp1d(b, x, kind='linear')
    
    tideLine = bedInterpolator(0) 
    
    return(b, tideLine)    
    

def width(x, **kwargs):
    '''
    Returns the valley width as a function of longitudinal position. 

    Parameters
    ----------
    x : firedrake function that contains the longitudinal coordinates of the grid points
    Q : firedrake function space used for scalars

    Returns
    -------
    W : firedrake function that contains the valley width
    '''
    
    
    k = 5 # smoothing parameter
    x_step = 10e3 # location of step [m]
    w_fjord = 2e3 # fjord width [m]
    w_max = w_fjord + 3e3 # maximum valley width [m]
    
    if "Q" in kwargs: # create firedrake function
        Q = kwargs["Q"]
        # using a logistic function, which makes the fjord walls roughly parallel
        w_array = w_max * (1 - 1 / (1+exp(-k*(x-x_step)/x_step))) + w_fjord
        
        w = firedrake.interpolate(w_array, Q)
    
    else:
        w = w_max * (1 - 1 / (1+np.exp(-k*(x-x_step)/x_step))) + w_fjord
        
        
    return(w)


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
    
    h_divide, h_terminus = 500, 300 # thickness at the divide and the terminus [m]
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


def massBalance(x, s, u, h, w, param):
    '''
    Mass balance profile. Returns function space for the "adjusted" mass 
    balance rate. For a flow line model,
    
    dh/dt = a - 1/w * d/dx(uhw) = a - d/dx(uh) - uh/w * dw/dx
    
    This code combines a - uh/w * dw/dx into one term, the adjusted mass
    balance rate, so that the prognostic solver doesn't need to worry about
    variations in glacier width.

    Parameters
    ----------
    x, s, u, h, w : firedrake functions for the longitudinal coordinates,
        surface elevation, velocity, thickness, and width
    param : class instance that contains: 
        param.b_max : maximum mass balance rate (at high elevations) [m a^{-1}]
        param.b_gradient : mass balance gradient [m a^{1} / m]
        param.ELA : equilbrium line altitude [m]

    Returns
    -------
    a, a_mod : firedrake functions for the mass balance rate and adjusted mass 
        balance rate
    '''
    
    b_max = param.b_max
    b_gradient = param.b_gradient
    ELA = param.ELA
    
    k = 0.005 # smoothing factor
        
    z_threshold = b_max/b_gradient + ELA # location where unsmoothed balance rate switches from nonzero to zero

    # unmodified mass balance rate    
    a = firedrake.interpolate(b_gradient*((s-ELA) - 1/(2*k) * ln(1+exp(2*k*(s - z_threshold)))), s.function_space())
    
    # width gradient
    dwdx = icepack.interpolate(w.dx(0), w.function_space())
    
    # modified mass balance rate
    a_mod = icepack.interpolate(a-u*h/w*dwdx, h.function_space())
    
    return a, a_mod



#%% functions for re-meshing the model
def find_endpoint_massflux(L, x, a, u, h, w, dt):
    '''
    Finds the new glacier length using the mass-flux calving parameterization
    of Amundson (2016) and Amundson and Carroll (2018).

    Parameters
    ----------
    L : current glacier length [m]
    x, a, u, h, w : firedrake functions for the longitudinal position, mass 
        balance rate, velocity, thickness, and width
    dt : time step size [a]

    Returns
    -------
    L_new : new glacier length [m]
    '''
    
    alpha = 1.15 # frontal ablation parameter
    
    index = np.argsort(x.dat.data)
    xGlacier = x.dat.data[index]
    balanceRate = a.dat.data[index]
    width = w.dat.data[index]
    
    hGlacier = h.dat.data[index]
    uGlacier = icepack.depth_average(u).dat.data[index]
    Ut = uGlacier[-1] # terminus velocity [m a^{-1}]
    
    Qb = np.trapz(balanceRate*width, dx=xGlacier[1]) # terminus balance flux [m^3 a^{-1}]
    Ub = Qb/(hGlacier[-1]*width[-1]) # terminus balance velocity [m a^{-1}]
    
    dLdt = (alpha-1)*(Ub-Ut)
    
    L_new = L + dLdt*dt
    
    return L_new


def find_endpoint_haf(L, h, s):
    '''
    Finds new glacier length using the height above flotation calving criteria.
    For some reason this only works if desired height above flotation is more 
    than about 5 m.

    Parameters
    ----------
    L : domain length [m]
    h, s : firedrake functions for the thickness and surface elevation

    Returns
    -------
    L_new : new domain length [m]

    '''
    
    zb = firedrake.interpolate(s-h, s.function_space()) # glacier bed elevation [m]
    
    h_flotation = firedrake.interpolate(-rho_water/rho_ice*zb, s.function_space()) # thickness flotation (thickness if at flotation) [m]
    haf = firedrake.interpolate(h - h_flotation, s.function_space()) # height above flotation [m]

    haf_dat = haf.dat.data # height above flotation as a numpy array
    x_dat = np.linspace(0, L, len(h.dat.data), endpoint=True) # grid points as a numpy array [m]
    haf_interpolator = interp1d(haf_dat, x_dat, kind='linear', fill_value='extrapolate')
    
    haf_desired = 10 # desired height above flotation at the terminus
    L_new = haf_interpolator(haf_desired) # new domain length

    return L_new


def regrid(n, x, L, L_new, h, u, sed):
    '''
    Create new mesh and interpolate functions onto the new mesh
    
    Parameters
    ----------
    n : number of grid points
    L : length of source function [m]
    L_new : length of destination function [m]
    h, u : firedrake thickness and velocity functions [m; m a^{-1}]
    sed : instance of the sediment model class
        
    Returns
    -------
    Q, V : destination scalar and vector function spaces
    x, h, u, b, s, w : destination position, thickness, velocity, bed, surface, and width functions
    mesh : destination mesh
    '''
    
    x_old = x # need to keep previous position function
    
    # Initialize mesh and mesh spaces
    mesh1d = firedrake.IntervalMesh(n, L_new)
    mesh = firedrake.ExtrudedMesh(mesh1d, layers=1, name="mesh")
    Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
    V = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="GL", vdegree=2, name='velocity')
    

    x_sc, z_sc = firedrake.SpatialCoordinate(mesh)
    x = firedrake.interpolate(x_sc, Q)
    # z = firedrake.interpolate(z_sc, Q)
    
    
    zBed_interpolator = interp1d(sed.x, sed.H+sed.zBedrock, kind='linear') # create interpolator to put bed elevation into a firedrake function
    zBed_xarray = xarray.DataArray(zBed_interpolator(x.dat.data), [x.dat.data], 'x')
    b = icepack.interpolate(zBed_xarray, Q)

    # remesh thickness and velocity
    h =  adjust_function(h, x_old, L, L_new, Q)
    u =  adjust_function(u, x_old, L, L_new, V)

    # compute new surface based on thickness and bed; this determines whether
    # or not the ice is floating    
    s = icepack.compute_surface(thickness = h, bed = b) 
    
    w = width(x, Q=Q)
    
    return Q, V, h, u, b, s, w, mesh, x



def adjust_function(func_src, x, L, L_new, funcspace_dest):
    '''
    Shrinks function spaces.

    Parameters
    ----------
    func_src : source function
    x : old position function [m]
    L : length of source function [m]
    L_new : length of destination function [m]
    funcspace_dest : destination function

    Returns
    -------
    func_dest : destination function

    '''
    
    x = x.dat.data
    index = np.argsort(x)
    x = x[index]
        
    # extract data from func_src
    data = func_src.dat.data
    
    # need to do this differently if the function space is a scalar or vector
    if funcspace_dest.topological.name=='velocity':
        data_len = int(len(data)/3)
    else:
        data_len = len(data)
    
    x = np.linspace(0, L, data_len, endpoint=True)
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
        u_ = u_[index]
        
        data_interpolator = interp1d(x, u_, kind='linear', fill_value='extrapolate')
        data_new = data_interpolator(x_new)
        
    else:
        data_interpolator = interp1d(x, data[index], kind='linear', fill_value='extrapolate')
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

    U_0 = firedrake.Constant(100)
    
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


def side_drag(w):
    '''
    Calculate the drag coefficient for side drag, for the equation:
    1/W*(4/(A*W))^{1/3} * |u|^{-2/3}*u
    
    Currently assumes that the width is constant. Later may adjust this to pass 
    in a width function.
    
    Parameters
    ----------
    h : firedrake thickness function; only used to determine the function space
    
    Returns
    -------
    Cs : lateral drag coefficient
    
    '''
    
    Cs = firedrake.interpolate( 1/w * (4/(constant.A*w))**(1/3), w.function_space())

    return Cs

#%% sediment transport model
# be careful with CFL condition?
# bed function not updating correctly?

class sedModel:

    def __init__(self, L, sedDepth):
        '''
        Create grid and initial values of the sediment model.
    
        Parameters
        ----------
        L : domain length [m]
        sedDepth = depth to sediment from sea level [m]
    
        '''
        
        self.x = np.linspace(0, L, 50, endpoint=True)
        
        self.zBedrock, _ = bedrock(self.x)
        
        # determine x-location of start of sediment
        bedInterpolator = interp1d(self.zBedrock, self.x, kind='linear')
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

    

    def sedTransportImplicit(self, x, h, a, b, u, Q, dt):
        
        dx = np.sort(x.dat.data)[1]
        xGlacier = np.arange(0, param.L, dx)
        H_guess = np.ones(len(xGlacier))
        
        result = root(self.__sedTransportImplicit2, H_guess, (x, h, a, b, u, Q, dt), method='hybr', options={'maxfev':int(1e6)})#, 'xtol':1e-12})
        
        # result = root(self.__sedTransportImplicit, self.H, (x, h, a, b, u, Q, dt), method='hybr', options={'maxfev':int(1e6)})#, 'xtol':1e-12})
        self.H = result.x
        # self.H[self.H<0] = 0 # temporary hack to keep thickness greater than 0!
        
        index = np.argsort(x.dat.data)
        xGlacier = x.dat.data[index]
        
        zBed = self.H+self.zBedrock # new bed elevation at sediment model grid points
        zBed_interpolator = interp1d(self.x, zBed, kind='linear') # create interpolator to put bed elevation into a firedrake function
    
        zBed_xarray = xarray.DataArray(zBed_interpolator(xGlacier), [xGlacier], 'x')
        # zBed_xarray = xarray.DataArray(zBed, [xGlacier], 'x')
        
        b = icepack.interpolate(zBed_xarray, Q)
        
        return b
    

    def __sedTransportImplicit(self, H_guess, x, h, a, b, u, Q, dt):
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
        bedGlacier = b.dat.data[index]
        uGlacier = icepack.depth_average(u).dat.data[index]
    
        # mask out areas where the ice is less than 10 m thick, then extract melt rate
        # at glacier x-coordinates
        iceMask = icepack.interpolate(conditional(h<=constant.hmin+1e-4, 0, 1), Q)
        # meltRate = icepack.interpolate(conditional(a>0, 0, -a), Q)
        runoff = icepack.interpolate(param.b_max-a, Q)
        runoff = icepack.interpolate(runoff*iceMask, Q)
        runoff = runoff.dat.data[index]
    
        runoffInterpolator = interp1d(xGlacier, runoff, fill_value=np.array([0]), kind='cubic', bounds_error=False)
        runoff = runoffInterpolator(self.x)
    
        w = width(self.x)
        
        # determine subglacial charge per unit width on the sediment model grid points
        dx = self.x[1] 
        self.Qw = cumtrapz(runoff*w, dx=dx, initial=0)/w # subglacial discharge, calculated from balance rate

        # delta_s = [np.min((x, 1)) for x in self.H] # erosion goes to 0 if sediment thickness is 0
        delta_s = (1-np.exp(-self.H)) # similar to what Brinkerhoff has in code?
        
        # effective thickness is water depth if not glacier at the grid point
        # allows for some small erosion in front of the glacier, especially if the water gets shallow
        h_eff = paramSed.h_eff*np.ones(len(self.x)) 
        
        index = self.x>np.max(xGlacier)
        h_eff[index] = -(self.H[index]+self.zBedrock[index]) 
        
        self.erosionRate = paramSed.c * self.Qw**2/h_eff**3 * delta_s
        # self.erosionRate[self.x > xGlacier[-1]] = 0 # no erosion in front of the glacier; only works for tidewater, but okay because no erosion when land-terminating
    
    
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
    
    
        # test higher water flux in the fjord
        #self.Qw[self.x > xGlacier[-1]] = 2*self.Qw[self.x > xGlacier[-1]]
        self.Qw[self.x > xGlacier[-1]] += -bedGlacier[-1]*uGlacier[-1]*0.5 # add more buoyancy forcing?
        
        # end test
        
        self.depositionRate = np.zeros(len(Qs))
        self.depositionRate[self.Qw>0] = paramSed.w*Qs[self.Qw>0]/self.Qw[self.Qw>0]
    
        zBed = self.H+self.zBedrock # elevation of glacier bed from previous time step
    
        # calculate bed curvature for hillslope diffusion; dz^2/dx^2(zBed)
        zBed_curvature = (zBed[2:]-2*zBed[1:-1]+zBed[:-2])/dx**2
        zBed_left = (zBed[2]-2*zBed[1]+zBed[0])/dx**2
        zBed_right = (zBed[-1]-2*zBed[-2]+zBed[-3])/dx**2
    
        zBed_curvature = np.concatenate(([zBed_left], zBed_curvature, [zBed_right]))
    
        # self.hillslope = paramSed.k*(1-np.exp(-self.H/10))*zBed_curvature*np.sign(self.H)
        # redefine delta_a
        delta_s = (1-np.exp(-self.H/10))
        self.hillslope = paramSed.k*zBed_curvature*delta_s
        
        
        self.dHdt = self.depositionRate - self.erosionRate + self.hillslope
        RHS_new = self.dHdt
    
        theta = 1 # theta = 1 => backwards euler; theta = 0 => forward euler; theta = 0.5 => Crank-Nicolson
        # backwards euler seems to work best
        res = H_guess - H_old - dt*(theta*RHS_new + (1-theta)*RHS_old)
        
        return(res)

    def __sedTransportImplicit2(self, H_guess, x, h, a, b, u, Q, dt):
        # use Crank-Nicolson method; need to use minimization because right hand side depends on H (sediment thickness)
        
        
        
        # don't need minimization??? or not implemented correctly?
        # RHS new should depend on H_new, but it doesn't, I don't think
        
        
        # compute new values
        # create numpy array of glacier x-coordinates; make sure they are properly 
        # sorted
        index = np.argsort(x.dat.data)
        xGlacier = x.dat.data[index]
        bedGlacier = b.dat.data[index]
        uGlacier = icepack.depth_average(u).dat.data[index]
    
        # mask out areas where the ice is less than 10 m thick, then extract melt rate
        # at glacier x-coordinates
        iceMask = icepack.interpolate(conditional(h<=constant.hmin+1e-4, 0, 1), Q)
        # meltRate = icepack.interpolate(conditional(a>0, 0, -a), Q)
        runoff = icepack.interpolate(param.b_max-a, Q)
        runoff = icepack.interpolate(runoff*iceMask, Q)
        runoff = runoff.dat.data[index]
    
        dx = xGlacier[1]
    
        
        
        # fill value needs some thought!
        sedH_interpolator = interp1d(self.x, self.H, kind='linear', fill_value='extrapolate')
        sedBedrock_interpolator = interp1d(self.x, self.zBedrock, kind='linear', fill_value='extrapolate')   
        
        self.x = np.arange(0, param.L, dx)
        self.H = sedH_interpolator(self.x) # sediment thickness on extended glacier grid
        H_old = self.H
        self.zBedrock, _ = bedrock(self.x) # sediment thickness on extended glacier grid 
        w = width(self.x)
    
        # determine subglacial charge per unit width on the glacier model grid points
        # first need to extend the runoff to the end of the domain
        runoff = np.concatenate((runoff, np.zeros(len(self.x)-len(runoff))))
        
        self.Qw = cumtrapz(runoff*w, dx=dx, initial=0)/w # subglacial discharge, calculated from balance rate
    
        # delta_s = [np.min((x, 1)) for x in self.H] # erosion goes to 0 if sediment thickness is 0
        delta_s = (1-np.exp(-self.H)) # similar to what Brinkerhoff has in code?
        
        # effective thickness is water depth if not glacier at the grid point
        # allows for some small erosion in front of the glacier, especially if the water gets shallow
        h_eff = paramSed.h_eff*np.ones(len(self.x)) 
        
        index = self.x>np.max(xGlacier)
        h_eff[index] = -(self.H[index]+self.zBedrock[index]) 
        
        self.erosionRate = paramSed.c * self.Qw**2/h_eff**3 * delta_s
        # self.erosionRate[self.x > xGlacier[-1]] = 0 # no erosion in front of the glacier; only works for tidewater, but okay because no erosion when land-terminating
    
    
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
    
    
        # test higher water flux in the fjord
        self.Qw[self.x > xGlacier[-1]] = 5*self.Qw[self.x > xGlacier[-1]]
        # self.Qw[self.x > xGlacier[-1]] += -bedGlacier[-1]*uGlacier[-1]*0.5 # add more buoyancy forcing?
        
        # end test
        
        self.depositionRate = np.zeros(len(Qs))
        self.depositionRate[self.Qw>0] = paramSed.w*Qs[self.Qw>0]/self.Qw[self.Qw>0]
    
        zBed = self.H+self.zBedrock # elevation of glacier bed from previous time step
    
        # calculate bed curvature for hillslope diffusion; dz^2/dx^2(zBed)
        zBed_curvature = (zBed[2:]-2*zBed[1:-1]+zBed[:-2])/dx**2
        zBed_left = (zBed[2]-2*zBed[1]+zBed[0])/dx**2
        zBed_right = (zBed[-1]-2*zBed[-2]+zBed[-3])/dx**2
    
        zBed_curvature = np.concatenate(([zBed_left], zBed_curvature, [zBed_right]))
    
        # self.hillslope = paramSed.k*(1-np.exp(-self.H/10))*zBed_curvature*np.sign(self.H)
        # redefine delta_a
        delta_s = (1-np.exp(-self.H/10))
        self.hillslope = paramSed.k*zBed_curvature*delta_s
        
        self.dHdt = self.depositionRate - self.erosionRate + self.hillslope
        RHS_new = self.dHdt
    
        # theta = 1 # theta = 1 => backwards euler; theta = 0 => forward euler; theta = 0.5 => Crank-Nicolson
        # backwards euler seems to work best
        # res = H_guess - H_old - dt*(theta*RHS_new + (1-theta)*RHS_old)
        res = H_guess - H_old - dt*RHS_new
        
        return(res)
