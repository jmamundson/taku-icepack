# sediment model not working with hillslope diffusion if grid size too small?
# hillslope diffusion not allowing for erosion?
# need to increase water flux?
# model behaves better with height above flotation criterion (why?) or just use floatation criteria for initial spin up?
# if time step is too large get negative sediment thickness
# main issues with sediment model:
#   1. kink in bed slope at bedrock-sediment transition
#   2. too much deposition at the terminus
#   3. "oscillating" behavior if grid size too small    
# someway to reduce erosion if too much curvature? maybe make hs larger?
# make sure no sediment is deposited up stream
# cite Creyts et al (2013): Evolution of subglacial overdeepenings in response to sediment redistribution and glaciohydraulic supercooling
# also Delaney paper on sugset_2d; discuss supply limited vs transport limited regimes
# linear interpolation for height above flotation
# to do: create glacier class
# need to be careful with interpolators

import numpy as np

import firedrake
from firedrake import exp, ln, inner, sqrt, conditional, dx, assemble

from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy import sparse

import icepack

import xarray
from scipy.optimize import root

from matplotlib import pyplot as plt

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
        self.C = firedrake.Constant(0.1) # friction coefficient
        self.U0 = firedrake.Constant(500) # threshold velocity for friction [m a^{-1}]
        self.hmin = 10 # minimum glacier thickness [m]

constant = constants()

class params:
    '''
    Scalar parameters.
    '''
    
    def __init__(self):
        self.n = 100 # number of grid points
        self.dt = 0.1 # time step size [a]
        self.L = 40e3 # initial domain length [m]
        self.Lsed = 80e3 # length of sediment model [m]
        self.sedDepth = 50 # depth to sediment, from sea level, prior to advance scenario [m]
        self.a_max = 4 # maximum mass balance rate [m a^{-1}]
        self.a_gradient = 0.01 # mass balance gradient
        self.ELA = 1000 # equilbrium line altitude [m]

param = params()

class paramsSed:
    '''
    Parameters for the sediment transport model.
    '''
    
    def __init__(self):
        # Brinkerhoff model
        self.h_eff = 0.1 # effective water thickness [m]
        self.c = 2e-12 # Brinkerhoff used 2e-12
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
    w_max = 10e3 - w_fjord # maximum valley width [m]
    
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
    
    h_divide, h_terminus = 300, 300 # thickness at the divide and the terminus [m]
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





#%%
def schoof_approx_friction(**kwargs):
    u = kwargs['velocity']
    h = kwargs['thickness']
    s = kwargs['surface']
    C = kwargs['friction']
    U0 = kwargs['U0']
    
    U = sqrt(inner(u,u))
    tau_0 = firedrake.interpolate(
        C*( U0**(1/weertman+1) + U**(1/weertman+1) )**(1/(weertman+1)), u.function_space()
    )
    
    # Update pressures
    p_water = rho_water * g * firedrake.max_value(0, h - s) # assuming phreatic surface at sea level
    p_ice = rho_ice * g * h
    phi = 1 - p_water/p_ice

    U = sqrt(inner(u, u))
    return tau_0 * phi * (
        (U0**(1/weertman+1) + U**(1/weertman+1))**(weertman/(weertman+1))-U0
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


#%% glacier model
class glacier:
    
    def __init__(self):
        L = param.L
        self.mesh1d = firedrake.IntervalMesh(param.n, L)
        self.mesh = firedrake.ExtrudedMesh(self.mesh1d, layers=1, name="mesh")

        # set up function spaces for the scalars (Q) and vectors (V) for the 2D mesh
        self.Q = firedrake.FunctionSpace(self.mesh, "CG", 2, vfamily="R", vdegree=0)
        self.V = firedrake.FunctionSpace(self.mesh, "CG", 2, vfamily="GL", vdegree=2, name='velocity')

        # create scalar functions for the spatial coordinates
        x_sc, z_sc = firedrake.SpatialCoordinate(self.mesh)
        self.x = firedrake.interpolate(x_sc, self.Q)
        self.z = firedrake.interpolate(z_sc, self.Q)

        # create initial geometry
        self.b, self.tideLine = bedrock(self.x, Q=self.Q) 
        self.h = initial_thickness(self.x, self.Q)                  
        self.s = icepack.compute_surface(thickness = self.h, bed = self.b) 
        self.u = initial_velocity(self.x, self.V)
        self.w = width(self.x, Q=self.Q)
    
    def regrid(self, n, L, L_new, sed):
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
        
        x_old = self.x # need to keep previous position function
        
        # Initialize mesh and mesh spaces
        self.mesh1d = firedrake.IntervalMesh(n, L_new)
        self.mesh = firedrake.ExtrudedMesh(self.mesh1d, layers=1, name="mesh")
        self.Q = firedrake.FunctionSpace(self.mesh, "CG", 2, vfamily="R", vdegree=0)
        self.V = firedrake.FunctionSpace(self.mesh, "CG", 2, vfamily="GL", vdegree=2, name='velocity')
        

        x_sc, z_sc = firedrake.SpatialCoordinate(self.mesh)
        self.x = firedrake.interpolate(x_sc, self.Q)
        self.z = firedrake.interpolate(z_sc, self.Q)
        
        xSed = sed.x.dat.data
        index = np.argsort(xSed)
        xSed = xSed[index]
        bedSed = sed.H.dat.data[index] + sed.bedrock.dat.data[index]
        
        # zBed_interpolator = interp1d(sed.x, sed.H+sed.zBedrock, kind='linear') # create interpolator to put bed elevation into a firedrake function
        zBed_interpolator = interp1d(xSed, bedSed, kind='quadratic') # create interpolator to put bed elevation into a firedrake function
        zBed_xarray = xarray.DataArray(zBed_interpolator(self.x.dat.data), [self.x.dat.data], 'x')
        self.b = icepack.interpolate(zBed_xarray, self.Q)

        # remesh thickness and velocity
        self.h =  self.__adjust_function(self.h, x_old, L, L_new, self.Q)
        self.u =  self.__adjust_function(self.u, x_old, L, L_new, self.V)

        # compute new surface based on thickness and bed; this determines whether
        # or not the ice is floating    
        self.s = icepack.compute_surface(thickness = self.h, bed = self.b) 
        
        self.w = width(self.x, Q=self.Q)
        
        



    def __adjust_function(self, func_src, x_old, L, L_new, funcspace_dest):
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
        
        x = x_old.dat.data
        index = np.argsort(x)
        x = x[index]
            
        # extract data from func_src
        data = func_src.dat.data
        
        # need to do this differently if the function space is a scalar or vector
        if funcspace_dest.topological.name=='velocity':
            data_len = int(len(data)/3)
        else:
            data_len = len(data)
        
        x = np.linspace(0, L, data_len, endpoint=True) # !!! redundant?
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
        
        return(func_dest)    
    
    
    def massBalance(self):
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
        
        a_max = param.a_max
        a_gradient = param.a_gradient
        ELA = param.ELA
        
        k = 0.005 # smoothing factor
            
        z_threshold = a_max/a_gradient + ELA # location where unsmoothed balance rate switches from nonzero to zero
    
        # unmodified mass balance rate    
        self.a = firedrake.interpolate(a_gradient*((self.s-ELA) - 1/(2*k) * ln(1+exp(2*k*(self.s - z_threshold)))), self.Q)
        
        # width gradient
        dwdx = icepack.interpolate(self.w.dx(0), self.Q)
        
        # modified mass balance rate
        self.a_mod = icepack.interpolate(self.a-self.u*self.h/self.w*dwdx, self.Q)
        
             

#%% sediment transport model
class sediment:
    
    def __init__(self, sedDepth):

        L = param.Lsed
        self.mesh1d = firedrake.IntervalMesh(param.n, L)
        self.mesh = firedrake.ExtrudedMesh(self.mesh1d, layers=1, name="mesh")

        # set up function spaces for the scalars (Q) and vectors (V) for the 2D mesh
        self.Q = firedrake.FunctionSpace(self.mesh, "CG", 2, vfamily="R", vdegree=0)

        # create scalar functions for the spatial coordinates
        x_sc, z_sc = firedrake.SpatialCoordinate(self.mesh)
        self.x = firedrake.interpolate(x_sc, self.Q)
        self.z = firedrake.interpolate(z_sc, self.Q)

        self.bedrock, _ = bedrock(self.x, Q=self.Q)
        
        # initial sediment thickness
        self.H = icepack.interpolate(-sedDepth-self.bedrock, self.Q)
        self.H = icepack.interpolate(conditional(self.H<0, 0, self.H), self.Q)
        
        # create inital functions for erosion, deposition, hillslope processes, and dH/dt
        self.erosionRate = icepack.interpolate(0*self.H, self.Q)
        self.depositionRate = icepack.interpolate(0*self.H, self.Q)
        self.hillslopeDiffusion = icepack.interpolate(0*self.H, self.Q)
        self.dHdt = icepack.interpolate(0*self.H, self.Q)
        
        # create initial functions for water flux and sediment flux
        self.Qw = icepack.interpolate(0*self.H, self.Q)
        self.Qs = icepack.interpolate(0*self.H, self.Q)
        
        
    def transport(self, glac, dt):
        '''
        

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        
        # first need to regrid so that sediment grid points align with glacier grid points
        
        index = np.argsort(glac.x.dat.data)
        x_ = glac.x.dat.data[index]
        
        deltaX = x_[2] # x[2] because using 2nd order finite elements
       
        nSed = np.floor(param.Lsed/deltaX) # number of sediment grid points
        LSed = nSed*deltaX # length of sediment domain
        
        index = np.argsort(self.x.dat.data)
        sedH_interpolator = interp1d(self.x.dat.data[index], self.H.dat.data[index], kind='quadratic', fill_value='extrapolate')

        
        # new mesh and function space
        self.mesh1d = firedrake.IntervalMesh(nSed, LSed)
        self.mesh = firedrake.ExtrudedMesh(self.mesh1d, layers=1, name="mesh")

        # set up function spaces for the scalars (Q) and vectors (V) for the 2D mesh
        self.Q = firedrake.FunctionSpace(self.mesh, "CG", 2, vfamily="R", vdegree=0)
        
        x_sc, z_sc = firedrake.SpatialCoordinate(self.mesh)
        self.x = firedrake.interpolate(x_sc, self.Q)
        self.z = firedrake.interpolate(z_sc, self.Q)

        # interpolate sediment thickness onto new mesh;
        # assumes that sediment extends far enough from glacier that bedrock 
        # and sediment are flat at end of domain
        index = np.argsort(self.x.dat.data)
        x_tmp = self.x.dat.data[index]
        H_tmp = sedH_interpolator(x_tmp)
        H_xarray = xarray.DataArray(H_tmp, [x_tmp], 'x')
        
        self.H = icepack.interpolate(H_xarray, self.Q)
        
        self.bedrock, _ = bedrock(self.x, Q=self.Q)        
    
        self.calcQw(glac.a, glac.w, glac.h) # runoff on the glacier domain
        
        
        
        # compute erosion rate
        bed = icepack.interpolate(self.bedrock+self.H, self.Q)
        alpha = bed.dx(0) # bed slope

        self.h_eff = icepack.interpolate(paramSed.h_eff * exp(10*alpha), self.Q) # !!! might need some more thought !!!
        
        self.delta_s = icepack.interpolate(1-exp(-self.H/10), self.Q)
        
        self.erosionRate = icepack.interpolate(paramSed.c * self.Qw**2/self.h_eff**3 * self.delta_s, self.Q)
        
        
        # compute sediment flux; don't want discharge to go to zero at the terminus, so first redefine Qw
        self.Qw = icepack.interpolate(conditional(self.x>np.max(x_), 5*np.max(self.Qw.dat.data), self.Qw), self.Q)
        
        self.calcQs()
        
        self.depositionRate = icepack.interpolate(paramSed.w * self.Qs / self.Qw, self.Q)
        
        # not yet including hillslope diffusion, and should be implicit
        self.dHdt = icepack.interpolate(self.depositionRate - self.erosionRate, self.Q)
        
        self.calcH(dt)
        
        
        # now need to fix the glacier bed to account for erosion and deposition
        bed = icepack.interpolate(self.H+self.bedrock, self.Q).dat.data
        xBed = self.x.dat.data
        
        index = np.argsort(xBed)
        xBed = xBed[index]
        bed = bed[index]
        
        bed_interpolator = interp1d(xBed, bed, kind='linear')
        
        glac.b.dat.data[:] = bed_interpolator(glac.x.dat.data)
        

    
    
    def calcH(self, dt):
        
        H_ = firedrake.Function(self.Q, name='thickness')
        H = firedrake.Function(self.Q, name='thicknessNext')
        
        v = firedrake.TestFunction(self.Q)
        
        H_.assign(self.H)
        H.assign(self.H)
        
        LHS = ((H-H_)/dt*v + paramSed.k*(H.dx(0)+self.bedrock.dx(0))*self.delta_s*v.dx(0) + (self.erosionRate - self.depositionRate)*v ) * dx
        
        firedrake.solve(LHS == 0, H)
        
        self.H = H
        
        
        
    def calcQs(self):
        '''
        Calculates the width-averaged sediment flux [m^2 a^{-1}].

        '''
        
        # nonlinear?
        # Qs = firedrake.Function(self.Q)
        # Qs.assign(self.Qw)
        
        # v = firedrake.TestFunction(self.Q)
        
        # LHS = (-Qs*v.dx(0) + Qs*paramSed.w/self.Qw*v - self.erosionRate*v ) * dx
        
        # bc = firedrake.DirichletBC(self.Q, 0, 1)
        # firedrake.solve(LHS == 0, Qs, bcs=[bc])
        
        # self.Qs = Qs
        
        
        # linear?
        u = firedrake.TrialFunction(self.Q) # unknown that we wish to determine
        
        v = firedrake.TestFunction(self.Q)

        # left and right hand sides of variational form
        LHS = u * ( -v.dx(0) + paramSed.w/self.Qw*v ) * dx
        # testing some artificial diffusion
        artificialDiffusionCoeff = firedrake.Constant(0)
        LHS = u * ( -v.dx(0) + paramSed.w/self.Qw*v) * dx + artificialDiffusionCoeff*u.dx(0)*v.dx(0) * dx
        RHS = (self.erosionRate) * v * dx
        
        Qs = firedrake.Function(self.Q) # create empty function to hold the solution
        
        bc = firedrake.DirichletBC(self.Q, (0), 2)
        
        firedrake.solve(LHS == RHS, Qs, bcs=[bc])
        
        self.Qs = Qs
        
    
    
        
    def calcQw(self, a, w, h):
        '''
        Calculates width-averaged runoff assuming 
        Q_w = 1/w * \int (a_max - a) * w * dx.
    
        Parameters
        ----------
        a : mass balance rate [m a^{-1}]
        w : glacier width [m]
        h : glacier thickness [m]; used to determine glacier mask, which is really 
            only relevant for land-terminating glaciers since then the grid is 
            fixed and there is a minimum ice thickness in front of the glacier
    
        Returns
        -------
        runoff : width-averaged runoff [m^2 a^{-1}]
    
        '''    
        
        Q = a.function_space()
        u = firedrake.TrialFunction(Q) # unknown that we wish to determine
    
        v = firedrake.TestFunction(Q)
    
        iceMask = icepack.interpolate(conditional(h<=constant.hmin+1e-4, 0, 1), Q)
    
        # left and right hand sides of variational form
        LHS = u.dx(0) * v * dx
        RHS = ( (param.a_max-a)  * iceMask ) * w * v * dx # the thing that we are integrating
    
        Qw = firedrake.Function(Q) # create empty function to hold the solution
    
        bc = firedrake.DirichletBC(Q, (1e-1), 1) # runoff equals 0 at the divide
    
        firedrake.solve(LHS == RHS, Qw, bcs=[bc]) # runoff in m^3 a^{-1}
        
        Qw = icepack.interpolate(Qw/w, Q) # width-averaged runoff on the glacier grid
        
        # determine runoff on sediment domain
        # for erosion purposes, no runoff beyond the end of the glacier
        # however, for deposition we need to account for transport by fjord 
        # circulation, which is included later
        self.Qw = icepack.interpolate(0*self.bedrock, self.Q)
        self.Qw.dat.data[:len(Qw.dat.data)] = Qw.dat.data
        
        return(Qw)        
        
        
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
        
        
        
        
class sedModel: # old finite difference approach

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
        bedInterpolator = interp1d(self.zBedrock, self.x, kind='cubic')
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
        xSed = np.arange(0, param.Lsed, dx)
        H_guess = np.ones(len(xSed))*50
        
        result = root(self.__sedTransportImplicit, H_guess, (x, h, a, b, u, Q, dt), method='hybr', options={'maxfev':int(1e6)})#, 'xtol':1e-12})
        
        # result = root(self.__sedTransportImplicit, self.H, (x, h, a, b, u, Q, dt), method='hybr', options={'maxfev':int(1e6)})#, 'xtol':1e-12})
        self.H = result.x
        # self.H[self.H<0] = 0 # temporary hack to keep thickness greater than 0!
        
        index = np.argsort(x.dat.data)
        xGlacier = x.dat.data[index]
        
        zBed = self.H+self.zBedrock # new bed elevation at sediment model grid points
        zBed_interpolator = interp1d(self.x, zBed, kind='cubic') # create interpolator to put bed elevation into a firedrake function
    
        zBed_xarray = xarray.DataArray(zBed_interpolator(xGlacier), [xGlacier], 'x')
        # zBed_xarray = xarray.DataArray(zBed, [xGlacier], 'x')
        
        b = icepack.interpolate(zBed_xarray, Q)
        
        return b
    
    def __sedTransportImplicit(self, H_guess, x, h, a, b, u, Q, dt):
        # use Backward Euler
        
        
        
        # create numpy array of glacier x-coordinates; make sure they are properly 
        # sorted
        index = np.argsort(x.dat.data)
        xGlacier = x.dat.data[index]
        bedGlacier = b.dat.data[index]
        uGlacier = icepack.depth_average(u).dat.data[index]
        hGlacier = h.dat.data[index]
        
        # mask out areas where the ice is less than 10 m thick, then extract melt rate
        # at glacier x-coordinates
        iceMask = icepack.interpolate(conditional(h<=constant.hmin+1e-4, 0, 1), Q)
        # meltRate = icepack.interpolate(conditional(a>0, 0, -a), Q)
        runoff = icepack.interpolate(param.b_max-a, Q)
        runoff = icepack.interpolate(runoff*iceMask, Q)
        runoff = runoff.dat.data[index]
    
        dx = xGlacier[1]
    
        
        
        # fill value needs some thought!
        sedH_interpolator = interp1d(self.x, self.H, kind='cubic', fill_value='extrapolate')
        sedBedrock_interpolator = interp1d(self.x, self.zBedrock, kind='cubic', fill_value='extrapolate')   
        
        self.x = np.arange(0, param.Lsed, dx)
        self.H = sedH_interpolator(self.x) # sediment thickness on extended glacier grid
        H_old = self.H
        
        self.zBedrock, _ = bedrock(self.x) # bedrock on extended glacier grid 
        w = width(self.x)
    
        # determine subglacial charge per unit width on the glacier model grid points
        # first need to extend the runoff to the end of the domain
        runoff = np.concatenate((runoff, np.zeros(len(self.x)-len(runoff))))
        
        self.Qw = cumtrapz(runoff*w, dx=dx, initial=0)/w # subglacial discharge, calculated from balance rate
    
    
    
        # delta_s = [np.min((x, 1)) for x in self.H] # erosion goes to 0 if sediment thickness is 0
        delta_s = (1-np.exp(-H_guess/10)) # similar to what Brinkerhoff has in code?
        
        alpha = np.gradient(self.zBedrock + H_guess, self.x) # bed slope at next time step
        
        # effective thickness is water depth if not glacier at the grid point
        # allows for some small erosion in front of the glacier, especially if the water gets shallow
        h_eff = paramSed.h_eff*np.ones(len(self.x)) * np.exp(10*alpha)
        
        index = self.x>np.max(xGlacier)
        h_eff[index] = -(H_guess[index]+self.zBedrock[index]) 
        
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
    
        zBed = H_guess+self.zBedrock # elevation of glacier bed from previous time step
    
        # calculate bed curvature for hillslope diffusion; d^2z/dx^2(zBed)
        zBed_curvature = (zBed[2:]-2*zBed[1:-1]+zBed[:-2])/dx**2
        zBed_left = (zBed[2]-2*zBed[1]+zBed[0])/dx**2
        zBed_right = (zBed[-1]-2*zBed[-2]+zBed[-3])/dx**2
    
        zBed_curvature = np.concatenate(([zBed_left], zBed_curvature, [zBed_right]))
    
        # self.hillslope = paramSed.k*(1-np.exp(-self.H/10))*zBed_curvature*np.sign(self.H)
        # redefine delta_a
        # delta_s = (1-np.exp(-H_guess/10))
        self.hillslope = paramSed.k*zBed_curvature*delta_s # sediment comes out of nowhere if not careful (under glacier, I mean)
        
        self.dHdt = self.depositionRate - self.erosionRate + self.hillslope
        RHS_new = self.dHdt
    
        # theta = 1 # theta = 1 => backwards euler; theta = 0 => forward euler; theta = 0.5 => Crank-Nicolson
        # backwards euler seems to work best
        # res = H_guess - H_old - dt*(theta*RHS_new + (1-theta)*RHS_old)
        res = H_guess - H_old - dt*RHS_new
        
        return(res)



    
        


#%% plotting functions

def basicPlot(glac, sed, basename, time):
    
    plt.ioff()
    
    index = np.argsort(glac.x.dat.data)
    x = glac.x.dat.data[index]*1e-3
    h = glac.h.dat.data[index]
    s = glac.s.dat.data[index]
    u = icepack.depth_average(glac.u).dat.data[index]
    b = glac.b.dat.data[index]
    w = glac.w.dat.data[index]*1e-3
    
    L = x[-1]
    
    fig, axes = plt.subplots(4, 2)
    fig.set_figwidth(10)
    fig.set_figheight(8)
    
    xlim = np.array([0,60])
    
    axes[0,0].set_xlabel('Longitudinal Coordinate [km]')
    axes[0,0].set_ylabel('Elevation [m]')
    axes[0,0].set_xlim(np.array([20,60]))
    axes[0,0].set_ylim(np.array([-500,1000]))
    axes[0,0].annotate("{:.0f}".format(np.floor(time)) + ' yrs', (58, 800), ha='right', va='top')
    
    
    axes[1,0].set_xlabel('Longitudinal Coordinate [km]')
    axes[1,0].set_ylabel('Transverse Coordinate [km]')
    axes[1,0].set_xlim(xlim)
    axes[1,0].set_ylim(np.array([-6, 6]))
    
    axes[2,0].set_xlabel('Longitudinal Coordinate [km]')
    axes[2,0].set_ylabel('Speed [m/yr]')
    axes[2,0].set_xlim(xlim)
    axes[2,0].set_ylim(np.array([0,2000]))
    
    axes[0,1].set_xlabel('Longitudinal Coordinate [km]')
    axes[0,1].set_ylabel('Flux per width [km$^2$ a$^{-1}$]')
    axes[0,1].set_xlim(xlim)
    axes[0,1].set_ylim(np.array([-0.1,0.6]))
    
    axes[1,1].set_xlabel('Longitudinal coordinate [km]')
    axes[1,1].set_ylabel('Rate [m a$^{-1}$]')
    axes[1,1].set_xlim(xlim)
    axes[1,1].set_ylim([-10,70])
    
    axes[2,1].set_xlabel('Longitudinal coordinate [km]')
    axes[2,1].set_ylabel('Rate [m a$^{-1}$]')
    axes[2,1].set_xlim(xlim)
    axes[2,1].set_ylim([-5, 5])
    
    axes[3,1].set_xlabel('Longitudinal coordinate [km]')
    axes[3,1].set_ylabel('Sediment thickness [m]')
    axes[3,1].set_xlim(xlim)
    axes[3,1].set_ylim([0,300])
    
    plt.tight_layout()
    

    
    sealevel = icepack.interpolate(0*sed.bedrock, sed.Q)
    sealevel = icepack.interpolate(conditional(sed.bedrock>0, sed.bedrock, 0), sed.Q)
    
    
    axes[0,0].fill_between(sed.x.dat.data*1e-3, sed.bedrock.dat.data, sed.bedrock.dat.data+sed.H.dat.data, color='saddlebrown')
    axes[0,0].fill_between(sed.x.dat.data*1e-3, sealevel.dat.data, sed.bedrock.dat.data+sed.H.dat.data, color='cornflowerblue')    
    axes[0,0].plot(np.concatenate((x,x[::-1])), np.concatenate((s,b[::-1])), 'k')
    axes[0,0].fill_between(x, s, b, color='w', linewidth=1)
    axes[0,0].plot(sed.x.dat.data*1e-3, sed.bedrock.dat.data, 'k')
    
    sed.w = width(sed.x, Q=sed.Q)
    
    axes[1,0].plot(np.array([L,L]), np.array([w[-1]/2,-w[-1]/2]), 'k')
    axes[1,0].plot(sed.x.dat.data*1e-3, sed.w.dat.data/2*1e-3, 'k')
    axes[1,0].plot(sed.x.dat.data*1e-3, -sed.w.dat.data/2*1e-3, 'k')
    axes[1,0].fill_between(sed.x.dat.data*1e-3, sed.w.dat.data/2*1e-3, -sed.w.dat.data/2*1e-3, color='cornflowerblue')
    axes[1,0].fill_between(x, w/2, -w/2, color='white')
    
    
    axes[2,0].plot(x, u, 'k')
    
    axes[3,0].axis('off')
    
    axes[0,1].plot(sed.x.dat.data*1e-3, sed.Qw.dat.data*1e-6, 'k', label='runoff')
    axes[0,1].plot(sed.x.dat.data*1e-3, sed.Qs.dat.data*1e-5, 'k:', label=r'sediment flux ($\times 10$)')
    axes[0,1].legend(loc='upper left')
    
    
    axes[1,1].plot(sed.x.dat.data*1e-3, sed.erosionRate.dat.data, 'k', label='erosion rate')
    axes[1,1].plot(sed.x.dat.data*1e-3, sed.depositionRate.dat.data, 'k--', label='deposition rate')
    axes[1,1].legend(loc='upper left')
    
    
    # axes[2,1].plot(sed.x*1e-3, sed.hillslope, 'k', label='hillslope processes')
    axes[2,1].plot(sed.x.dat.data*1e-3, sed.dHdt.dat.data, 'k', label='$\partial H_s/\partial t$')
    axes[2,1].legend(loc='upper left')
    
    # # axes[2,1].plot(sed.x*1e-3, sed.dHdt)
    axes[3,1].plot(sed.x.dat.data*1e-3, sed.H.dat.data, 'k')

    

    plt.savefig(basename + '.png', format='png', dpi=150)
    plt.close()
    plt.ion()
    
    
