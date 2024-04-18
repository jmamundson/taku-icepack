# model behaves better with height above flotation criterion (why?) or just use floatation criteria for initial spin up?
# if time step is too large get negative sediment thickness
# main issues with sediment model:
#   1. kink in bed slope at bedrock-sediment transition
#   2. too much deposition at the terminus
#   3. "oscillating" behavior if grid size too small    
# make sure no sediment is deposited up stream
# cite Creyts et al (2013): Evolution of subglacial overdeepenings in response to sediment redistribution and glaciohydraulic supercooling
# also Delaney paper on sugset_2d; discuss supply limited vs transport limited regimes
# linear interpolation for height above flotation
# issue with eroding where there isn't sediment (as result of backward Euler?)
# cheap hack to keep moraine submerged!!!
# behaves strangely if terminus goes afloat?

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
import copy
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
        self.L = 30e3 # initial domain length [m]
        self.Lsed = 80e3 # length of sediment model [m]
        self.sedDepth = 50 # depth to sediment, from sea level, prior to advance scenario [m]
        self.a_max = 5 # maximum mass balance rate [m a^{-1}]
        self.a_gradient = 0.012 # mass balance gradient
        self.ELA = 800 # equilbrium line altitude [m]

param = params()

class paramsSed:
    '''
    Parameters for the sediment transport model.
    '''
    
    def __init__(self):
        # Brinkerhoff model
        self.h_eff = 0.1 # effective water thickness [m]; Brinkerhoff use 0.1 m
        self.c = 5e-12 # Brinkerhoff used 2e-12
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
    
    
    k = 4 # smoothing parameter
    x_step = 10e3 # location of step [m]
    w_fjord = 4e3 # fjord width [m]
    w_max = 14e3 - w_fjord # maximum valley width [m]
    
    if "Q" in kwargs: # create firedrake function
        Q = kwargs["Q"]
        # using a logistic function, which makes the fjord walls roughly parallel
        w_array = w_max * (1 - 1 / (1+exp(-k*(x-x_step)/x_step))) + w_fjord
        
        
        w = firedrake.interpolate(w_array, Q)
    
    else:
        w = w_max * (1 - 1 / (1+np.exp(-k*(x-x_step)/x_step))) + w_fjord
        
        
    return(w)










#%%
def schoof_approx_friction(**kwargs):
    u = kwargs['velocity']
    h = kwargs['thickness']
    s = kwargs['surface']
    C = kwargs['friction']
    U0 = kwargs['U0']
    # x = kwargs['x']
    # b = kwargs['b']
    
    U = sqrt(inner(u,u))
    tau_0 = firedrake.interpolate(
        C*( U0**(1/weertman+1) + U**(1/weertman+1) )**(1/(weertman+1)), u.function_space()
    )
    
    
    # L = np.max(x.dat.data)
    # b0 = np.max(b.dat.data)
    # phreaticSurface = b/L*(L-x)
    
    # test = icepack.interpolate(rho_water * g * glac.x/L*(glac.h-glac.s), glac.Q)
    # test2 = icepack.interpolate(rho_water * g * firedrake.max_value(0, glac.h - glac.s), glac.Q)
    # fig, axes = plt.subplots(1,1)
    # firedrake.plot(icepack.depth_average(test), axes=axes)
    # firedrake.plot(icepack.depth_average(test2), axes=axes)
    
    #p_water = rho_ice * g * x/L*(h-s) # may be a problem if floating!
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
    
    
    Parameters
    ----------
    h : firedrake thickness function; only used to determine the function space
    
    Returns
    -------
    Cs : lateral drag coefficient
    
    '''
    
    Cs = firedrake.interpolate( 2/w * (4/(constant.A*w))**(1/3), w.function_space()) # consistent with Gagliardini et al. (2010)

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
        self.bed, self.tideLine = bedrock(self.x, Q=self.Q) # bedrock + sediment
        self.h = self.__initial_thickness()                  
        self.s = icepack.compute_surface(thickness = self.h, bed = self.bed) 
        self.u = self.__initial_velocity()
        self.w = width(self.x, Q=self.Q)
        self.b = icepack.interpolate(self.s-self.h, self.Q) # bottom of glacier; may or may not be floating
    
    def __initial_velocity(self):
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
        u = u_divide + (u_terminus - u_divide) * self.x/self.x.dat.data[-1] # velocity profile
        u0 = firedrake.interpolate(u, self.V) # Vector function space

        return(u0)
    
    def __initial_thickness(self):
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
        h = h_divide - (h_divide-h_terminus) * self.x/self.x.dat.data[-1] # thickness profile [m]
        h0 = firedrake.interpolate(h, self.Q)

        return(h0)
    

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
        
        # determine "bed" elevation, i.e., the elevation of the valley bottom
        # xSed = sed.x.dat.data # x-coordinate in sediment model
        # index = np.argsort(xSed)
        # xSed = xSed[index]
        # bedSed = sed.H.dat.data[index] + sed.bedrock.dat.data[index] # bedrock + sediment in sediment model
        
        # use with finite differences
        xSed = sed.x
        bedSed = sed.H + sed.bedrock
        
        zBed_interpolator = interp1d(xSed, bedSed, kind='linear') # create interpolator to put bed elevation into a firedrake function
        
        
        index = np.argsort(self.x.dat.data)
        zBed_xarray = xarray.DataArray(zBed_interpolator(self.x.dat.data[index][::2]), [self.x.dat.data[index][::2]], 'x')
        self.bed = icepack.interpolate(zBed_xarray, self.Q)

        # remesh thickness and velocity
        self.h =  self.__adjust_function(self.h, x_old, L, L_new, self.Q)
        self.u =  self.__adjust_function(self.u, x_old, L, L_new, self.V)

        # compute new surface based on thickness and bed; this determines whether
        # or not the ice is floating    
        self.s = icepack.compute_surface(thickness = self.h, bed = self.bed) 
        self.b = icepack.interpolate(self.s - self.h, self.Q)
        
        self.w = width(self.x, Q=self.Q)
        
        self.massBalance()


    def __adjust_function(self, func_src, x_old, L, L_new, funcspace_dest):
        '''
        Adjusts (shrinks or grows) function spaces.

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
    
    
    def massBalance(self, **kwargs):
        '''
        Mass balance profile. 
    
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
        
        if "ELA" in kwargs:
            param.ELA = kwargs["ELA"]
        
        a_max = param.a_max
        a_gradient = param.a_gradient
        ELA = param.ELA
        
        k = 0.005 # smoothing factor
            
        z_threshold = a_max/a_gradient + ELA # location where unsmoothed balance rate switches from nonzero to zero
    
        # unmodified mass balance rate    
        self.a = firedrake.interpolate(a_gradient*((self.s-ELA) - 1/(2*k) * ln(1+exp(2*k*(self.s - z_threshold)))), self.Q)
        
        
    def balanceFlux(self):
        
        u = firedrake.TrialFunction(self.Q) # unknown that we wish to determine
    
        v = firedrake.TestFunction(self.Q)
    
        iceMask = icepack.interpolate(conditional(self.h<=constant.hmin+1e-4, 0, 1), self.Q)
    
        # left and right hand sides of variational form
        LHS = u.dx(0) * v * dx
        RHS = (self.a * self.w * iceMask ) * v * dx # the thing that we are integrating
    
        bc = firedrake.DirichletBC(self.Q, (1e-1), 1)
        
        Qb = firedrake.Function(self.Q)
        
        firedrake.solve(LHS == RHS, Qb, bcs=[bc])
        
        self.Qb = Qb # balance flux [m^3 a^{-1}]
        self.ub = icepack.interpolate(self.Qb/(self.h*self.w), self.Q) # balance velocity [m a^{-1}]
        # self.ub = icepack.interpolate(self.Qb/self.h, self.Q) # balance velocity [m a^{-1}]
        
    def HAF(self):
        '''
        Finds new glacier length using the height above flotation calving criteria.
        For some reason this only works if desired height above flotation is more 
        than about 5 m. Don't use all grid points?
    
        Parameters
        ----------
        L : domain length [m]
        h, s : firedrake functions for the thickness and surface elevation
    
        Returns
        -------
        L_new : new domain length [m]
    
        '''
        
        
        h_flotation = firedrake.interpolate(-rho_water/rho_ice*self.b, self.Q) # thickness flotation (thickness if at flotation) [m]
        haf = firedrake.interpolate(self.h - h_flotation, self.Q) # height above flotation [m]
    
        
        index = np.argsort(self.x.dat.data)
        x = self.x.dat.data[index]
        haf_dat = haf.dat.data[index] # height above flotation as a numpy array
        
        haf_interpolator = interp1d(haf_dat[::2], x[::2], kind='linear', fill_value='extrapolate')
        
        haf_desired = 10 # desired height above flotation at the terminus
        L_new = haf_interpolator(haf_desired) # new domain length
        
        L = x[-1]
        dLdt = (L_new-L)/param.dt
        print('dL/dt: ' + "{:.02f}".format(dLdt) + ' m a^{-1}')
    
        return(L_new)


    def HAFmodified(self):
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
        
        q = 0.15
        f = icepack.interpolate(self.h + rho_water/rho_ice*(1+q)*self.b, self.Q) # function to find zero of
        
        index = np.argsort(self.x.dat.data)
        x = self.x.dat.data[index]
        f = f.dat.data[index]
        
        L = x[-1]
        
        f_interpolator = interp1d(f, x, kind='linear', fill_value='extrapolate')
        
        L_new = f_interpolator(0)
        
       
        dLdt = (L_new-L)/param.dt
        print('dL/dt: ' + "{:.02f}".format(dLdt) + ' m a^{-1}')
    
        return(L_new)


    def massFlux(self):
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
        
        alpha = 1.13 # frontal ablation parameter
        
        # Qb = firedrake.assemble(glac.a * glac.w * dx) # balance flux [m^3 a^{-1}]
        
        index = np.argsort(self.x.dat.data)
        x = self.x.dat.data[index]
        # Wt = glac.w.dat.data[index[-1]] # terminus width [m]
        # Ht = glac.h.dat.data[index[-1]] # terminus thickness
        
        # Ub = Qb / (Wt*Ht) # balance velocity [m a^{-1}]
        
        self.balanceFlux()
        Ub = self.ub.dat.data[index[-1]]
        Ut = icepack.depth_average(self.u_bar).dat.data[index[-1]] # centerline terminus velocity [m a^{-1}]
        
        
        dLdt = (alpha-1)*(Ub-Ut) # rate of length change [m a^{-1}]
        
        L = x[-1]
        L_new = L + dLdt*param.dt
       
        
        dLdt = (L_new-L)/param.dt
        print('dL/dt: ' + "{:.02f}".format(dLdt) + ' m a^{-1}')
        
        return(L_new)

    def crevasseDepth(self):
        
        u_ = icepack.interpolate(4/5*self.u, self.Q) # width- and depth-averaged velocity
        exx = icepack.interpolate(u_.dx(0), self.Q) # width- and depth-averaged strain rate
        # note: assume critical strain rate for crevasse opening is 0
        
        d = icepack.interpolate(2/(rho_ice*g)*(exx/constant.A)**(1/3), self.Q) # crevasse depth [m]

        f = icepack.interpolate(self.s-d, self.Q) # function to find zero of

        # create interpolator
        index = np.argsort(self.x.dat.data)
        x = self.x.dat.data[index]
        f = f.dat.data[index]
        
        f_interpolator = interp1d(f, x, kind='linear', fill_value='extrapolate')
        
        L_new = f_interpolator(0)
        L = x[-1]
        
        dLdt = (L_new-L)/param.dt
        print('dL/dt: ' + "{:.02f}".format(dLdt) + ' m a^{-1}')
        
        return(L_new)



    def vonMises(self):
        
        index = np.argsort(self.x.dat.data)
        Ut = icepack.depth_average(self.u).dat.data[index][-1]
        Ht = self.h.dat.data[index][-1] # terminus thickness
        L = self.x.dat.data[index][-1]
        
        couplingLength = Ht*4.5 # somewhat arbitrary
        
        u = icepack.interpolate(4/5*self.u, self.Q) # depth- and width-averaged velocity
        
        # eigenvalues of the horizontal strain rate; eps2 assumed to equal 0
        eps1 = icepack.interpolate(u.dx(0), self.Q)#.dat.data[index] # depth- and width-averaged strain rate
        
        eps1 = eps1.dat.data[index][-1]
        #x = self.x.dat.data
        
        
        #eps1 = icepack.interpolate(conditional(self.x<L-couplingLength, np.nan, eps1), self.Q)
        #eps1 = np.nanmean(eps1.dat.data)
        
        
        eps_e = np.sqrt(0.5*np.max(eps1,0)**2) # effective tensile strain rate
        
        sigma = np.sqrt(3)*constant.A.dat.data[0]**(-1/3)*eps_e**(1/3)
        
        sigma_threshold = 0.1 # MPa, I think!
        
        Uc = Ut*sigma/sigma_threshold 

        dLdt = Ut-Uc

        L_new = L + dLdt*param.dt
        
        print('dL/dt: ' + "{:.02f}".format(dLdt) + ' m a^{-1}')
        return(L_new)


    def eigencalving(self):
        '''
        not quite sure how to implement in 1D model

        Returns
        -------
        None.

        '''
        
        index = np.argsort(self.x.dat.data)
        x = self.x.dat.data[index][::2]
        u = icepack.interpolate(4/5*self.u, self.Q).dat.data[index][::2]
        
        L = x[-1]
        exx = (u[-1]-u[-2])/x[1]
        Ut = u[-1]
        K = 1.5e3
        
        dLdt = Ut - K*exx
        
        L_new = L + dLdt*param.dt
        
        print('dL/dt: ' + "{:.02f}".format(dLdt) + ' m a^{-1}')
        
        return(L_new)
    
    
    
#%% sediment transport model with finite differences
    
class sedimentFD:
    

    def __init__(self, sedDepth):
        '''
        Create grid and initial values of the sediment model.
    
        Parameters
        ----------
        L : domain length [m]
        sedDepth = depth to sediment from sea level [m]
    
        '''
        
        self.x = np.linspace(0, param.Lsed, param.n, endpoint=True)
        
        self.bedrock, _ = bedrock(self.x)
        
        # determine x-location of start of sediment
        bedInterpolator = interp1d(self.bedrock, self.x, kind='linear')
        sedLevelInit = bedInterpolator(-sedDepth) # sediment fills fjord to 50 m depth
    
        # initial sediment thickness
        self.H = np.zeros(len(self.bedrock)) # sediment thickness
        self.H[self.x>sedLevelInit] = -sedDepth-self.bedrock[self.x>sedLevelInit]
    
        
        self.Qw = np.zeros(len(self.x))
        self.Qs = np.zeros(len(self.x))
        self.dHdt = np.zeros(len(self.x))
        self.erosionRate = np.zeros(len(self.x))
        self.depositionRate = np.zeros(len(self.x))

    

    def transport(self, glac, dt):
        
        
        index = np.argsort(glac.x.dat.data)
        xGlacier = glac.x.dat.data[index]
        dx = xGlacier[1]
        
        # remesh sediment model
        # this works because the sediment thickness is constant far from the glacier
        sedH_interpolator = interp1d(self.x, self.H, kind='linear', fill_value='extrapolate')
         
        self.x = np.arange(0, param.Lsed, dx)
        self.H = sedH_interpolator(self.x) # sediment thickness on extended glacier grid
        self.H_old = self.H # store old value for backward Euler
        
        self.bedrock, _ = bedrock(self.x) # bedrock on extended glacier grid 
        #self.w = width(self.x)
        
        
        
        result = root(self.__sedTransportImplicit, self.H_old, (glac, dt), method='hybr', options={'maxfev':int(1e6)})#, 'xtol':1e-12})
        
        self.H = result.x
        # self.H[self.H<0] = 0 # temporary hack to keep thickness greater than 0!
        
        
        zBed = self.H+self.bedrock # new bed elevation at sediment model grid points
        
        zBed_xarray = xarray.DataArray(zBed[:len(xGlacier)], [xGlacier], 'x')
        # zBed_xarray = xarray.DataArray(zBed, [xGlacier], 'x')
        
        glac.b = icepack.interpolate(zBed_xarray, glac.Q)
        
        
    



    def __sedTransportImplicit(self, H_guess, glac, dt):
        # use Backward Euler
        

        # create numpy array of glacier x-coordinates; make sure they are properly 
        # sorted
        index = np.argsort(glac.x.dat.data)
        xGlacier = glac.x.dat.data[index]
        dx = xGlacier[1]
    
        # mask out areas where the ice is less than 10 m thick, then extract melt rate
        # at glacier x-coordinates
        # iceMask really only does something if glacier is land-terminating
        iceMask = icepack.interpolate(conditional(glac.h<=constant.hmin+1e-4, 0, 1), glac.Q)
        
        runoff = icepack.interpolate(param.a_max-glac.a, glac.Q)
        runoff = icepack.interpolate(runoff*iceMask, glac.Q)
        
        runoff = runoff.dat.data[index]
        w = glac.w.dat.data[index]
        Qw = cumtrapz(runoff*w, dx=dx, initial=0)/w # runoff per unit width on the glacier grid
    
    
    
        ##### Increase runoff in fjord to account for submarine melting ##### 
        beta = (Qw[-1]-Qw[-2])/(xGlacier[-1]-xGlacier[-2])
        k = 0.001
    
        x_inflection = 5000 # distance from terminus to put "inflection point"
    
        coeff1 = beta*(1+np.exp(x_inflection*k))/(np.exp(x_inflection*k))
        coeff2 = Qw[-1] + coeff1/k*np.log(1+np.exp(k*x_inflection))
    
        xFjord = self.x[self.x>xGlacier[-1]]
        Qw_fjord = -coeff1/k*np.log(1+ np.exp(-k*(xFjord-(xGlacier[-1]+x_inflection)))) + coeff2
    
        self.Qw = np.concatenate((Qw, Qw_fjord))
        
    
           
        self.delta_s = (1-np.exp(-H_guess/2)) # similar to what Brinkerhoff has in code?
        
        # effective thickness is water depth if not glacier at the grid point
        # allows for some small erosion in front of the glacier, especially if the water gets shallow
        self.h_eff = paramSed.h_eff*np.ones(len(self.x)) 
        
        


        zBed = H_guess+self.bedrock # elevation of glacier bed from previous time step
        alpha = np.gradient(zBed, self.x)    
        
        
        index = self.x>xGlacier[-1]
        self.h_eff[index] = -(H_guess[index]+self.bedrock[index]) 
        
        self.h_eff = self.h_eff*np.exp(5*alpha)
        
        self.erosionRate = paramSed.c * self.Qw**2/self.h_eff**3 * self.delta_s
        
    
        ##### solve for sediment transport with left-sided difference #####
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
        
        self.Qs = np.linalg.solve(D,f)
        
    
        # extra step needed here because problems when Qw==0
        self.depositionRate = np.zeros(len(self.Qs))
        self.depositionRate[self.Qw>0] = paramSed.w*self.Qs[self.Qw>0]/self.Qw[self.Qw>0]
    
        
    
        # calculate bed curvature for hillslope diffusion; dz^2/dx^2(zBed)
        zBed_curvature = (zBed[2:]-2*zBed[1:-1]+zBed[:-2])/dx**2
        zBed_left = (zBed[2]-2*zBed[1]+zBed[0])/dx**2
        zBed_right = (zBed[-1]-2*zBed[-2]+zBed[-3])/dx**2
    
        zBed_curvature = np.concatenate(([zBed_left], zBed_curvature, [zBed_right]))
    
        # self.hillslope = paramSed.k*(1-np.exp(-self.H/10))*zBed_curvature*np.sign(self.H)
        # redefine delta_s for hillslope?
        delta_s = (1-np.exp(-H_guess/10))
        self.hillslope = paramSed.k*zBed_curvature#*delta_s
        
        
        self.dHdt = self.depositionRate - self.erosionRate + self.hillslope
        RHS = self.dHdt
    
        # backward Euler (RHS depends on H_guess)
        res = H_guess - self.H_old - RHS*dt
        
        return(res)
    
    
    
    
    
    
    
    
    
    
    
#%% sediment transport model
class sediment:
    
    def __init__(self, sedDepth):

        L = param.Lsed
        
        self.mesh = firedrake.IntervalMesh(param.n, L)
        self.Q = firedrake.FunctionSpace(self.mesh, "DG", 1)
        x, = firedrake.SpatialCoordinate(self.mesh)
        
        self.x = firedrake.interpolate(x, self.Q)
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
        
        
        
        # self.mesh1d = firedrake.IntervalMesh(param.n, L)
        # self.mesh = firedrake.ExtrudedMesh(self.mesh1d, layers=1, name="mesh")

        # # set up function space for the scalars 
        # self.Q = firedrake.FunctionSpace(self.mesh1d, "DG", 1)

        # # create scalar functions for the spatial coordinates
        # x_sc, z_sc = firedrake.SpatialCoordinate(self.mesh)
        # self.x = firedrake.interpolate(x_sc, self.Q)
        # self.z = firedrake.interpolate(z_sc, self.Q)

        # self.bedrock, _ = bedrock(self.x, Q=self.Q)
        
        # # initial sediment thickness
        # self.H = icepack.interpolate(-sedDepth-self.bedrock, self.Q)
        # self.H = icepack.interpolate(conditional(self.H<0, 0, self.H), self.Q)
        
        # # create inital functions for erosion, deposition, hillslope processes, and dH/dt
        # self.erosionRate = icepack.interpolate(0*self.H, self.Q)
        # self.depositionRate = icepack.interpolate(0*self.H, self.Q)
        # self.hillslopeDiffusion = icepack.interpolate(0*self.H, self.Q)
        # self.dHdt = icepack.interpolate(0*self.H, self.Q)
        
        # # create initial functions for water flux and sediment flux
        # self.Qw = icepack.interpolate(0*self.H, self.Q)
        # self.Qs = icepack.interpolate(0*self.H, self.Q)
        
    
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
        
        deltaX = x_[2] # needed to make different types of elements align...
       
        nSed = np.floor(param.Lsed/deltaX) # number of sediment grid points
        LSed = nSed*deltaX # length of sediment domain
        
        # index = np.argsort(self.x.dat.data)
        # sedH_interpolator = interp1d(self.x.dat.data[index], self.H.dat.data[index], kind='linear', fill_value='extrapolate')
        # erosionRate_interpolator = interp1d(self.x.dat.data[index], self.erosionRate.dat.data[index], kind='linear', fill_value='extrapolate')
        # depositionRate_interpolator = interp1d(self.x.dat.data[index], self.depositionRate.dat.data[index], kind='linear', fill_value='extrapolate')
        
        
        sedH_interpolator = interp1d(self.x.dat.data, self.H.dat.data, kind='linear', fill_value='extrapolate')
        erosionRate_interpolator = interp1d(self.x.dat.data, self.erosionRate.dat.data, kind='linear', fill_value='extrapolate')
        depositionRate_interpolator = interp1d(self.x.dat.data, self.depositionRate.dat.data, kind='linear', fill_value='extrapolate')
        
        
        # new mesh and function space
        #self.mesh = firedrake.IntervalMesh(nSed, LSed)
        self.mesh = firedrake.IntervalMesh(nSed, LSed)
        # self.mesh = firedrake.ExtrudedMesh(self.mesh, layers=1, name="mesh")

        # set up function spaces for the scalars (Q) and vectors (V) for the 2D mesh
        self.Q = firedrake.FunctionSpace(self.mesh, "DG", 1)
        

        # x_sc, z_sc = firedrake.SpatialCoordinate(self.mesh)
        x, = firedrake.SpatialCoordinate(self.mesh)
        self.x = firedrake.interpolate(x, self.Q)
        
        # self.z = firedrake.interpolate(z_sc, self.Q)

        # interpolate sediment thickness, erosion rate, and deposition rate 
        # onto new mesh; need old values for Crank-Nicolson
        # assumes that sediment extends far enough from glacier that bedrock 
        # and sediment are flat at end of domain
        # index = np.argsort(self.x.dat.data)
        # x_tmp = self.x.dat.data[index]
        
        self.H = firedrake.Function(self.Q)
        self.H.dat.data[:] = sedH_interpolator(self.x.dat.data)

        self.erosionRate = firedrake.Function(self.Q)
        self.erosionRate.dat.data[:] = erosionRate_interpolator(self.x.dat.data)
        
        self.depositionRate = firedrake.Function(self.Q)
        self.depositionRate.dat.data[:] = depositionRate_interpolator(self.x.dat.data)
        
        
        
        # H_tmp = sedH_interpolator(x_tmp)
        # H_xarray = xarray.DataArray(H_tmp, [x_tmp], 'x')
        # self.H = icepack.interpolate(H_xarray, self.Q)
        
        # erosionRate_tmp = erosionRate_interpolator(x_tmp)
        # erosionRate_xarray = xarray.DataArray(erosionRate_tmp, [x_tmp], 'x')
        # self._erosionRate = icepack.interpolate(erosionRate_xarray, self.Q)
        
        # depositionRate_tmp = depositionRate_interpolator(x_tmp)
        # depositionRate_xarray = xarray.DataArray(depositionRate_tmp, [x_tmp], 'x')
        # self._depositionRate = icepack.interpolate(depositionRate_xarray, self.Q)
        
        self.bedrock, _ = bedrock(self.x, Q=self.Q)        
    
    
        self.calcQw(glac) 
        
        
        # compute erosion rate
        bed = icepack.interpolate(self.bedrock+self.H, self.Q)
        alpha = bed.dx(0) # bed slope

        self.h_eff = icepack.interpolate(paramSed.h_eff * exp(10*alpha)/exp(10*alpha), self.Q) # !!! might need some more thought !!!
        
        self.h_eff = icepack.interpolate(conditional(self.x>x_[-1], -(self.bedrock+self.H), self.h_eff), self.Q)
        
        # self.h_eff = icepack.interpolate(conditional(self.x>x_[-1], -(self.bedrock+self.H), paramSed.h_eff), self.Q)
        
        self.delta_s = icepack.interpolate(1-exp(-self.H/2), self.Q)
        
        self.erosionRate = icepack.interpolate(paramSed.c * self.Qw**2/self.h_eff**3 * self.delta_s, self.Q)
        
        
        # Discharge should not go to zero at the terminus because sediment is
        # carried out into the fjord. To account for additional input from 
        # submarine melting (of the glacier and icebergs), this section 
        # extends the runoff to some higher, constant value by using a 
        # logistic function.
        
        beta = (self.Qw.dat.data[2*param.n]-self.Qw.dat.data[2*param.n-1])/(self.x.dat.data[2*param.n+1]-self.x.dat.data[2*param.n])
        k = 0.001
        
        x_inflection = 5000 # distance from terminus to put "inflection point"
        
        coeff1 = beta*(1+np.exp(x_inflection*k))/(np.exp(x_inflection*k))
        coeff2 = np.max(self.Qw.dat.data) + coeff1/k*np.log(1+np.exp(k*x_inflection))
        
        Qw_fjord = icepack.interpolate(-coeff1/k*ln(1+ exp(-k*(self.x-(self.x.dat.data[2*param.n]+x_inflection)))) + coeff2, self.Q)
        
        # use previous calculation for Qw where the glacier exists, and Qw_fjord
        # in front of the glacier
        self.Qw = icepack.interpolate(conditional(self.x<=x_[-1], self.Qw, Qw_fjord), self.Q)
        
        
        self.calcQs()
        
        self.depositionRate = icepack.interpolate(paramSed.w * self.Qs / self.Qw, self.Q)
        
        
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
        
        # theta = 0.5 # 0.5 = Crank-Nicolson; 0 and 1 are explicit and implicit Euler
        
        # !!! check diffusion term; may need to multiply by delta_s
        LHS = ( (H-H_)/dt*v + 
               paramSed.k*(H.dx(0)+self.bedrock.dx(0))*v.dx(0) + 
               (self.erosionRate - self.depositionRate)*v ) * dx
        
        firedrake.solve(LHS == 0, H)
        
        H_new = icepack.interpolate(conditional(H<0.1, 0, H), self.Q)
        self.dHdt = icepack.interpolate((H_new - self.H)/dt, self.Q)
        
        self.H = H_new
        
        icepack.interpolate(conditional(self.H<-self.bedrock-5, self.H, -5), self.Q) # cheap hack to keep moraine submerged !!!
        
        
    def calcQs(self):
        '''
        Calculates the width-averaged sediment flux [m^2 a^{-1}].

        L = glacier length
        '''
        
        
        # First sort and extract arrays that are needed. These will be put back 
        # into functions for solving the ODE. Note that we only need every 
        # other data point since we are using quadratic functions (at least I
        # think that is why the arrays contain 2n+1 points).
        # index = np.argsort(self.x.dat.data)
        # x = self.x.dat.data[index][::2]
        # H = self.H.dat.data[index][::2]
        # Qw = self.Qw.dat.data[index][::2]
        # erosionRate = self.erosionRate.dat.data[index][::2]
        # deltaS = self.delta_s.dat.data[index][::2]
        
        # First extract arrays from sediment object that will be shortened.
            # x_ = self.x.dat.data
            # H_ = self.H.dat.data        
            # Qw_ = self.Qw.dat.data
            # erosionRate_ = self.erosionRate.dat.data
            # deltaS_ = self.delta_s.dat.data
            
            
            # # Determine the domain to be used.
            # index = H_.nonzero()[0] # grid points that have sediment
            # #index = np.concatenate(([index[0]-1], index)) # 
            # ind0 = index[0] # index of first grid point with sediment
            # x0 = x_[ind0] # terminus or first point of nonzero thickness
            
            
            # xL = np.max(self.x.dat.data) # last grid point
            
            # # Create shortened xarrays for putting back into functions.
            # # H_xarray = xarray.DataArray(H[index], [x[index]], 'x')
            # # Qw_xarray = xarray.DataArray(Qw[index], [x[index]], 'x')
            # # erosionRate_xarray = xarray.DataArray(erosionRate[index], [x[index]], 'x')
            # # deltaS_xarray = xarray.DataArray(deltaS[index], [x[index]], 'x')
            
            # # Create new mesh and set up new function space.       
            # # First determine grid spacing and number of grid points in the new 
            # # mesh.
            # #deltaX = x_[1] 
            # n = len(index)-1 #int((xL-x0)/deltaX)+1
            
            # mesh = firedrake.IntervalMesh(n, x0, right=xL)
            # Q = firedrake.FunctionSpace(mesh, "DG", 1)
            # x, = firedrake.SpatialCoordinate(mesh)
            
            # x = firedrake.interpolate(x, Q)
        
        
        # mesh1d = firedrake.IntervalMesh(n, x0, right=xL)
        # mesh = firedrake.ExtrudedMesh(mesh1d, layers=1, name="mesh")
        # Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
        
        # x_sc, z_sc = firedrake.SpatialCoordinate(mesh)
        # x = firedrake.interpolate(x_sc, Q)

        # Put arrays back into firedrake functions.
            # H = firedrake.Function(Q)
            # H.dat.data[:] = H_[index]
            
            # Qw = firedrake.Function(Q)
            # Qw.dat.data[:] = Qw_[index]
            
            # erosionRate = firedrake.Function(Q)
            # erosionRate.dat.data[:] = erosionRate_[index]
            
            # deltaS = firedrake.Function(Q)
            # deltaS.dat.data[:] = deltaS_[index]
        
        
        # H = icepack.interpolate(H_xarray, Q)
        # Qw = icepack.interpolate(Qw_xarray, Q)
        # erosionRate = icepack.interpolate(erosionRate_xarray, Q)
        # deltaS = icepack.interpolate(deltaS_xarray, Q)
        
        
        # # For some reason, the last grid point is often set to nan. Fix this.
        # H.dat.data[-1] = H_xarray[-1]
        # Qw.dat.data[-1] = Qw_xarray[-1]
        # erosionRate.dat.data[-1] = erosionRate_xarray[-1]
        # deltaS.dat.data[-1] = deltaS_xarray[-1]
        
        u = firedrake.TrialFunction(self.Q) # unknown that we wish to determine
        
        v = firedrake.TestFunction(self.Q)

        
        # left and right hand sides of variational form
        LHS = u * ( -v.dx(0) + paramSed.w/self.Qw*v ) * dx
        RHS = (self.erosionRate) * v * dx # ! includes 1 mm/yr bedrock erosion
        
        Qs = firedrake.Function(self.Q) # create empty function to hold the solution
        
        
        bc = firedrake.DirichletBC(self.Q, (0), 1) 
        
        
        firedrake.solve(LHS == RHS, Qs, bcs=[bc])
        self.Qs = Qs
        
        
        # Qs_interpolator = interp1d(x.dat.data, Qs.dat.data, fill_value = 0, bounds_error=False)
        
        # self.Qs = firedrake.Function(self.Q)
        # self.Qs.dat.data[:] = Qs_interpolator(self.x.dat.data)
        
        # Qs_xarray = xarray.DataArray(Qs_interpolator(self.x.dat.data), [self.x.dat.data], 'x')
        # self.Qs = icepack.interpolate(Qs_xarray, self.Q)
        
        
        
    
    
        
    def calcQw(self, glac):
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
        
        Q = glac.Q
        u = firedrake.TrialFunction(Q) # unknown that we wish to determine
    
        v = firedrake.TestFunction(Q)
    
        iceMask = icepack.interpolate(conditional(glac.h<=constant.hmin+1e-4, 0, 1), Q)
    
        # left and right hand sides of variational form
        LHS = u.dx(0) * v * dx
        RHS = ( (param.a_max-glac.a)  * iceMask ) * glac.w * v * dx # the thing that we are integrating
    
        Qw = firedrake.Function(Q) # create empty function to hold the solution
    
        bc = firedrake.DirichletBC(Q, (1e-1), 1) # runoff equals 0 at the divide
    
        firedrake.solve(LHS == RHS, Qw, bcs=[bc]) # runoff in m^3 a^{-1}
        
        Qw = icepack.interpolate(Qw/glac.w, Q) # width-averaged runoff on the glacier grid
        
        index = np.argsort(glac.x.dat.data)
    
        # Qw is on the glacier grid points, staggered half a point away from the sediment grid points
        # so need to interpolate 
        Qw_dat = (Qw.dat.data[index][:-1]+Qw.dat.data[index][1:])/2
        
        # determine runoff on sediment domain
        # for erosion purposes, no runoff beyond the end of the glacier
        # however, for deposition we need to account for transport by fjord 
        # circulation, which is included later
        
        self.Qw = firedrake.Function(self.Q) # populates with zeros
        self.Qw.dat.data[1:len(Qw_dat)+1] = Qw_dat
        # self.Qw = icepack.interpolate(0*self.bedrock, self.Q)
        # self.Qw.dat.data[:len(Qw.dat.data)] = Qw.dat.data
        
        return(Qw)        
        
        
        
     
        
     
        
     
        
     
        
     
        
     
        
     
        
        
        
        


#%% plotting functions

def basicPlot(glac, sed, basename, time):
    
    plt.ioff()
    
    index = np.argsort(glac.x.dat.data)
    x = glac.x.dat.data[index]*1e-3
    h = glac.h.dat.data[index]
    s = glac.s.dat.data[index]
    u = icepack.depth_average(glac.u_bar).dat.data[index]
    b = glac.b.dat.data[index]
    w = glac.w.dat.data[index]*1e-3
    ub = glac.ub.dat.data[index]
    
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
    axes[1,0].set_ylim(np.array([-10, 10]))
    
    axes[2,0].set_xlabel('Longitudinal Coordinate [km]')
    axes[2,0].set_ylabel('Speed [m/yr]')
    axes[2,0].set_xlim(xlim)
    axes[2,0].set_ylim(np.array([0,3000]))
    
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
    
    
    axes[2,0].plot(x, u, 'k', label='average velocity')
    axes[2,0].plot(x, ub, 'k--', label='balance velocity')
    axes[2,0].legend(loc='upper left')
    
    
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
    
    


def basicPlotFD(glac, sed, basename, time):
    
    plt.ioff()
    
    index = np.argsort(glac.x.dat.data)
    x = glac.x.dat.data[index]*1e-3
    h = glac.h.dat.data[index]
    s = glac.s.dat.data[index]
    u = icepack.depth_average(glac.u_bar).dat.data[index]
    b = glac.b.dat.data[index]
    w = glac.w.dat.data[index]*1e-3
    ub = glac.ub.dat.data[index]
    
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
    axes[1,0].set_ylim(np.array([-10, 10]))
    
    axes[2,0].set_xlabel('Longitudinal Coordinate [km]')
    axes[2,0].set_ylabel('Speed [m/yr]')
    axes[2,0].set_xlim(xlim)
    axes[2,0].set_ylim(np.array([0,3000]))
    
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
    

    
    # sealevel = icepack.interpolate(0*sed.bedrock, sed.Q)
    # sealevel = icepack.interpolate(conditional(sed.bedrock>0, sed.bedrock, 0), sed.Q)
    sealevel = copy.deepcopy(sed.bedrock)
    sealevel[sealevel<0] = 0
    
    axes[0,0].fill_between(sed.x*1e-3, sed.bedrock, sed.bedrock+sed.H, color='saddlebrown')
    axes[0,0].fill_between(sed.x*1e-3, sealevel, sed.bedrock+sed.H, color='cornflowerblue')    
    axes[0,0].plot(np.concatenate((x,x[::-1])), np.concatenate((s,b[::-1])), 'k')
    axes[0,0].fill_between(x, s, b, color='w', linewidth=1)
    axes[0,0].plot(sed.x*1e-3, sed.bedrock, 'k')
    
    sed.w = width(sed.x)
    
    axes[1,0].plot(np.array([L,L]), np.array([w[-1]/2,-w[-1]/2]), 'k')
    axes[1,0].plot(sed.x*1e-3, sed.w/2*1e-3, 'k')
    axes[1,0].plot(sed.x*1e-3, -sed.w/2*1e-3, 'k')
    axes[1,0].fill_between(sed.x*1e-3, sed.w/2*1e-3, -sed.w/2*1e-3, color='cornflowerblue')
    axes[1,0].fill_between(x, w/2, -w/2, color='white')
    
    
    axes[2,0].plot(x, u, 'k', label='average velocity')
    axes[2,0].plot(x, ub, 'k--', label='balance velocity')
    axes[2,0].legend(loc='upper left')
    
    
    axes[3,0].axis('off')
    
    axes[0,1].plot(sed.x*1e-3, sed.Qw*1e-6, 'k', label='runoff')
    axes[0,1].plot(sed.x*1e-3, sed.Qs*1e-5, 'k:', label=r'sediment flux ($\times 10$)')
    axes[0,1].legend(loc='upper left')
    
    
    axes[1,1].plot(sed.x*1e-3, sed.erosionRate, 'k', label='erosion rate')
    axes[1,1].plot(sed.x*1e-3, sed.depositionRate, 'k--', label='deposition rate')
    axes[1,1].legend(loc='upper left')
    
    
    # axes[2,1].plot(sed.x*1e-3, sed.hillslope, 'k', label='hillslope processes')
    axes[2,1].plot(sed.x*1e-3, sed.dHdt, 'k', label='$\partial H_s/\partial t$')
    axes[2,1].legend(loc='upper left')
    
    # # axes[2,1].plot(sed.x*1e-3, sed.dHdt)
    axes[3,1].plot(sed.x*1e-3, sed.H, 'k')

    

    plt.savefig(basename + '.png', format='png', dpi=150)
    plt.close()
    plt.ion()
    
    
