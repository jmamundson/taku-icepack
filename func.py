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
        self.n = 200 # number of grid points
        self.dt = 0.1 # time step size [a]
        self.L = 30e3 # initial domain length [m]
        self.Lsed = 80e3 # length of sediment model [m]
        self.sedDepth = 50 # depth to sediment, from sea level, prior to advance scenario [m]
        self.a_max = 5 # maximum mass balance rate [m a^{-1}]
        self.a_gradient = 0.012 # mass balance gradient
        self.ELA = 500 # equilbrium line altitude [m]

param = params()

class paramsSed:
    '''
    Parameters for the sediment transport model.
    '''
    
    def __init__(self):
        # Brinkerhoff model
        factor = 1
        self.h_eff = 0.1 # effective water thickness [m]
        self.c = 2e-12*factor # Brinkerhoff used 2e-12
        self.w = 500*factor # settling velocity [m a^{-1}]; Brinkerhoff used 500
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
    w_max = 12e3 - w_fjord # maximum valley width [m]
    
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
def find_endpoint_massflux(glac, L, dt):
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
    
    Qb = firedrake.assemble(glac.a * glac.w * dx) # balance flux [m^3 a^{-1}]
    
    index = np.argsort(glac.x.dat.data)
    
    Wt = glac.w.dat.data[index[-1]] # terminus width [m]
    Ht = glac.h.dat.data[index[-1]] # terminus thickness
    
    Ub = Qb / (Wt*Ht) # balance velocity [m a^{-1}]
    
    
    Ut = icepack.depth_average(glac.u_bar).dat.data[index[-1]] # centerline terminus velocity [m a^{-1}]
    
    dLdt = (alpha-1)*(Ub-Ut) # rate of length change [m a^{-1}]
    
    L_new = L + dLdt*dt
    
    return(L_new)



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
    dLdt = (L_new-L)/param.dt
    print('dL/dt: ' + "{:.02f}".format(dLdt) + ' m a^{-1}')

    return(L_new)





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
    
    Cs = firedrake.interpolate( 2/w * (4/(constant.A*w))**(1/3), w.function_space())

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
        self.h = initial_thickness(self.x, self.Q)                  
        self.s = icepack.compute_surface(thickness = self.h, bed = self.bed) 
        self.u = initial_velocity(self.x, self.V)
        self.w = width(self.x, Q=self.Q)
        self.b = icepack.interpolate(self.s-self.h, self.Q) # bottom of glacier; may or may not be floating
    
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
        sedH_interpolator = interp1d(self.x.dat.data[index], self.H.dat.data[index], kind='linear', fill_value='extrapolate')

        
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
        
        self.h_eff = icepack.interpolate(conditional(self.x>x_[-1], -(self.bedrock+self.H), self.h_eff), self.Q)
        
        self.delta_s = icepack.interpolate(1-exp(-self.H/10), self.Q)
        
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
        
        
        
        
        self.calcQs(x_[-1])
        
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
        
        LHS = ((H-H_)/dt*v + paramSed.k*(H.dx(0)+self.bedrock.dx(0))*self.delta_s*v.dx(0) + (self.erosionRate - self.depositionRate)*v ) * dx
        
        firedrake.solve(LHS == 0, H)
        
        H_new = icepack.interpolate(conditional(H<0.1, 0, H), self.Q)
        self.dHdt = icepack.interpolate((H_new - self.H)/dt, self.Q)
        
        self.H = H_new
        
        
    def calcQs(self, L):
        '''
        Calculates the width-averaged sediment flux [m^2 a^{-1}].

        L = glacier length
        '''
        
        
        # First sort and extract arrays that are needed. These will be put back 
        # into functions for solving the ODE. Note that we only need every 
        # other data point since we are using quadratic functions (at least I
        # think that is why the arrays contain 2n+1 points).
        index = np.argsort(self.x.dat.data)
        x = self.x.dat.data[index][::2]
        H = self.H.dat.data[index][::2]
        Qw = self.Qw.dat.data[index][::2]
        erosionRate = self.erosionRate.dat.data[index][::2]
        
        #eL = erosionRate[x==L]
        
        
        # Determine the domain to be used.
        index = H.nonzero()[0] # grid points that have sediment
        ind0 = index[0] # index of first grid point with sediment
        x0 = x[ind0] # first point of nonzero thickness
        xL = np.max(self.x.dat.data) # last grid point
        
        # Create shortened xarrays for putting back into functions.
        H_xarray = xarray.DataArray(H[index], [x[index]], 'x')
        Qw_xarray = xarray.DataArray(Qw[index], [x[index]], 'x')
        erosionRate_xarray = xarray.DataArray(erosionRate[index], [x[index]], 'x')
        
        
        # Create new mesh and set up new function space.       
        # First determine grid spacing and number of grid points in the new 
        # mesh.
        deltaX = x[1] 
        n = int((xL-x0)/deltaX)
        
        mesh1d = firedrake.IntervalMesh(n, x0, right=xL)
        mesh = firedrake.ExtrudedMesh(mesh1d, layers=1, name="mesh")
        Q = firedrake.FunctionSpace(mesh, "CG", 2, vfamily="R", vdegree=0)
        
        x_sc, z_sc = firedrake.SpatialCoordinate(mesh)
        x = firedrake.interpolate(x_sc, Q)

        # Put xarrays back into firedrake functions.
        H = icepack.interpolate(H_xarray, Q)
        Qw = icepack.interpolate(Qw_xarray, Q)
        erosionRate = icepack.interpolate(erosionRate_xarray, Q)
        
        
        # For some reason, the last grid point is often set to nan. Fix this.
        H.dat.data[-1] = H_xarray[-1]
        Qw.dat.data[-1] = Qw_xarray[-1]
        erosionRate.dat.data[-1] = erosionRate_xarray[-1]
        
        
        u = firedrake.TrialFunction(Q) # unknown that we wish to determine
        
        v = firedrake.TestFunction(Q)


        # left and right hand sides of variational form
        LHS = u * ( -v.dx(0) + paramSed.w/Qw*v ) * dx
        RHS = erosionRate * v *dx
        
        Qs = firedrake.Function(Q) # create empty function to hold the solution
        
        bc = firedrake.DirichletBC(Q, (0), 1)
        
        firedrake.solve(LHS == RHS, Qs, bcs=[bc])
        
        Qs_index = np.argsort(x.dat.data)
        
        
        Qs_interpolator = interp1d(x.dat.data[Qs_index], Qs.dat.data[Qs_index], fill_value = 0, bounds_error=False)
        
        Qs_xarray = xarray.DataArray(Qs_interpolator(self.x.dat.data), [self.x.dat.data], 'x')
        self.Qs = icepack.interpolate(Qs_xarray, self.Q)
        
        # create empty array for Qs and populate
        
        #self.Qs = icepack.interpolate(0*self.Qw, self.Q) # create empty array for Qs
        #self.Qs.dat.data[2*index[0]:] = Qs.dat.data[Qs_index] # populate with solution
        #self.Qs.dat.data[-len(Qs_index):] = Qs.dat.data[Qs_index] # populate with solution
        
    
    
        
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
    axes[1,0].set_ylim(np.array([-6, 6]))
    
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
    
    
