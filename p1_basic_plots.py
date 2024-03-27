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
constant = constants()
param = params()

import glob
import pickle

#%% 
base = 'premature'
files = sorted(glob.glob('./results/' + base + '/' + base + '*.h5'))
filesSed = sorted(glob.glob('./results/' + base + '/' + base + '*.pickle'))

plt.ioff()

b_t = np.zeros(len(files))
x_t = np.zeros(len(files))
h_0 = np.zeros(len(files))
L = np.zeros(len(files))

dt = param.dt
t = np.linspace(0,(len(files)-1)*dt, len(files))

for j in np.arange(0, len(files)):
    with firedrake.CheckpointFile(files[j], "r") as checkpoint:
        mesh = checkpoint.load_mesh(name="mesh")
        x = checkpoint.load_function(mesh, name="position")
        h = checkpoint.load_function(mesh, name="thickness")
        s = checkpoint.load_function(mesh, name="surface")
        u = checkpoint.load_function(mesh, name="velocity")
        b = checkpoint.load_function(mesh, name="bed")
        w = checkpoint.load_function(mesh, name="width")
    
    
    index = np.argsort(x.dat.data)
    x = x.dat.data[index]*1e-3
    h = h.dat.data[index]
    s = s.dat.data[index]
    u = icepack.depth_average(u).dat.data[index]
    b = b.dat.data[index]
    w = w.dat.data[index]*1e-3
    
    
    L[j] = x[-1]
    
    
plt.plot(t, L)