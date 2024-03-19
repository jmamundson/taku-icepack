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
files = sorted(glob.glob('./results/mature/*h5'))
filesSed = sorted(glob.glob('./results/mature/*.pickle'))

plt.ioff()

b_t = np.zeros(len(files))
x_t = np.zeros(len(files))
h_0 = np.zeros(len(files))

# lastfile = 199
#for j in np.linspace(0, lastfile, int(lastfile/1 + 1), endpoint=True, dtype=int): #np.arange(0,300):#len(files)):
# for j in np.arange(0:len(files)): #np.linspace(0, 99, 100, endpoint=True, dtype=int): #np.arange(0,300):#len(files)):
for j in np.arange(0,len(files)):
    
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
    
    
    L = x[-1]
    
    with open(filesSed[j], 'rb') as file:
            sed = pickle.load(file)
            file.close()    
    

    fig, axes = plt.subplots(4, 2)
    fig.set_figwidth(10)
    fig.set_figheight(8)
    
    xlim = np.array([0,40])
    
    axes[0,0].set_xlabel('Longitudinal Coordinate [km]')
    axes[0,0].set_ylabel('Elevation [m]')
    axes[0,0].set_xlim(xlim)
    axes[0,0].set_ylim(np.array([-500,2500]))
    
    axes[1,0].set_xlabel('Longitudinal Coordinate [km]')
    axes[1,0].set_ylabel('Transverse Coordinate [km]')
    axes[1,0].set_xlim(xlim)
    axes[1,0].set_ylim(np.array([-4, 4]))
    
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
    
    
    
    sed.w = func.width(sed.x)
    sed.sealevel = np.zeros(len(sed.x))
    sed.sealevel[sed.zBedrock>0] = sed.zBedrock[sed.zBedrock>0]
    
    
    axes[0,0].fill_between(sed.x*1e-3, sed.zBedrock, sed.zBedrock+sed.H, color='saddlebrown')
    axes[0,0].fill_between(sed.x*1e-3, sed.sealevel, sed.zBedrock+sed.H, color='cornflowerblue')    
    axes[0,0].plot(np.concatenate((x,x[::-1])), np.concatenate((s,b[::-1])), 'k')
    axes[0,0].fill_between(x, s, b, color='w', linewidth=1)
    axes[0,0].plot(sed.x*1e-3, sed.zBedrock, 'k')
    
    
    
    axes[1,0].plot(np.array([L,L]), np.array([w[-1]/2,-w[-1]/2]), 'k')
    axes[1,0].plot(sed.x*1e-3, sed.w/2*1e-3, 'k')
    axes[1,0].plot(sed.x*1e-3, -sed.w/2*1e-3, 'k')
    axes[1,0].fill_between(sed.x*1e-3, sed.w/2*1e-3, -sed.w/2*1e-3, color='cornflowerblue')
    axes[1,0].fill_between(x, w/2, -w/2, color='white')
    
    
    axes[2,0].plot(x, u, 'k')
    
    axes[3,0].axis('off')
    
    axes[0,1].plot(sed.x*1e-3, sed.Qw*1e-6, 'k', label='subglacial discharge')
    axes[0,1].plot(sed.x*1e-3, sed.Qs*1e-6, 'k:', label='sediment flux')
    axes[0,1].legend()
    
    
    axes[1,1].plot(sed.x*1e-3, sed.erosionRate, 'k', label='erosion rate')
    axes[1,1].plot(sed.x*1e-3, sed.depositionRate, 'k--', label='deposition rate')
    axes[1,1].legend()
    
    
    axes[2,1].plot(sed.x*1e-3, sed.hillslope, 'k', label='hillslope processes')
    axes[2,1].plot(sed.x*1e-3, sed.depositionRate-sed.erosionRate, 'k:', label='deposition-erosion')
    axes[2,1].legend()
    
    axes[3,1].plot(sed.x*1e-3, sed.H, 'k')
    
    b_t[j] = b[-1]
    x_t[j] = x[-1]
    h_0[j] = h[0]
    
    plt.savefig(files[j][:-2] + 'png', format='png', dpi=150)
    plt.close()