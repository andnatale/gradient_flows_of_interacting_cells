''' 
Porous medium test case: initial uniform cross distribution and quadratic potential
Convergence towards Barenblatt profile
'''

from src.energy_transport import EnergyTransport
from src.utils import *
from pysdot import OptimalTransport

from pysdot.radial_funcs import RadialFuncUnit
from pysdot.radial_funcs import RadialFuncInBall
from pysdot.radial_funcs import RadialFuncPpWmR2
from pysdot import PowerDiagram

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# Set up initial conditions (cross distribution)
n =150 
coords0plus= np.linspace(0,1,n+1)
coords0 = .5*(coords0plus[:-1] + coords0plus[1:])
x0, y0 = np.meshgrid(coords0,coords0)
positions = np.zeros((n**2,2))
positions[:,0] = x0.flatten()
positions[:,1] = y0.flatten()
positions = positions[np.random.permutation(n**2),:]
flag_cross = np.logical_or(np.logical_and(positions[:,0]<.66,positions[:,0]>.33),np.logical_and(positions[:,1]<.66,positions[:,1]>.33))


Y0 = positions[flag_cross,:].copy()
nu = np.ones(Y0.shape[0])
nu = nu/np.sum(nu)*0.12
mass = np.sum(nu)

# Power for the energy
gamma = 2.
energy_fun = lambda areas: np.sum(((nu/areas)**gamma)*areas)

# Exact asymptotic energy
Efinal = 4*np.pi/3*((mass/(2*np.pi))**1.5)  

# Paremeters for integrator
epsilon = 1./n
t0 = 0.
T = 4.
dt = 1./n/4

# Domain and quadratic potential minimum
domain = make_square([-5.,-5.,6.,6.])
x0potential=np.array([.5,.5])

# Time integration
funprime,funsecond = power_energy_functions(gamma, epsilon, nu)
et = EnergyTransport(Y0, domain=domain, funprime=funprime,funsecond=funsecond, radial_func=RadialFuncInBall(),verbosity = True,positivity_backtrack = True)
Y_hist, psi_hist, E_hist,Eeps_hist = gradient_flow_integrator(et,epsilon,nu,t0,T, dt,x0potential=x0potential, energy_fun = energy_fun)

# Plots
#tvec = np.linspace(t0,2,30)**2
#times = (tvec/dt).astype(int)
times = [0, int(.05/dt), int(.2/dt), -1]

for k in range(len(times)):
    
    time = times[k]
    pd = PowerDiagram(Y_hist[time], psi_hist[time], domain, RadialFuncInBall())
    areas = pd.integrals()
    fig, ax = plt.subplots()
  
    scat = ax.scatter(Y_hist[time][:,0],Y_hist[time][:,1],s=.5, c = nu/areas, vmin=0., vmax=.25)
    ax.axis('off')
    ax.set_xlim([-.2,1.2])
    ax.set_ylim([-.2,1.2])
    ax.set_aspect('equal', adjustable='box')
    filename = "data/cross_test/pcross{}.png".format(k)
    fig.set_size_inches(w=2.5, h=2.5)
    fig.tight_layout()
    fig.colorbar(scat)#(scat,ticks=[0,.2])
    fig.savefig(filename, dpi=300)

tvec = np.arange(len(E_hist))*dt
fig, ax = plt.subplots()

ax.plot(tvec,Eeps_hist,'-k',label='(a)')
ax.plot(tvec,E_hist,'--k',label='(b)')
ax.plot(tvec,np.ones(tvec.shape)*Efinal ,'r-',linewidth =2, label ='(c)')
ax.grid('on')
ax.legend()
ax.set_xlabel('$t$')
fig.set_size_inches(w=2.5, h=2.5)
fig.tight_layout()
fig.savefig("data/cross_test/energycross.png", dpi=300)




