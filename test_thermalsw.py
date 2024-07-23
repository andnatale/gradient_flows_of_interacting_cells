from src.energy_transport import EnergyTransport
from src.utils import *
from pysdot import OptimalTransport
from pysdot.radial_funcs import RadialFuncUnit
from pysdot.radial_funcs import RadialFuncInBall
from pysdot.radial_funcs import RadialFuncPpWmR2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm



# Set up initial conditions
n = 140
coords0plus= np.linspace(0,1,n+1)
coords0 = .5*(coords0plus[:-1] + coords0plus[1:])
x0, y0 = np.meshgrid(coords0,coords0)

positions = np.zeros((n**2,2))
positions[:,0] = x0.flatten()
positions[:,1] = y0.flatten()
positions = positions[np.random.permutation(n**2),:]
positions += (np.random.rand(*positions.shape)-.5)*1e-3

Y0 = positions.copy()
V0 = np.zeros(Y0.shape)

mass = 1.5 # total mass
nu = np.ones(Y0.shape[0])
nu = nu/np.sum(nu)*mass 



# Power for the energy
gamma = 2.
energy_fun = lambda areas: np.sum(((nu/areas)**gamma)*areas)


#Static vortex
#alpha = 6. 
#sigma = .2
#R0 = np.linalg.norm(Y0-.5,axis=1)
#V0[:,0] = - alpha* np.exp(-R0**2/2/sigma**2) *( Y0[:,1]-.5)
#V0[:,1] = alpha*np.exp(-R0**2/2/sigma**2) *( Y0[:,0]-.5)
#s = 1. - alpha**2*sigma**2/np.sum(nu)/2*np.exp(-R0**2/sigma**2)


#Two vortices
alpha = 5.#6. 
sigma = .15
p1,p2 = .35,.65
R0 = np.linalg.norm(Y0-p1,axis=1)
V0[:,0] = - alpha* np.exp(-R0**2/2/sigma**2) *( Y0[:,1]-p1)
V0[:,1] = alpha*np.exp(-R0**2/2/sigma**2) *( Y0[:,0]-p1)
s = 1.-  alpha**2*sigma**2/np.sum(nu)/2*np.exp(-R0**2/sigma**2)  
R0 = np.linalg.norm(Y0-p2,axis=1)
V0[:,0] = V0[:,0] - alpha* np.exp(-R0**2/2/sigma**2) *( Y0[:,1]-p2)
V0[:,1] = V0[:,1] + alpha*np.exp(-R0**2/2/sigma**2) *( Y0[:,0]-p2)
s += -  alpha**2*sigma**2/np.sum(nu)/2*np.exp(-R0**2/sigma**2)

# Parameters
epsilon = 1./n/4
dt = 1/n/50 #2./n**2
t0 = 0
T  = 2. 

# Domain 
domain = make_square([0.,0.,1.,1.])


# Time integration
nus = nu*np.sqrt(s)
funprime,funsecond = power_energy_functions(gamma, epsilon, nus)
et = EnergyTransport(Y0,weights = np.sqrt(nu), domain=domain, funprime=funprime,funsecond=funsecond,
                                radial_func=RadialFuncInBall(),verbosity = True,positivity_backtrack = True)
Y_hist, psi_hist, E_hist,Eeps_hist,K_hist = hamiltonian_flow_integrator(et, epsilon,nu,t0,T, dt, V=V0, energy_fun = energy_fun)


# Modification for external potential (topography) quadratic potential centered at x0pot
# potcoeff = 0.
# x0pot = np.array([0.5,0.25])
# factor = poetcoff*s    
#Y_hist, psi_hist, E_hist,Eeps_hist,K_hist = hamiltonian_flow_integrator(et, epsilon,nu,t0,T, dt, V=V0, x0potential = x0pot, 
#                                                                                 factorpotential = factor, energy_fun = energy_fun)


# Plots
tvec = np.linspace(0,T-dt,40)
times = (tvec/dt).astype(int)
#times = [0, int(.05/dt), int(.2/dt), -1]

for k in range(len(times)):

    time = times[k]
    pd = PowerDiagram(Y_hist[time], psi_hist[time], domain, RadialFuncInBall())
    areas = pd.integrals()
 
    fig, ax = plt.subplots(1,2)
    scat = ax[0].scatter(Y_hist[time][:,0],Y_hist[time][:,1],s=.5, c = nu/areas, vmin=1.45, vmax=1.65)
    ax[0].axis('off')
    ax[0].set_aspect('equal', adjustable='box')
    fig.colorbar(scat,orientation="horizontal",ax=ax[0],fraction=0.043, pad=0.04)#(scat,ticks=[0,.2])
    
    scat = ax[1].scatter(Y_hist[time][:,0],Y_hist[time][:,1],s=.5, c = s, vmin=np.min(s), vmax=np.max(s))
    ax[1].axis('off')
    ax[1].set_aspect('equal', adjustable='box')
    filename = "data/thermal_test/thermalj{}.png".format(k)
    fig.set_size_inches(w=4., h=2.5)
    fig.tight_layout()
    fig.colorbar(scat,orientation="horizontal",ax=ax[1],fraction=0.043, pad=0.04)#(scat,ticks=[0,.2])
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(filename, dpi=300)
    plt.close()


tvec = np.arange(len(E_hist))*dt
fig, ax = plt.subplots()
ax.plot(tvec,Eeps_hist,'-k',label='(a)')
ax.plot(tvec,E_hist,'--k',label='(b)')
#ax.plot(tvec, K_hist,'r-',linewidth =2, label ='(c)')
ax.grid('on')
ax.legend()
ax.set_xlabel('$t$')
fig.set_size_inches(w=2.5, h=2.5)
fig.tight_layout()
fig.savefig("data/thermal_test/energythermal.png", dpi=300)



