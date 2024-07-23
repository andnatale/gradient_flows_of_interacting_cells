from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.domain_types import ScaledImage
from pysdot.radial_funcs import RadialFuncUnit
from pysdot.radial_funcs import RadialFuncInBall
from pysdot.radial_funcs import RadialFuncPpWmR2
#from pysdot.util import FastMarching
#from pysdot import OptimalTransport
from pysdot import PowerDiagram

from IPython.display import clear_output
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import numpy as np
import importlib

def make_square(box=[0, 0, 1, 1]):
    domain = ConvexPolyhedraAssembly()
    domain.add_box([box[0], box[1]], [box[2], box[3]])
    return domain


def power_energy_functions(power,eps,nu):
    """
    Provides functions to solver energy trasport problem 
                                           
       (*)      P(nu_i/|L_i|) = psi_i/(2*eps)  or equivalently   |L_i| = fprime_i(psi_i)
    
    where P(r) = r^power is the pressure, psi_i i-th potential defining the Laguerre cell ( |x-x_i|^2 - psi_i \leq |x-x_j|^2 - psi_j) 

    Parameters

    power : float, power defining pressure
    eps : float, regularisation parameter
    nu : numpy array, vector of masses

    Returns
   
    funprime : function defined by (*)
    funsecond : derivative of funprime
    """
  
    p = power
    funprime = lambda psi: nu*(psi/(2*eps))**(-1./p)
    funsecond = lambda psi: nu/(2*p*eps)*(psi/(2*eps))**(-1./p-1)
    return funprime, funsecond


def gradient_flow_integrator(et, eps,nu,t0,T,tau,Y=None, x0potential=None,factorpotential=1., energy_fun = None):  
    """
    Time integrator for the gradient flow problem

                   x_i' = - |L_i|(x_i-b_i)/(eps*nu_i) - factor*(x_i - x0potential)
     
    whrere L_i is the ith Laguerre cell of the point cloud (x_i)_i generated via an energy transport problem 
     
    Parameters 

    et : energy transport object
    eps : float, regularization parameter
    nu : vector of masses (gradient flow metric)
    t0 : float, initial time
    T : float, max time
    tau : time step
    x0potential: numpy array of size 2 with coordinates of quadratic potential minimum
    factorpotential: float or array of size number of points, scaling factor for potential forces  
    energy_fun : function of areas returning energy of tessellation
    """ 
      
    
    Y_hist =[]
    psi_hist = []
    energy_hist = []
    energy_eps_hist =[]
    t=t0
    
    # Initialize positions
    if Y is None:  
         Y = et.get_positions().copy()
    else:
         et.set_positions(Y)
  
    # Power diagram for tessellation energy computation
    pdsq = PowerDiagram(Y, domain = et.get_domain(), radial_func=RadialFuncPpWmR2())

    while t+tau/2.<=T:
        et.adjust_weights()
        Y_hist.append(Y)
        psi_hist.append(et.get_weights())
        
        B = et.get_centroids()
        Areas = et.get_integrals()
        dissip_rate =  (Areas/nu).reshape([Y.shape[0],1])/eps


        if not(energy_fun is None): 
            energy_hist.append(energy_fun(Areas))
            pdsq.set_weights(psi_hist[-1])
            pdsq.set_positions(Y)
            Eeps = (np.dot(psi_hist[-1],Areas) - np.sum(pdsq.integrals()))/(2*eps) + energy_hist[-1]
            energy_eps_hist.append(Eeps)
        if not(x0potential is None): 
            B = (dissip_rate*B+np.tensordot(factorpotential,x0potential,axes =0))/(dissip_rate+factorpotential) 
            dissip_rate = dissip_rate + factorpotential
        Y = B + np.exp(-tau*dissip_rate)*(Y-B)
 
        et.set_positions(Y)
        t +=tau
        print('Time t={}'.format(t))
    
    if energy_fun is None:
        return Y_hist, psi_hist
    else:
        return Y_hist, psi_hist, ernergy_hist, energy_eps_hist, kinetic_hist




def hamiltonian_flow_integrator(et, eps,nu,t0,T,tau, Y = None, V=None, x0potential=None,factorpotential=1., energy_fun = None):  
    """
    Time integrator for the hamiltonian dynamics

                   x_i'' = - |L_i|(x_i-b_i)/(eps*nu_i) - factor*(x_i - x0potential)
     
    whrere L_i is the ith Laguerre cell of the point cloud (x_i)_i generated via an energy transport problem 
     
    Parameters 

    et : energy transport object
    eps : float, regularization parameter
    nu : vector of masses (gradient flow metric)
    t0 : float, initial time
    T : float, max time
    tau : time step
    Y: numpy array of positions shape (#points, 2)
    V: numpy array of velocity  shape (#points, 2)
    x0potential: numpy array of size 2 with coordinates of quadratic potential minimum
    factorpotential: float or array of size number of points, scaling factor for potential forces  
    energy_fun : function of areas returning energy of tessellation
    """ 
      
    
    Y_hist =[]
    V_hist = []
    psi_hist = []
    energy_hist = []
    energy_eps_hist =[]
    kinetic_hist = []
    t=t0

    # Initialize positions and velocities
    if Y is None:
         Y = et.get_positions().copy()
    else:
         et.set_positions(Y)
    if V is None:
         V = Y*0.


    # Power diagram for tessellation energy computation
    pdsq = PowerDiagram(Y, domain =et.get_domain(), radial_func=RadialFuncPpWmR2())

    while t+tau/2.<=T:
        et.adjust_weights()
        Y_hist.append(Y)
        psi_hist.append(et.get_weights())
        
        B = et.get_centroids()
        Areas = et.get_integrals()
        dissip_rate =  (Areas/nu).reshape([Y.shape[0],1])/eps

        if not(energy_fun is None):
            energy_hist.append(energy_fun(Areas))
            pdsq.set_weights(psi_hist[-1])
            pdsq.set_positions(Y)
            Eeps = (np.dot(psi_hist[-1],Areas) - np.sum(pdsq.integrals()))/(2*eps) + energy_hist[-1]
            energy_eps_hist.append(Eeps)
            kinetic_hist.append(np.dot(nu,np.sum(V*V,axis=1))/2.)
        if not(x0potential is None):
            B = (dissip_rate*B+np.tensordot(factorpotential,x0potential,axes=0))/(dissip_rate+factorpotential)
            dissip_rate = dissip_rate + factorpotential

        Y =  B  + np.cos(tau*np.sqrt(dissip_rate))*(Y- B ) +  np.sin(tau*np.sqrt(dissip_rate))*V/np.sqrt(dissip_rate)
        V = np.sqrt(dissip_rate)* (-np.sin(tau*np.sqrt(dissip_rate))*(Y- B )) +  np.cos(tau*np.sqrt(dissip_rate))*V


        et.set_positions(Y)
        t +=tau
        print('Time t={}'.format(t))

    if energy_fun is None:
        return Y_hist, psi_hist
    else:
        return Y_hist, psi_hist, energy_hist, energy_eps_hist, kinetic_hist

