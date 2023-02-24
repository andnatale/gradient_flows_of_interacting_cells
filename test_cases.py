from src.utils import *
from src.energy_transport import EnergyTransport
from pysdot import OptimalTransport
from pysdot.radial_funcs import RadialFuncUnit
from pysdot.radial_funcs import RadialFuncInBall

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint

from tabulate import tabulate

class TestCase:
  
    def __init__(self, pressure_power = 2):
    
        # Parameters for initial conditions
        self.gamma = pressure_power
        
        if self.gamma>1:
            self.C = 1/3
            self.t0 = 1/16
            self.T = 1.
            
            # Additional parameters for initial conditions
            self.alpha = 2./(2*(self.gamma-1)+2)
            self.beta = self.alpha/2.
            self.k = self.beta*(self.gamma-1)/(2.*self.gamma)


        if self.gamma ==1:
            self.t0 = 0.
            self.T = 1.
        
    def rho_init(self,x,y):
        if self.gamma>1:
           return self.t0**(-self.alpha)*np.maximum(0,(self.C**2 - (x**2 + y**2)*self.k*self.t0**(-2*self.beta)))**(1./(self.gamma-1))
        elif self.gamma==1:
           return 1.5 + np.cos(np.pi*x)*np.cos(np.pi*y)
         
    def flow_exact(self, X):
       
        if self.gamma>1:
            factor = (self.T/self.t0)**self.beta
            return X*factor
       
        if self.gamma==1:

            def vectorfield_trig(X,t):
                n = int(np.size(X)/2)
                X = X.reshape((n,2))
                x = X[:,0].copy()
                y = X[:,1].copy()
                G = X.copy()

                ft = np.exp(-np.pi**2*t)
                G[:,0] = (-ft*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)) /( 1.5 + ft* np.cos(np.pi*x)*np.cos(np.pi*y))
                G[:,1] = (-ft*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)) /( 1.5 + ft* np.cos(np.pi*x)*np.cos(np.pi*y))

                return -G.flatten()
       
            t= np.linspace(self.t0,self.T,1000)
    
            # split calculation in 4 to avoid memory error    
            k = 4
            klen = int(X.size/k)
            solk  = []
            for i in range(k):
                 sol = odeint(vectorfield_trig, X.flatten()[klen*i:klen*(i+1)],t )
                 solk.append(sol[-1,:])
            solT = np.concatenate(solk)

            return solT.reshape((int(np.size(X)/2),2))

    def initial_conditions(self,n,Ngrid=4000,verbose = True):

        if self.gamma>1:
            # Determine positions in reference domain
            R1 = np.sqrt(2/self.beta)*(self.C**(self.gamma/(self.gamma-1.)))
            coords0plus= np.linspace(-R1/2.,R1/2.,n+1)
            coords0 = .5*(coords0plus[:-1] + coords0plus[1:])
            x0, y0 = np.meshgrid(coords0,coords0)
            positions = np.zeros((n**2,2))
            positions[:,0] = x0.flatten()
            positions[:,1] = y0.flatten()
            positions = positions[np.random.permutation(n**2),:]

            costheta = np.max(np.abs(positions),axis=1)
            positions = positions/ np.linalg.norm(positions,axis=1).reshape([len(costheta),1])/(R1/2.)*costheta.reshape([len(costheta),1])*R1

            # Determine masses in reference domain 
            coords = np.linspace(-R1,R1,Ngrid)
            x, y = np.meshgrid(coords, coords)

            img =np.maximum(R1**2-(x**2+y**2),0)
            img[img>0]=1.
            domain = ScaledImage([-R1,-R1], [R1, R1], img)

            # Push forward positions to target density
            rref_squared = positions[:,0]**2 + positions[:,1]**2
            flag0 = (rref_squared >0)

            r_squared = (self.C**2 - ( self.C**(2*self.gamma/(self.gamma-1.))-self.beta*rref_squared[flag0]/2.)**((self.gamma-1.)/self.gamma)) * self.t0**(2.*self.beta)/self.k

            X = positions.copy()
            X[flag0,:] = X[flag0,:]*((r_squared/rref_squared[flag0])**(1/2)).reshape(((np.sum(flag0),1)))
 
        if self.gamma ==1:
            coords0plus= np.linspace(0,1,n+1)
            coords0 = .5*(coords0plus[:-1] + coords0plus[1:])
            x0, y0 = np.meshgrid(coords0,coords0)

            positions = np.zeros((n**2,2))
            positions[:,0] = x0.flatten()
            positions[:,1] = y0.flatten()
            positions = positions[np.random.permutation(n**2),:]
            positions += (0.5-np.random.rand(n**2,2))*(1/(5*n))
            X = positions.copy()

            # Determine masses in reference domain 
            coords = np.linspace(0,1,Ngrid)
            x, y = np.meshgrid(coords, coords)

            img = self.rho_init(x,y)
            domain = ScaledImage([0,0], [1, 1], img)
 
        pd = PowerDiagram(positions, np.zeros(n**2), domain, RadialFuncUnit())
        nu= pd.integrals()

        if verbose and self.gamma>1:
             mass = 2*np.pi*(self.C**(2*self.gamma/(self.gamma-1)))/self.beta
             print('Numerical and exact total mass:', np.sum(nu),mass)
       
        return X,nu
        

    def run_test(self,npoints,radial_func=RadialFuncInBall()):


        Y0,nu = self.initial_conditions(npoints)

        eps = 20./(npoints)
        tau = 20.*self.T/npoints**2

        Ntsteps = int((self.T)/tau)

        if self.gamma>1: 
           domain = make_square([-2.,-2.,2.,2.])
        elif self.gamma ==1:
           domain = make_square([0.,0.,1.,1.])
        
        funprime,funsecond = power_energy_functions(self.gamma, eps, nu) 
        et = EnergyTransport(Y0, domain=domain, funprime=funprime,funsecond=funsecond, radial_func=radial_func,verbosity = True,positivity_backtrack = True)
        Y_hist, psi_hist = gradient_flow_integrator(et,eps,nu,self.t0,self.T, tau)   

        self.Y_hist = Y_hist
        Yend = self.flow_exact(Y0)
        error = (np.sqrt(np.sum(((Y_hist[-1][:,0] - Yend[:,0])**2 +(Y_hist[-1][:,1] - Yend[:,1])**2)*nu)))
        return error , Y_hist 


    def make_convergence_table(self,Ntests = 4, factor = 9, radial_func = RadialFuncInBall()):

        nvec = factor*2**(np.arange(Ntests))
        errorvec = []
        for i in range(Ntests):
            print('Test '+str(i+1)+' running..')
            npoints = nvec[i]
            error, Y_hist =self.run_test(npoints,radial_func)
            errorvec.append(error)
        
        self.errorvec = errorvec
        self.nvec = nvec
        #self.Y_hist =Y_hist
 
        ratevec = (np.log(errorvec[1:]) - np.log(errorvec[:-1]))/(np.log(nvec[1:])-np.log(nvec[:-1])) 

        table = np.zeros((len(errorvec),3))
        table[:,0] = 1/nvec
        table[:,1] = errorvec
        table[1:,2] = -ratevec
        headers = ["$1/\sqrt{N}$","$E_\\varphi$","rate"] #,"$E_{\mathcal{U}}$","rate"]
        print(tabulate(table,headers,tablefmt="latex_raw",floatfmt=".2e", numalign="left"))        
                 
        #return errorvec, ratevec, Y_hist



test = TestCase(pressure_power=1.)
test.make_convergence_table(Ntests=4,factor =10, radial_func = RadialFuncInBall())


