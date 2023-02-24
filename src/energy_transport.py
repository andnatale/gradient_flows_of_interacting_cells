from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.domain_types import ScaledImage
from pysdot.radial_funcs import RadialFuncUnit
from pysdot.radial_funcs import RadialFuncInBall

from pysdot import PowerDiagram

from IPython.display import clear_output
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import numpy as np
import importlib


class EnergyTransport:
    def __init__(self, positions=None, weights=None, domain=None,funprime = None, funsecond=None,  radial_func=RadialFuncUnit(),
                 obj_max_dw=1e-8, verbosity=0, positivity_backtrack = 0):
        
        """ Class to solve energy transport problem:     |L_i| = fprime(psi)  """ 




        self.pd = PowerDiagram(positions, weights, domain, radial_func)
        self.obj_max_dw = obj_max_dw
        self.verbosity = verbosity
        self.positivity_backtrack = positivity_backtrack
        self.max_iter = 100
        self.delta_w = []
        self.funsecond =funsecond
        self.funprime =funprime

    def get_positions(self):
        return self.pd.positions
    

    def get_integrals(self):
        return self.pd.integrals()
    
    def get_centroids(self):
        return self.pd.centroids()

    def set_positions(self, new_positions):
        self.pd.set_positions(new_positions)

    def get_domain(self):
        return self.pd.get_domain()

    def set_domain(self, new_domain):
        self.pd.set_domain(new_domain)

    def get_weights(self):
        return self.pd.weights

    def set_weights(self, new_weights):
        self.pd.set_weights(new_weights)
   
    
    def adjust_weights(self, initial_weights=None, ret_if_err=False, relax=1.0):
         
        if not ( initial_weights is None ):
            self.set_weights( initial_weights )
            
        if self.pd.domain is None:
            TODO

        if self.funprime is None:
            self.funprime = lambda w : np.ones(w.shape[0])/w.shape[0]
        
        if self.funsecond is None:
            self.funrpime = lambda w : 0*w

        if self.pd.weights is None:
            self.set_weights(np.ones(self.pd.positions.shape[0]))

        old_weights = self.pd.weights + 0.0
        #if np.sum(old_weights<0): print("Negative weight")
        for _ in range(self.max_iter):
            # derivatives
            mvs = self.pd.der_integrals_wrt_weights(stop_if_void=True)
            if mvs.error:
                ratio = 0.5
                self.pd.set_weights(
                    (1 - ratio) * old_weights + ratio * self.pd.weights
                )
                mvs = self.pd.der_integrals_wrt_weights(stop_if_void=True)

                if (self.verbosity >= 1):
                    print("bim (going back)")
                continue
            old_weights = self.pd.weights.copy()

            #
            if self.pd.radial_func.need_rb_corr():
                mvs.m_values[0] *= 2
            mvs.v_values -= self.funprime(old_weights)

            #print(old_weights)
            N = old_weights.shape[0]
         
            Apy = csr_matrix((mvs.m_values,mvs.m_columns,mvs.m_offsets),shape=(N,N)) 
            Apy += csr_matrix((self.funsecond(old_weights),np.arange(N),np.arange(N+1)),shape=(N,N))
            A = Apy.copy()         

            b = mvs.v_values
            x = spsolve(A, b)

            # update weights
            if(self.positivity_backtrack>=1):
                 while np.min(self.pd.get_weights() - relax * x)<0:
                     relax = relax/2.
              
            self.pd.set_weights(self.pd.get_weights() - relax * x)
            relax = np.min((2 *relax,1.))
            nx = np.max(np.abs(x))
            
            if (self.verbosity >= 1):
                print("max dw:", nx)
            
            
            self.delta_w.append(nx)

            if nx < self.obj_max_dw:
                break
                
        return False

