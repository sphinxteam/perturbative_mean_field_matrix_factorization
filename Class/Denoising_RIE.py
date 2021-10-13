"""
The L2 Rotationally Invariant Estimator
Since we have a Gaussian additive noise, it is quite simple to build
"""

import numpy as np
from scipy import linalg, optimize
from numpy.polynomial import Polynomial
from .functions import g_Y, random_orthogonal

class Denoising_RIE():

    def __init__(self, Delta_, parameters_):
        self.Delta = Delta_
        self.rescaled = (self.Delta > 1)
        self.parameters = parameters_
        self.S_type = parameters_['S_type']
        self.M = parameters_['M']
        assert self.S_type in ["wigner", "wishart", "uniform", "orthogonal"], "ERROR: Unknown matrix type for S"
        if self.S_type == "wishart":
            self.alpha = parameters_["alpha"]
            self.N = int(self.M / self.alpha)
        elif self.S_type == "uniform":
            self.Lmax = parameters_["Lmax"] #Uniform distribution in [-Lmax,Lmax]
        elif self.S_type == "orthogonal":
            self.sigma = parameters_["sigma"] #Scaling of the orthogonal matrix: we have Y/sqrt(M) = sigma O + sqrt(Delta) Z / sqrt(M)
        self.epsilon_imag = parameters_['epsilon_imag'] #Small imaginary part used in the calculations 
        self.verbosity = parameters_['verbosity']
        self.generate_data()
        if self.verbosity >= 2:
            print("Data generated !")
    
    def generate_data(self):
        #Generates the signal S*, here we denote S* = H*/sqrt(M)
        #Note that the signal is shifted to have zero-trace in all cases, which does not affect the MMSE
        rng = np.random.default_rng()
        if self.S_type == "wishart":
            self.Xstar = rng.normal(0, 1, (self.M, self.N))
            self.Hstar = (self.Xstar @ np.transpose(self.Xstar))/np.sqrt(self.N) - np.sqrt(self.N)*np.eye(self.M)
        elif self.S_type == "wigner":
            self.Hstar = rng.normal(0, 1, (self.M, self.M))
            self.Hstar = (self.Hstar + np.transpose(self.Hstar)) / np.sqrt(2.)
        elif self.S_type == "uniform":
            #Generate the diagonal 
            diag = rng.uniform(-self.Lmax, self.Lmax, self.M)
            #Generate a random rotation matrix
            U = random_orthogonal(self.M, rng)
            self.Hstar = np.sqrt(self.M) * U @ np.multiply(diag, np.transpose(U))
        elif self.S_type == "orthogonal":
            #We generate a random *symmetric* orthogonal matrix
            diag = 2*rng.integers(0, 1, size = self.M, endpoint=True) - 1 #Random vector of +- 1
            U = random_orthogonal(self.M, rng)
            self.Hstar = np.sqrt(self.M) * self.sigma * U @ np.multiply(diag, np.transpose(U))

        #Adding Gaussian noise
        Z = np.random.normal(0, 1, (self.M,self.M))
        Z = (Z + np.transpose(Z)) / np.sqrt(2.) #We symmetrize it
        self.Y = self.Hstar + np.sqrt(self.Delta) * Z
        if self.rescaled:
            self.Y /= np.sqrt(self.Delta)
        #Here we take the spectrum of Y / sqrt(M)
        self.spec_Y, self.evec_Y = linalg.eigh(self.Y / np.sqrt(self.M))

    def find_rho_v(self, zr):
        gs = g_Y(self.S_type, self.rescaled, self.parameters, self.Delta,  zr + 1j*self.epsilon_imag)
        return np.imag(gs) / np.pi, -np.real(gs)

    def run(self, get_estimator = False):
        y_MSE = 0.
        vs = np.zeros(self.M) #The list of values of v_Y
        xis = np.zeros(self.M) #The denoised eigenvalues
        for mu in range(self.M):
            _, vs[mu] = self.find_rho_v(self.spec_Y[mu])
        if not(self.rescaled):
            xis = self.spec_Y - 2*self.Delta*vs
        else: #Rescaled
            xis = np.sqrt(self.Delta)*(self.spec_Y - 2*vs)
        
        estimator = np.sqrt(self.M) * self.evec_Y @ np.diag(xis) @ np.transpose(self.evec_Y)
        y_MSE = np.mean((self.Hstar - estimator)**2) #Recall H* = sqrt(M) S* 
        if not(get_estimator):
            return y_MSE

        return {'y_MSE':y_MSE,'Y':self.Y, 'rescaled':self.rescaled, 'Hstar':self.Hstar, 'estimator':estimator} 