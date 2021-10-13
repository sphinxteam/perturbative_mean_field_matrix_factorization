import numpy as np
import pickle
from scipy import linalg

class Denoising_TAP:
    def __init__(self, parameters_):
        self.parameters = parameters_
        self.S_type = parameters_['S_type']
        self.M = parameters_['M']
        assert self.S_type in ["wigner", "wishart"], "ERROR: Unknown matrix type for S"
        self.verbosity = parameters_['verbosity']
        self.Delta = parameters_["Delta"]
        self.rescaled = (self.Delta > 1) #For Delta > 1 we rescale the observations and consider instead Y' = Y / sqrt(Delta)
        if self.S_type == "wishart":
            self.alpha = parameters_["alpha"]
            self.N = int(self.M / self.alpha)
        elif self.S_type == "orthogonal":
            self.sigma = parameters_["sigma"] #Scaling of the orthogonal matrix: we have Y/sqrt(M) = sigma O + sqrt(Delta) Z / sqrt(M)
        
        self.order = parameters_["order_TAP"]
        assert self.order in [2, 3], "ERROR: Unimplemented order of perturbation"
        self.generate_data()
        self.find_solution()

    def generate_data(self):
        #Generate the data Y and the signal S*
        rng = np.random.default_rng()
        if self.S_type == "wishart":
            self.Xstar = np.random.normal(0.,1., (self.M, self.N)) 
            self.Sstar = (self.Xstar @ np.transpose(self.Xstar))/np.sqrt(self.N*self.M)
            np.fill_diagonal(self.Sstar, 0)
        elif self.S_type == "wigner":
            self.Sstar = rng.normal(0, 1, (self.M, self.M))
            self.Sstar = (self.Sstar + np.transpose(self.Sstar)) / np.sqrt(2.*self.M)
            np.fill_diagonal(self.Sstar, 0)
        Z = np.random.normal(0, 1, (self.M,self.M))
        Z = (Z + np.transpose(Z)) / np.sqrt(2.) #We symmetrize it, with variance Delta
        if not(self.rescaled):
            self.Y = np.sqrt(self.M)*self.Sstar + np.sqrt(self.Delta) * Z
        else: #Rescaled variables
            self.Y = np.sqrt(self.M/self.Delta)*self.Sstar + Z
        np.fill_diagonal(self.Y, 0)
        self.spec_Y, self.evec_Y = linalg.eigh(self.Y / np.sqrt(self.M))

    def get_y_mse(self):
        #We return the actual MSE (S* - <S>)^2
        if not(self.rescaled):
            S_estimated = (self.Y - self.Delta * self.g)/np.sqrt(self.M)
        else: #Rescaled variables
            S_estimated = np.sqrt(self.Delta/self.M)*(self.Y - self.g)
        np.fill_diagonal(S_estimated, 0.)
        return np.sum((self.Sstar - S_estimated)**2)/self.M
    
    def get_solution(self):
        return {'rescaled':self.rescaled, 'parameters':self.parameters, 'Sstar':self.Sstar, 'Y':self.Y, 'g':self.g, 'r':self.r, 'omega':self.omega, 'b':self.b}

    def save(self):
        #Saving the current state
        output = {'rescaled':self.rescaled, 'N':self.N, 'M':self.M, 'parameters':self.parameters, 'Xstar':self.Xstar, 'Y':self.Y, 'g':self.g, 'r':self.r, 'omega':self.omega, 'b':self.b}
        if self.S_type == "wigner":
            filename = "Data/tmp/TAP_denoising_wigner_M_"+str(self.M)+"_Delta_"+str(self.Delta)+".pkl"
        elif self.S_type == "wishart":
            filename = "Data/tmp/TAP_denoising_wishart_M_"+str(self.M)+"_alpha_"+str(self.alpha)+"_Delta_"+str(self.Delta)+".pkl"
        outfile = open(filename,'wb')
        pickle.dump(output,outfile)
        outfile.close()
    
    def find_solution(self):
        #We simply take g diagonal in the eigenbasis of Y, and then we remove its diagonal.
        spec_g = np.zeros(self.M)
        if not(self.rescaled):
            if self.order >= 2:
                spec_g += self.spec_Y / (self.Delta + 1)
            if self.order >= 3:
                spec_g += np.sqrt(self.alpha)*(self.Delta + 1 - self.spec_Y**2)/((self.Delta + 1)**3)
        else:
            if self.order >= 2:
                spec_g += self.spec_Y / (1.+ 1./self.Delta)
            if self.order >= 3:
                spec_g += (np.sqrt(self.alpha)/(self.Delta)**(3./2))*(1. + 1./self.Delta - self.spec_Y**2)/((1./self.Delta + 1)**3)
        self.g = np.sqrt(self.M) * self.evec_Y @ np.diag(spec_g) @ np.transpose(self.evec_Y)
        np.fill_diagonal(self.g, 0)
        self.compute_other_variables()
       
    def compute_other_variables(self):
        #Compute all other variables (omega, b, r) from the knowledge of g
        if not(self.rescaled):
            self.b = np.ones((self.M,self.M))
            np.fill_diagonal(self.b,0.)
            self.omega = self.Y - (self.Delta + self.b)*self.g
            np.fill_diagonal(self.omega,0.)
            self.r = 1./(self.Delta + self.b)
            np.fill_diagonal(self.r,0.)
        else: #Then we use rescaled = Y / sqrt(Delta) and all rescaled variables
            self.omega = self.Y - (1 + 1./self.Delta)*self.g #omega' = omega/sqrt(Delta)
            np.fill_diagonal(self.omega,0.)
            self.b = np.ones((self.M,self.M)) / self.Delta #b' = b/Delta
            np.fill_diagonal(self.b,0.)
            self.r = 1./(1. + 1. / self.Delta) * np.ones((self.M,self.M)) #r' = Delta r
            np.fill_diagonal(self.r,0.)

    def get_free_entropy(self):
        phi = 0.
        if self.S_type == "wishart":
            #The linear terms in the Lagrange multipliers 
            phi += (self.M/(2.*self.N))*(- np.mean(self.omega*self.g) - np.mean(self.b*(-self.r + self.g**2)/2.)) #Careful, we symmetrized so there is a factor 2 here
            if not(self.rescaled):
                #The Gaussian channel log partition,again with a global factor 1/2, using b = (variance_prior)^2 = 1 here 
                phi += (self.M/(2.*self.N))*(np.mean(- (self.Y- self.omega)**2 / (2*(1+self.Delta))) - (1./2)*np.log(2*np.pi*(1+self.Delta)))
                #The order 2 in eta, again factor 1/2 for mu < nu
                phi += (self.M/(4*self.N)) * np.mean(self.g**2 - self.r)
                #The order 3 if present
                if self.order >= 3:
                    phi += (1./(6*self.M*self.N**(3./2)))*np.sum(self.g * (self.g @ self.g)) #Tr[g^3]

            else: #Rescaled case
                #The Gaussian channel log partition,again with a global factor 1/2, using b = v^2
                phi += (self.M/(2.*self.N))*(np.mean(- (self.Y- self.omega)**2 / (2*(1./self.Delta + 1))) - (1./2)*np.log(self.Delta) - (1./2)*np.log(2*np.pi*(1./ self.Delta + 1)))
                #The order 2 in eta, again factor 1/2 for mu < nu, using r = 1 / (Delta + v^2)
                phi += - (self.M/(4*self.N*self.Delta)) * np.mean(self.g**2 - self.r) 
                #The order 3 if present
                if self.order >= 3:
                    phi += (1./(6*self.M*(self.Delta*self.N)**(3./2)))*np.sum(self.g * (self.g @ self.g)) #Tr[g^3]

        elif self.S_type == "wigner":
            #The linear terms in the Lagrange multipliers 
            phi += (1./2)*(- np.mean(self.omega*self.g) - np.mean(self.b*(-self.r + self.g**2)/2.)) #Careful, we symmetrized so there is a factor 2 here
            if not(self.rescaled):
                #The Gaussian channel log partition,again with a global factor 1/2, using b = (variance_prior)^2 = 1 here 
                phi += (1./2)*(np.mean(- (self.Y- self.omega)**2 / (2*(1+self.Delta))) - (1./2)*np.log(2*np.pi*(1+self.Delta)))
                #The order 2 in eta, again factor 1/2 for mu < nu
                phi += (1./4) * np.mean(self.g**2 - self.r)

            else: #Rescaled case
                #The Gaussian channel log partition,again with a global factor 1/2, using b = v^2
                phi += (1./2)*(np.mean(- (self.Y- self.omega)**2 / (2*(1./self.Delta + 1))) - (1./2)*np.log(self.Delta) - (1./2)*np.log(2*np.pi*(1./ self.Delta + 1)))
                #The order 2 in eta, again factor 1/2 for mu < nu, using r = 1 / (Delta + v^2)
                phi += - (1./(4*self.Delta)) * np.mean(self.g**2 - self.r) 
        
        return phi
