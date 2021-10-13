import time
import numpy as np
from scipy import integrate
from .functions import g_Y

class Denoising_Matytsin():

    def __init__(self, Deltas_, parameters_):
        self.parameters = parameters_
        self.Deltas = np.sort(Deltas_) 
        self.S_type = parameters_['S_type']
        assert self.S_type in ["wigner", "wishart", "orthogonal"], "ERROR: Unknown matrix type for S"
        if self.S_type == "wishart":
            self.alpha = parameters_["alpha"]
            if self.alpha > 1:
                self.epsilon_regularization = parameters_['epsilon_regularization']
        elif self.S_type == "orthogonal":
            self.sigma = parameters_["sigma"] #Scaling of the orthogonal matrix: Y = sqrt(M) sigma O + sqrt(Delta) Z

        self.epsilon_imag = parameters_['epsilon_imag'] #Small imaginary part used in the calculations 
        self.NB_POINTS_x = parameters_['NB_POINTS_x'] #Nb of points used to discretize the spectrum in the Deltas points, must be of the type 2**k + 1
        self.NB_POINTS_x_default = int(2**13 + 1) #Used to discretize the spectrum when not in a Delta or very small t point
        self.log_scale_t, self.NB_POINTS_t = parameters_['log_scale_t'], parameters_['NB_POINTS_t'] #Nb of points used to discretize t in [0, Delta_max], and do we use a log-scale or not

        if not(self.log_scale_t):
            smallest_t = 0
            if self.S_type == "wishart" and self.alpha > 1:
                smallest_t = self.epsilon_regularization
            self.ts = np.linspace(smallest_t, self.Deltas[-1], num = self.NB_POINTS_t)
        else:
            self.ts = 10**np.linspace(-4, np.log10(self.Deltas[-1]), num = self.NB_POINTS_t)
        
        if self.S_type == "wishart" and self.alpha > 1:
            #Then we have a regularization, so we should add many points close to it. We add NB_POINTS_t / 4 , in a linear scale.
            self.ts_around_regularization = np.linspace(self.epsilon_regularization, max(10*self.epsilon_regularization, 1e-4), num = int(self.NB_POINTS_t/4))
            self.ts = np.concatenate((self.ts_around_regularization, self.ts))
        else:
            #In the other cases we add 0 to the list ot ts
            self.ts = np.concatenate(([0.], self.ts))
        
        #I add as well the values of Deltas, and then sort it
        self.ts = np.concatenate((self.Deltas, self.ts))
        self.ts = np.sort(self.ts)
        self.ts = np.unique(self.ts) #I remove points that might be here twice

        self.indices_Deltas = np.ones_like(self.Deltas, dtype=int) # The indices of Deltas in the list ts
        for (i_D, Delta) in enumerate(self.Deltas):
            self.indices_Deltas[i_D] = np.where(self.ts == Delta)[0][0] #It should only come in at one spot

        assert not(self.S_type == "wishart" and self.alpha > 1 and min(self.ts) != self.epsilon_regularization), "ERROR: The smallest time is not the epsilon regularization!"
        
        self.verbosity = parameters_['verbosity']

    def preprocess(self):
        #We preprocess all rho(x,t) and v(x,t) for t between 0 and Delta_max
        if self.verbosity >= 1:
            print("Starting", self.S_type,"denoising. Preprocessing...")
            t0 = time.time()
        self.solutions = np.array([None for t in self.ts]) #Dictionary that contains all relevant objects
        step = int(self.ts.size / 20)
        for (i_t, t) in enumerate(self.ts):
            #For Wishart and alpha > 1 this does not include the case t = 0, we will compute rhoS' independently in this case
            more_precise = (t == 0 or t in self.Deltas) or (self.S_type == "wishart" and self.alpha > 1 and t <= max(self.ts_around_regularization))

            if self.verbosity >= 2:
                print("Starting preprocessing step", i_t+1, "/", self.ts.size, ", t =", t, "and precise =", more_precise)
            elif self.verbosity == 1 and i_t % step == 0:
                print("Starting preprocessing step", i_t+1, "/", self.ts.size, ", t =", t, "and precise =", more_precise)

            self.solutions[i_t] = self.find_rho_v(t, more_precise = more_precise)
            to_integrate = self.solutions[i_t]['rho']*(self.solutions[i_t]['v']**2 + (np.pi**2/3.)*self.solutions[i_t]['rho']**2)
            if self.solutions[i_t]['regular_grid']:
                self.solutions[i_t]['integral'] = integrate.romb(to_integrate, dx = self.solutions[i_t]['step'])
            else:
                self.solutions[i_t]['integral'] = integrate.simpson(to_integrate, self.solutions[i_t]['zs'])
            
        if self.verbosity >= 1:
            t1 = time.time()
            print("Preprocessing finished in", round(t1-t0,3), "seconds.")

    def compute_log_potentials(self, only_derivatives = True):
        log_potentials_Y, dlog_potentials_Y = np.zeros_like(self.Deltas), np.zeros_like(self.Deltas)
        log_potential_S = 0
        if self.verbosity >= 1:
            if only_derivatives:
                print("Starting derivatives of log potentials computations...")
            else:
                print("Starting log potentials and derivatives computations...")
            t0 = time.time()

        if not(only_derivatives):
            if not(self.S_type == "wishart" and self.alpha > 1):
                solution = self.solutions[0] #t = 0 is the first element of self.ts, it corresponds to S*
            else:
                solution = self.find_rho_v(t = 0, more_precise = True) #In this very specific case, we compute S' which was not preprocessed before

            step, zs, rho, regular_grid = solution['step'], solution['zs'], solution['rho'], solution['regular_grid']
            if regular_grid:
                one_integral = np.array([integrate.romb(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), dx = step) for z in zs]) #Integral over the last dimension 
                log_potential_S = integrate.romb(rho*one_integral, dx = step)
            else:
                one_integral = np.array([integrate.simpson(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), zs) for z in zs]) #Integral over the last dimension 
                log_potential_S = integrate.simpson(rho*one_integral, zs)

            if self.verbosity >= 2:
                print("Log potential of S computed!")
        
        step_count = max(int(self.Deltas.size / 10),1)
        for i_D in range(self.Deltas.size):
            if self.verbosity >= 2:
                print("Starting step", i_D+1, "/", self.Deltas.size)
            elif self.verbosity == 1 and i_D % step_count == 0:
                print("Starting step", i_D+1, "/", self.Deltas.size)

            if not(only_derivatives):
                solution = self.solutions[self.indices_Deltas[i_D]]
                step, zs, rho, regular_grid = solution['step'], solution['zs'], solution['rho'], solution['regular_grid']
                if regular_grid:
                    one_integral = np.array([integrate.romb(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), dx = step) for z in zs]) #Integral over the last dimension 
                    log_potentials_Y[i_D] = integrate.romb(rho*one_integral, dx = step)
                else:
                    one_integral = np.array([integrate.simpson(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), zs) for z in zs]) #Integral over the last dimension 
                    log_potentials_Y[i_D] = integrate.simpson(rho*one_integral, zs)

            #Now the differential of the log potential
            Delta_step = min(self.Deltas[i_D]*1e-2, 1e-1)
            Delta_next = self.Deltas[i_D] + Delta_step
            Delta_previous = self.Deltas[i_D] - Delta_step

            solution = self.find_rho_v(Delta_next, more_precise=True)
            step, zs, rho, regular_grid = solution['step'], solution['zs'], solution['rho'], solution['regular_grid']
            if regular_grid:
                one_integral = np.array([integrate.romb(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), dx = step) for z in zs]) #Integral over the last dimension 
                log_potential_next = integrate.romb(rho*one_integral, dx = step)
            else:
                one_integral = np.array([integrate.simpson(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), zs) for z in zs]) #Integral over the last dimension 
                log_potential_next = integrate.simpson(rho*one_integral, zs)

            solution = self.find_rho_v(Delta_previous, more_precise=True)
            step, zs, rho, regular_grid = solution['step'], solution['zs'], solution['rho'], solution['regular_grid']
            if regular_grid:
                one_integral = np.array([integrate.romb(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), dx = step) for z in zs]) #Integral over the last dimension 
                log_potential_previous = integrate.romb(rho*one_integral, dx = step)
            else:
                one_integral = np.array([integrate.simpson(rho*np.log(np.maximum(1e-8*np.ones_like(zs),np.abs(z - zs))), zs) for z in zs]) #Integral over the last dimension 
                log_potential_previous = integrate.simpson(rho*one_integral, zs)

            dlog_potentials_Y[i_D] = (log_potential_next - log_potential_previous)/(2*Delta_step)

        if self.verbosity >= 1:
            t1 = time.time()
            print("Log potentials and derivatives computations finished in", round(t1-t0,3), "seconds.")
        
        return log_potential_S, log_potentials_Y, dlog_potentials_Y
    
    #I compute the free entropies only for the Wishart and Wigner cases, otherwise I do not really care...
    def run(self):
        #The run function that computes all free energies and MSES for Deltas in Deltas
        y_MMSEs, Phis = np.zeros_like(self.Deltas), np.zeros_like(self.Deltas)
        self.preprocess()

        #We compute the free entropies only for the wishart and wigner cases
        only_derivatives = not(self.S_type in ["wishart", "wigner"])
        log_potential_S, log_potentials_Y, dlog_potentials_Y = self.compute_log_potentials(only_derivatives)

        #Now we combine all the previous calculations, taking into account the rescaling

        #Y_MMSE for all spectra
        for (i_D, Delta) in enumerate(self.Deltas):
            solution = self.solutions[self.indices_Deltas[i_D]]
            if solution['rescaled']:
                y_MMSEs[i_D] = Delta/2. - Delta * solution['integral'] - Delta**2 * dlog_potentials_Y[i_D] #Rescaled version
            else:
                y_MMSEs[i_D] = Delta - Delta**2 * solution['integral'] - Delta**2 * dlog_potentials_Y[i_D] #Unrescaled version
        
        #Computation of the free entropies 
        if not(only_derivatives):
            #We un-rescale all the integrals to obtain the original ones
            integrals = np.zeros_like(self.ts)
            for (i_t, t) in enumerate(self.ts):
                integrals[i_t] = self.solutions[i_t]['integral']
                if self.solutions[i_t]['rescaled']:
                    integrals[i_t] /= t

            #The integral term in phi
            current_t_index, current_integral = 0, 0. #The current t index, for the integral
            #If alpha > 1, the first time will be epsilon, which is good
            for (i_D, Delta) in enumerate(self.Deltas):
                current_integral += integrate.simps(integrals[current_t_index:self.indices_Deltas[i_D]], x = self.ts[current_t_index:self.indices_Deltas[i_D]])
                current_t_index = self.indices_Deltas[i_D]

                solution = self.solutions[self.indices_Deltas[i_D]]
                if self.S_type == "wigner":
                    Phis[i_D] = (-1./4)*current_integral -(1./4)* log_potentials_Y[i_D] #Un-Rescaled version for the log-pot
                    if solution['rescaled']:
                        Phis[i_D] += -(1./8)* np.log(Delta) #Rescaling term in the Y log potential
                elif self.S_type == "wishart": #Valid for alpha > 1 and alpha <= 1
                    Phis[i_D] = (-self.alpha/4)*current_integral -(self.alpha/4)* log_potentials_Y[i_D] #Un-Rescaled version for the log-pot
                    if solution['rescaled']:
                        Phis[i_D] += -(self.alpha/8)* np.log(Delta) #Rescaled version

            #Last we compute the additive term in the free entropy which is independent of Delta
            if self.S_type == "wigner":
                additive_term = log_potential_S / 4.
                solution = self.solutions[0] #t = 0 is the first element of self.ts, it corresponds to S
                step, zs, rho, regular_grid = solution['step'], solution['zs'], solution['rho'], solution['regular_grid']
                assert regular_grid, "ERROR: In Wigner, the grid of S is regular"
                additive_term += (-1./4)*integrate.romb(rho*zs**2, dx = step)

            elif self.S_type == "wishart" and self.alpha <= 1:
                #The first term + the log potential and moments of rho_S, using romb
                a = self.alpha
                if a < 1:
                    cste = (6. - 2.*a*np.log(np.pi*a) + 2.*np.log(a) - a*(3.+np.log(4)) + 2.*(1-a)**2*np.log(1-a)/a)/ 8
                elif a == 1:
                    cste = (1./8)*(3 - np.log(4.) - 2.*np.log(np.pi))
                solution = self.solutions[0] #t = 0 is the first element of self.ts, it corresponds to S
                step, zs, rho, regular_grid = solution['step'], solution['zs'], solution['rho'], solution['regular_grid']
                assert regular_grid, "ERROR: The grid of S must always be regular"

                additive_term = cste + (self.alpha/4.)*log_potential_S
                additive_term += (-np.sqrt(self.alpha)/2)*integrate.romb(rho*zs, dx = step)
                additive_term += -((self.alpha-1)/2)*integrate.romb(rho*np.log(np.maximum(zs, 1e-10*np.ones_like(zs))), dx = step)

            elif self.S_type == "wishart" and self.alpha > 1:
                #The first term + the log potential and moments of rho_S, using romb
                a = self.alpha
                cste = (6. - 2.*a*np.log(np.pi*a) + 2.*np.log(a) - a*(3.+np.log(4)) + 2.*(a-1)**2*np.log(a-1)/a)/ 8
                solution = self.find_rho_v(t = 0, more_precise = True) #In this case, we compute S' which was not preprocessed
                step, zs, rho, regular_grid = solution['step'], solution['zs'], solution['rho'], solution['regular_grid']
                assert regular_grid, "ERROR: The grid of S must always be regular"

                additive_term = cste + (1/(4.*self.alpha))*log_potential_S
                additive_term += (-1./(2*np.sqrt(self.alpha)))*integrate.romb(rho*zs, dx = step)
                additive_term += -self.alpha * (1.-1./self.alpha)**2 * (-1./4 + np.log(self.epsilon_regularization)/2.) / 4. #The regularization term in epsilon
            
            Phis += additive_term
            return {'Deltas':self.Deltas, 'y_MMSEs':y_MMSEs, 'Phis':Phis}

        else:
            return {'Deltas':self.Deltas, 'y_MMSEs':y_MMSEs}

    def find_rho_v(self, t, more_precise = False):
        rescaled = (t > 1) #For Delta > 1 we rescale the observations and consider instead Y' = Y / sqrt(Delta)
        NB_POINTS_x = self.NB_POINTS_x_default
        if more_precise:
            NB_POINTS_x = self.NB_POINTS_x
        regular_grid, zs, step = False, None, 0
        
        #We build the zs
        if self.S_type == "wigner":
            regular_grid = True
            if rescaled:
                zs, step = np.linspace(-2 * np.sqrt(1+1./t), 2*np.sqrt(1+1./t), num = NB_POINTS_x, retstep=True)
            else:
                zs, step = np.linspace(-2 * np.sqrt(1+t), 2*np.sqrt(1+t), num = NB_POINTS_x, retstep=True)
        
        elif self.S_type == "wishart":
            lmax_S, lmin_S = (1.+np.sqrt(self.alpha))**2/np.sqrt(self.alpha), (1.-np.sqrt(self.alpha))**2/np.sqrt(self.alpha)
            if self.alpha > 1 and t > 0 : #For t = 0 we must return S' so we keep lambdamin
                lmin_S = 0 #In this case we also have the delta peak in 0 that will be widened by the free addition
            lmax_Z, lmin_Z = 2*np.sqrt(t), -2.*np.sqrt(t) #Edges of the semi-circle
            #We find a rough estimate of the edges of the bulk since lmax(A+B) <= lmax(A) + lmax(B) and lmin(A+B) >=  lmin(A) + lmin(B)
            min_estimate, max_estimate = lmin_S + lmin_Z, lmax_S + lmax_Z 
            if rescaled:
                min_estimate /= np.sqrt(t)
                max_estimate /= np.sqrt(t)
            
            #To avoid some issues
            min_estimate -= 0.01 
            max_estimate += 0.01 

            #Now we compute the full gs
            zs, step = np.linspace(min_estimate, max_estimate, num = NB_POINTS_x, retstep=True)
            regular_grid = True

            #For small t > 0 and alpha > 1, we add many new points around t = 0 to increase the precision of the integrals
            if self.alpha > 1 and t > 0 and t < 1e-2*lmax_Z: #Then we add a lot of new points around t = 0, but the grid is no longer regular
                zs = np.concatenate((zs, np.linspace(lmin_Z, lmax_Z, num = int(NB_POINTS_x/2))))
                zs = np.sort(zs)
                zs = np.unique(zs) #Sorted and unique elements
                regular_grid = False
        
        elif self.S_type == "orthogonal":
            min_estimate, max_estimate = -self.sigma - 2.*np.sqrt(t), self.sigma + 2.*np.sqrt(t)
            if rescaled:
                min_estimate /= np.sqrt(t)
                max_estimate /= np.sqrt(t)
            #To avoid some issues
            min_estimate -= 0.01 
            max_estimate += 0.01 

            #Now we compute the full gs
            zs, step = np.linspace(min_estimate, max_estimate, num = NB_POINTS_x, retstep=True)
            regular_grid = True

        #Now we compute the Stieltjes transform, not shifted to have zero trace in the Wishart case
        gs = np.zeros_like(zs) + 1j*np.zeros_like(zs)
        for (i_z, z) in enumerate(zs):
            gs[i_z] = g_Y(self.S_type, rescaled, self.parameters, t, z + 1j*self.epsilon_imag, shifted_zero_trace=False)

        if self.S_type == "wishart" and self.alpha > 1 and t == 0: #In this case we return rho_S', and we have rho_S = (1-1/alpha) delta_0 + (1/alpha) rho_S'
            gs = self.alpha*gs + (self.alpha - 1.)/(zs + 1j*self.epsilon_imag)

        #From g we extract rho and V
        rho = np.imag(gs) / np.pi
        v = - np.real(gs)
        assert rho[0]**2 + rho[-1]**2 < 1e-7, "ERROR. For t = "+str(round(t,5))+", the densities at the edge are not zero, they are :"+str(round(rho[0],5))+" and "+str(round(rho[-1],5))

        return {'rescaled':rescaled, 'step':step, 'zs':zs, 'rho':rho, 'v':v, 'regular_grid':regular_grid}