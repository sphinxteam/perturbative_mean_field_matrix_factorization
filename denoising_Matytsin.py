import pickle, sys
import numpy as np
from Class.Denoising_Matytsin import Denoising_Matytsin

"""
This script performs the computation of the denoising MMSE and free entropy, using the Matytsin formalism.

In order to always have (1/m) Tr[(S - (1/m)Tr[S])^2] -> 1,we need:
 - Variance 1 for Wishart, with S = XX^T / sqrt(NM)
 - Wigner: Variance 1.
 - Orthogonal: sigma = 1
"""

if __name__== "__main__":

    S_type = sys.argv[1] #Type of matrix to denoise
    log_NB_points = int(sys.argv[2]) #Number of points used in the discretization of the spectrum
    verbosity = int(sys.argv[3])
    NB_POINTS_x = int(1+2**log_NB_points)
    NB_POINTS_t = int(sys.argv[4]) #Number of points used in the time discretization
    log_scale_t = bool(int(sys.argv[5])) #Do we use a log scale for the time grid

    assert S_type in ["wigner", "wishart", "orthogonal"], "ERROR: Unknown matrix type for S"
    print("Starting S_type =", S_type)
    if S_type == "wishart":
        alpha = float(sys.argv[6])
        print("alpha =", alpha)
        if alpha > 1:
            epsilon_regularization = 10**(-float(sys.argv[7])) #The small regularization parameter in the Matytsin integral, for alpha > 1
    elif S_type == "orthogonal":
        sigma = float(sys.argv[6])
        print("sigma =", sigma)

    print("Starting the computations with", NB_POINTS_x, "discretization points in x, using", NB_POINTS_t, "for t and log-scale boolean is", log_scale_t)

    Deltas = 10**np.linspace(2, -3, num=100, endpoint=True) #Values of the noise parameter

    epsilon_imag = 1e-8 #The small imaginary part used in the Stieltjes transform computations
    parameters = {'S_type':S_type, 'NB_POINTS_x':NB_POINTS_x, 'log_scale_t':log_scale_t, 'NB_POINTS_t':NB_POINTS_t, 'epsilon_imag':epsilon_imag, 'verbosity':verbosity}
    if S_type == "wishart":
        parameters['alpha'] = alpha
        if alpha > 1:
            parameters['epsilon_regularization'] = epsilon_regularization
    elif S_type == "orthogonal":
        parameters['sigma'] = sigma

    denoiser = Denoising_Matytsin(Deltas_ = Deltas, parameters_ = parameters)
    result = denoiser.run()
    result['parameters'] = parameters

    if S_type == "wigner":
        filename = "Data/Matytsin_wigner_log_NB_points_x_"+str(log_NB_points)+"_NB_POINTS_t_"+str(NB_POINTS_t)+"_log_scale_t_"+str(int(log_scale_t))+".pkl"
    elif S_type == "wishart":
        filename = "Data/Matytsin_wishart_alpha_"+str(alpha)+"_log_NB_points_x_"+str(log_NB_points)+"_NB_POINTS_t_"+str(NB_POINTS_t)+"_log_scale_t_"+str(int(log_scale_t))+".pkl"
    elif S_type == "orthogonal":
        filename = "Data/Matytsin_orthogonal_sigma_"+str(sigma)+"_log_NB_points_x_"+str(log_NB_points)+"_NB_POINTS_t_"+str(NB_POINTS_t)+"_log_scale_t_"+str(int(log_scale_t))+".pkl"
    outfile = open(filename,'wb')
    pickle.dump(result,outfile)