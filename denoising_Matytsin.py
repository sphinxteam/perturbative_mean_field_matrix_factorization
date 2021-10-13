import numpy as np
import pickle, sys
from Class.Denoising_Matytsin import Denoising_Matytsin

"""
In order to always have (1/m) Tr[S^2] -> 1, with S the shifted zero-trace version, we need:
 - Variance 1 for Wishart (usual), S = XX^T / sqrt(NM) (shifted)
 - Wigner: Variance 1
 - Uniform: Lmax = np.sqrt(3)
 - Orthogonal: sigma = 1
"""

if __name__== "__main__":

    S_type = sys.argv[1]
    log_NB_points = int(sys.argv[2])
    verbosity = int(sys.argv[3])
    NB_POINTS_x = int(1+2**log_NB_points)
    NB_POINTS_t = int(sys.argv[4])
    log_scale_t = bool(int(sys.argv[5]))
    assert S_type in ["wigner", "wishart", "uniform", "orthogonal"], "ERROR: Unknown matrix type for S"
    print("Starting S_type =", S_type)
    if S_type == "wishart":
        alpha = float(sys.argv[6])
        print("alpha =", alpha)
        if alpha > 1:
            epsilon_regularization = 10**(-float(sys.argv[7])) #The small regularization parameters for alpha > 1
    elif S_type == "uniform":
        #Lmax = float(sys.argv[6])
        Lmax = np.sqrt(3) 
        print("Lmax =", Lmax)
    elif S_type == "orthogonal":
        #sigma = float(sys.argv[6])
        sigma = 1.
        print("sigma =", sigma)

    print("Starting the computations with", NB_POINTS_x, "discretization points in x, using", NB_POINTS_t, "for t and log-scale boolean is", log_scale_t)

    Deltas = 10**np.linspace(2, -3, num=100, endpoint=True)
    epsilon_imag = 1e-8 #The small imaginary part
    parameters = {'S_type':S_type, 'NB_POINTS_x':NB_POINTS_x, 'log_scale_t':log_scale_t, 'NB_POINTS_t':NB_POINTS_t, 'epsilon_imag':epsilon_imag, 'verbosity':verbosity}
    if S_type == "wishart":
        parameters['alpha'] = alpha
        if alpha > 1:
            parameters['epsilon_regularization'] = epsilon_regularization
    elif S_type == "uniform":
        parameters['Lmax'] = Lmax
    elif S_type == "orthogonal":
        parameters['sigma'] = sigma

    denoiser = Denoising_Matytsin(Deltas_ = Deltas, parameters_ = parameters)
    result = denoiser.run()
    if verbosity >= 2:
        print("Deltas:", result['Deltas'])
        print("y_MMSEs:", result['y_MMSEs'])
        if 'Phis' in result.values():
            print("Phis:", result['Phis'])

    if S_type == "wigner" and verbosity >= 2:
        if verbosity >= 3:
            print("Expected y_MMSEs:", result['Deltas']/(1.+result['Deltas']))
        print("Max difference to the analytical MMSEs:",max(np.abs(result['y_MMSEs'] - result['Deltas']/(1.+result['Deltas']) ) ) )
    
    if S_type == "wigner":
        filename = "Data/Matytsin_wigner_log_NB_points_x_"+str(log_NB_points)+"_NB_POINTS_t_"+str(NB_POINTS_t)+"_log_scale_t_"+str(int(log_scale_t))+".pkl"
    elif S_type == "wishart":
        filename = "Data/Matytsin_wishart_alpha_"+str(alpha)+"_log_NB_points_x_"+str(log_NB_points)+"_NB_POINTS_t_"+str(NB_POINTS_t)+"_log_scale_t_"+str(int(log_scale_t))+".pkl"
    elif S_type == "uniform":
        #filename = "Data/Matytsin_uniform_Lmax_"+str(Lmax)+"_log_NB_points_x_"+str(log_NB_points)+"_NB_POINTS_t_"+str(NB_POINTS_t)+"_log_scale_t_"+str(int(log_scale_t))+".pkl"
        filename = "Data/Matytsin_uniform_Lmax_sqrt_3_log_NB_points_x_"+str(log_NB_points)+"_NB_POINTS_t_"+str(NB_POINTS_t)+"_log_scale_t_"+str(int(log_scale_t))+".pkl"
    elif S_type == "orthogonal":
        filename = "Data/Matytsin_orthogonal_alpha_"+str(sigma)+"_log_NB_points_x_"+str(log_NB_points)+"_NB_POINTS_t_"+str(NB_POINTS_t)+"_log_scale_t_"+str(int(log_scale_t))+".pkl"
    outfile = open(filename,'wb')
    result['parameters'] = parameters
    pickle.dump(result,outfile)