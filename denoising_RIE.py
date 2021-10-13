import numpy as np
import pickle, sys, time
from Class.Denoising_RIE import Denoising_RIE


"""
In order to always have (1/m) Tr[S^2] -> 1, with S the shifted zero-trace version, we need:
 - Variance 1 for Wishart (usual), S = XX^T / sqrt(NM) (shifted)
 - Wigner: Variance 1
 - Uniform: Lmax = np.sqrt(3)
 - Orthogonal: sigma = 1
 """

if __name__== "__main__":

    S_type = sys.argv[1]
    M = int(sys.argv[2])
    verbosity = int(sys.argv[3])
    NB_INSTANCES = int(sys.argv[4])
    assert S_type in ["wigner", "wishart", "uniform", "orthogonal"], "ERROR: Unknown matrix type for S"
    if S_type == "wishart":
        alpha = float(sys.argv[5])
        print("alpha =", alpha)
    elif S_type == "uniform":
        #Lmax = float(sys.argv[5])
        Lmax = np.sqrt(3.) 
        print("Lmax =", Lmax)
    elif S_type == "orthogonal":
        #sigma = float(sys.argv[5])
        sigma = 1.
        print("sigma =", sigma)

    Deltas = 10**np.linspace(2, -3, num=100, endpoint=True)
    epsilon_imag = 1e-8 #The small imaginary part
    parameters = {'S_type':S_type, 'epsilon_imag':epsilon_imag, 'verbosity':verbosity, 'M': M}
    if S_type == "wishart":
        parameters['alpha'] = alpha
    elif S_type == "uniform":
        parameters['Lmax'] = Lmax
    elif S_type == "orthogonal":
        parameters['sigma'] = sigma

    y_mses = np.zeros((Deltas.size, NB_INSTANCES))
    for (i_D, Delta) in enumerate(Deltas):
        print("Starting Delta =", Delta, "number", i_D+1, "/", Deltas.size)
        t0 = time.time()
        for k in range(NB_INSTANCES):
            denoiser = Denoising_RIE(Delta_ = Delta, parameters_ = parameters)
            y_mses[i_D][k] = denoiser.run()
        t1 = time.time()
        print("Finished this value of Delta in", round((t1-t0)/60,5), "minutes. Found y_mses:",round(np.mean(y_mses[i_D]), 5), "+-", round(np.std(y_mses[i_D]), 5))
        output = {'Delta':Delta, 'parameters':parameters, 'y_mse':y_mses[i_D], 'NB_INSTANCES':NB_INSTANCES}
        if S_type == "wigner":
            filename = "Data/tmp/denoising_RIE_wigner_M_"+str(M)+"_Delta_"+str(Delta)+".pkl"
        elif S_type == "wishart":
            filename = "Data/tmp/denoising_RIE_wishart_M_"+str(M)+"_alpha_"+str(alpha)+"_Delta_"+str(Delta)+".pkl"
        elif S_type == "uniform":
            #filename = "Data/tmp/denoising_RIE_uniform_M_"+str(M)+"_Lmax_"+str(Lmax)+"_Delta_"+str(Delta)+".pkl"
            filename = "Data/tmp/denoising_RIE_uniform_M_"+str(M)+"_Lmax_sqrt_3_Delta_"+str(Delta)+".pkl"
        elif S_type == "orthogonal":
            filename = "Data/tmp/denoising_RIE_orthogonal_M_"+str(M)+"_sigma_"+str(sigma)+"_Delta_"+str(Delta)+".pkl"
        outfile = open(filename,'wb')
        pickle.dump(output,outfile)

    if S_type == "wigner" and verbosity >= 2:
        if verbosity >= 3:
            print("Expected y_MSEs:", Deltas / (1.+Deltas))
        print("Max difference to the analytic MMSEs:",max(np.abs(np.mean(y_mses, axis = 1) - Deltas / (1.+Deltas)) ) )

    if S_type == "wigner":
        filename = "Data/RIE_wigner_M_"+str(M)+".pkl"
    elif S_type == "wishart":
        filename = "Data/RIE_wishart_M_"+str(M)+"_alpha_"+str(alpha)+".pkl"
    elif S_type == "uniform":
        #filename = "Data/RIE_uniform_M_"+str(M)+"_Lmax_"+str(Lmax)+".pkl"
        filename = "Data/RIE_uniform_M_"+str(M)+"_Lmax_sqrt_3.pkl"
    elif S_type == "orthogonal":
        filename = "Data/RIE_orthogonal_M_"+str(M)+"_sigma_"+str(sigma)+".pkl"
    outfile = open(filename,'wb')
    result = {'Deltas':Deltas, 'parameters':parameters, 'y_mses':y_mses, 'NB_INSTANCES':NB_INSTANCES}
    pickle.dump(result,outfile)