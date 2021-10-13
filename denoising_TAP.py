import numpy as np
import time, pickle, sys
from Class.Denoising_TAP import Denoising_TAP

"""
This finds the solution to the "denoising TAP" equations, in which we did not introduce any constraint on the field X to begin with.
"""

if __name__== "__main__":

    global_seed = int(time.time())
    np.random.seed(global_seed)

    S_type = sys.argv[1]
    M = int(sys.argv[2])
    order_TAP = int(sys.argv[3])
    verbosity = int(sys.argv[4])
    NB_INSTANCES = int(sys.argv[5])
    assert S_type in ["wigner", "wishart"], "ERROR: Unknown matrix type for S"
    if S_type == "wishart":
        alpha = float(sys.argv[6])

    Deltas = 10**np.linspace(2, -3, num=100, endpoint=True)

    if S_type == "wigner":
        print("TAP denoising solution at order", order_TAP)
    elif S_type == "wishart":
        print("TAP denoising solution at order", order_TAP, " with alpha =",alpha)

    phis, y_mses, nishimori_checks = np.zeros((Deltas.size,NB_INSTANCES)), np.zeros((Deltas.size,NB_INSTANCES)), np.zeros((Deltas.size,NB_INSTANCES))
    for (i_D,Delta) in enumerate(Deltas):
        print("Starting Delta =", Delta, ". Number", i_D+1, "/", Deltas.size)

        t0 = time.time()
        for k in range(NB_INSTANCES):
            parameters = {'M':M, 'S_type':S_type, 'Delta':Delta, 'verbosity': verbosity, 'order_TAP':order_TAP}
            if S_type == "wishart":
                parameters['alpha'] = alpha
            Denoising_TAP_solver = Denoising_TAP(parameters)
            solution = Denoising_TAP_solver.get_solution()
            y_mses[i_D][k] = Denoising_TAP_solver.get_y_mse()
            phis[i_D][k] = Denoising_TAP_solver.get_free_entropy()
            nishimori_checks[i_D][k] = Denoising_TAP_solver.check_nishimori()
        t1 = time.time()
        print("Finished this value of Delta in", round((t1-t0)/60,5), "minutes. Found y_mses:",round(np.mean(y_mses[i_D]), 5), "+-", round(np.std(y_mses[i_D]), 5))
        if verbosity >= 2:
            print("Finishing Delta =", Delta, ", mses over y =", y_mses[i_D], ", phi =", phis[i_D], "nishimori checks =", nishimori_checks[i_D])

        output = {'seed':global_seed,'S_type':S_type, 'M':M, 'order_TAP':order_TAP, 'NB_INSTANCES':NB_INSTANCES, 'Delta':Delta, 'phi':phis[i_D], 'mse_y': y_mses[i_D], 'nishimori_check':nishimori_checks[i_D]}
        if S_type == "wigner":
            filename = "Data/tmp/denoising_TAP_wigner_order_"+str(order_TAP)+"_M_"+str(M)+"_Delta_+"+str(Delta)+".pkl"
        elif S_type == "wishart":
            output['alpha'] = alpha
            filename = "Data/tmp/denoising_TAP_wishart_order_"+str(order_TAP)+"_M_"+str(M)+"_alpha_"+str(alpha)+"_Delta_+"+str(Delta)+".pkl"
        outfile = open(filename,'wb')
        pickle.dump(output,outfile)

    output = {'seed':global_seed,'S_type':S_type, 'M':M, 'order_TAP':order_TAP, 'NB_INSTANCES':NB_INSTANCES, 'Deltas':Deltas, 'phis':phis, 'mses_y': y_mses, 'nishimori_checks':nishimori_checks}
    if S_type == "wigner":
        filename = "Data/TAP_wigner_order_"+str(order_TAP)+"_M_"+str(M)+".pkl"
    elif S_type == "wishart":
        output['alpha'] = alpha
        filename = "Data/TAP_wishart_order_"+str(order_TAP)+"_M_"+str(M)+"_alpha_"+str(alpha)+".pkl"
    outfile = open(filename,'wb')
    pickle.dump(output,outfile)