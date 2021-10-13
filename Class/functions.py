import numpy as np
from scipy import linalg, optimize
from numpy.polynomial import Polynomial

def g_Y(S_type, rescaled, parameters, Delta, z, shifted_zero_trace = True):
    #Computes the Stieltjes-transform of Y/sqrt[M] = S + sqrt(Delta/M) Z, with Z a Gaussian Wigner matrix
    #If rescaled=True, then we actually consider Y / sqrt[Delta] rather than Y
    assert np.imag(z) > 0, "ERROR: We need Im[z] > 0 to compute g(z). Here Im[z] = "+str(np.imag(z))
    g, error = 0. + 0.*1j, np.inf
    tol = 1e-5 #Tolerance on the error

    #We use a polynomial solver on g = -s, which has positive imaginary part.
    if S_type == "wishart":
        alpha = parameters["alpha"]
    elif S_type == "orthogonal":
        sigma = parameters["sigma"] #The scale of the orthogonal matrix H/sqrt(M) = sigma * O

    #Wishart
    if S_type == "wishart" and not(rescaled):
        if shifted_zero_trace:
            P = Polynomial([1, z + np.sqrt(alpha), 1 + z*np.sqrt(alpha) + Delta, np.sqrt(alpha)*Delta])
        else:
            P = Polynomial([1, z + np.sqrt(alpha) - 1/np.sqrt(alpha), z*np.sqrt(alpha) + Delta, np.sqrt(alpha)*Delta])
    elif S_type == "wishart" and rescaled:
        if shifted_zero_trace:
            P = Polynomial([1, z + np.sqrt(alpha/Delta), (1+Delta+z*np.sqrt(alpha*Delta))/Delta , np.sqrt(alpha/Delta)])
        else:
            P = Polynomial([1, z + np.sqrt(alpha/Delta) - 1./np.sqrt(alpha*Delta), 1+z*np.sqrt(alpha/Delta) , np.sqrt(alpha/Delta)])
    
    #Wigner
    if S_type == "wigner" and not(rescaled):
        P = Polynomial([1, z, 1 + Delta])
    elif S_type == "wigner" and rescaled:
        P = Polynomial([1, z, 1 + 1/Delta])
    
    #Orthogonal
    if S_type == "orthogonal" and not(rescaled):
        P = Polynomial([z, z**2 + Delta - sigma**2 , 2*z*Delta, Delta**2])
    elif S_type == "orthogonal" and rescaled:
        P = Polynomial([z, 1 + z**2 - sigma**2/Delta , 2*z, 1])
    
    solutions = P.roots()
    #For the orthogonal case, we remove roots of the polynomial that are not solutions of the original equation
    if S_type == "orthogonal":
        def original_eq(g):
            if not(rescaled):
                return min(np.abs(- (1.+np.sqrt(1.+4*g**2*sigma**2))/(2.*g) -Delta*g -z), np.abs(- (1.-np.sqrt(1.+4*g**2*sigma**2))/(2.*g) - Delta*g -z))
            if rescaled:
                return min(np.abs(- (1.+np.sqrt(1.+4*g**2*sigma**2/Delta))/(2.*g) - g -z), np.abs(- (1.-np.sqrt(1.+4*g**2*sigma**2/Delta))/(2.*g) - g -z) )
        list_indices = []
        for (i, g) in enumerate(solutions):
            if np.abs(original_eq(g)) > 1e-6:
                list_indices.append(i)
        solutions = np.delete(solutions, list_indices)
    g = solutions[np.argmax(np.imag(solutions))] #Find the solution with largest imaginary part

    #Wishart
    if S_type == "wishart" and not(rescaled):
        error = 1./(np.sqrt(alpha)*(1+np.sqrt(alpha)*g)) - Delta*g  - (1./g) - z
        if shifted_zero_trace:
            error -= 1./np.sqrt(alpha)
    elif S_type == "wishart" and rescaled:
        error = 1./(alpha*g+np.sqrt(alpha*Delta)) - g  - (1./g) - z
        if shifted_zero_trace:
            error -= 1./np.sqrt(alpha*Delta)

    #Wigner
    if S_type == "wigner" and not(rescaled):
        error = -g - Delta*g  - (1./g) - z
    elif S_type == "wigner" and rescaled:
        error = - g/Delta - g - (1./g)  - z
    
    #Orthogonal
    if S_type == "orthogonal" and not(rescaled):
        error = min(np.abs(- (1.+np.sqrt(1.+4*g**2*sigma**2))/(2.*g) -Delta*g -z), np.abs(- (1.-np.sqrt(1.+4*g**2*sigma**2))/(2.*g) -Delta*g -z))
    elif S_type == "orthogonal" and rescaled:
        error = min(np.abs(- (1.+np.sqrt(1.+4*g**2*sigma**2/Delta))/(2.*g) - g -z), np.abs(- (1.-np.sqrt(1.+4*g**2*sigma**2/Delta))/(2.*g) - g -z) )

    assert np.abs(error) < tol, "ERROR: Stieltjes transform does not satisfy g^(-1)[g(z)] = z. The error is "+str(round(error,10))+" for Delta = "+str(Delta)+" and z = "+str(z)+" and g(z) = "+str(g)
    assert np.imag(g)/np.pi > -1e-5, "ERROR: Negative imaginary part of g: "+str(round(g,10))
    return g

def random_orthogonal(M, rng):
    #Returns a MxM uniformly distributed orthogonal matrix

    #We first generate a m x m orthogonal matrix by taking the QR decomposition of a random m x m gaussian matrix
    gaussian_matrix = rng.normal(0, 1, (M, M)) 
    U, R = linalg.qr(gaussian_matrix)

    #Then we multiply on the right by the signs of the diagonal of R to really get the Haar measure
    D = np.diagonal(R)
    Lambda = D / np.abs(D)
    return np.multiply(U, Lambda)
