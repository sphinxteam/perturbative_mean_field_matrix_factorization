import numpy as np
from scipy import linalg, optimize
from numpy.polynomial import Polynomial

def g_Y(S_type, rescaled, parameters, Delta, z, shifted_zero_trace = True):
    #Computes the Stieltjes-transform of Y/sqrt[M] = S + sqrt(Delta/M) Z, with Z a Gaussian Wigner matrix
    #If rescaled=True, then we actually consider Y / sqrt[Delta] rather than Y
    assert np.imag(z) > 0, "ERROR: We need Im[z] > 0 to compute g(z). Here Im[z] = "+str(np.imag(z))
    g, error = 0 + 0*1j, np.inf
    tol = 1e-5 #Tolerance on the error

    if S_type in ["wishart", "wigner", "orthogonal"]:
        #In all these cases we use a polynomial solver on g = -s, which has positive imaginary part.
        if S_type == "wishart":
            alpha = parameters["alpha"]
        elif S_type == "orthogonal":
            sigma = parameters["sigma"] #The scale of the orthogonal matrix H/sqrt(M) = sigma O

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
    
    elif S_type == "uniform":
        zr, zi = np.real(z), np.imag(z)
        assert zi >= 0 and zi < 1e-6, "ERROR: For the uniform spectrum, we can only compute the Stieltjes transform just above the real axis"
        #Then we essentially take zi = 0 afterwards.

        #The eigenvalues of S are uniform in [-Lmax,Lmax]
        # I think this might be due to the complex logarithm changing branches, or something of this type.
        Lmax_true = parameters["Lmax"]
        Lmax = Lmax_true
        if rescaled:
            Lmax /= np.sqrt(Delta) #Then we look at Uniform[L/sqrt(Delta)] + Z

        def gm1_Y(g_tab):
            gr, gi = g_tab[0], g_tab[1]
            x, y = Lmax*gr, Lmax*gi
            #Real and complex parts of coth(x+iy) = coth(Lmax * g)
            real_coth = - np.sinh(2*x) / ( np.cos(2*y) - np.cosh(2*x) )
            im_coth = np.sin(2*y) / (np.cos(2*y) - np.cosh(2*x) )
            if not(rescaled):
                res_r = -Delta*gr - Lmax*real_coth - zr
                res_i = -Delta*gi - Lmax*im_coth - zi
            else:
                res_r = -gr - Lmax*real_coth - zr
                res_i = -gi - Lmax*im_coth - zi
            return [res_r, res_i]

        #Now the Jacobian 
        def Jacobian_gm1(g_tab):
            gr, gi = g_tab[0], g_tab[1]
            x, y = Lmax*gr, Lmax*gi
            denominator = (np.cos(2*y) - np.cosh(2*x))**2
            if not(rescaled):
                res_rr = - Delta - Lmax**2*( 2 * (1 - np.cos(2*y)*np.cosh(2*x) ) ) / denominator
            else:
                res_rr = - 1 - Lmax**2*( 2 * (1 - np.cos(2*y)*np.cosh(2*x) ) ) / denominator
            res_ri = - Lmax**2 * 2 * np.sin(2*y)*np.sinh(2*x) / denominator #Derivative of res_r wrt gi
            res_ii = res_rr
            res_ir = - res_ri
            return [[res_rr, res_ir], [res_ri, res_ii]]

        x_im = - np.inf
        starts_im = 10**np.linspace(-5,2, num = 7)
        starts_im = np.concatenate((-starts_im,starts_im))
        starts_re = np.array([1.,-1.])
        i_starts_im, i_starts_re, found = 0, 0, False
        while i_starts_re < len(starts_re) and not(found):
            while i_starts_im < len(starts_im) and not(found):
                root = optimize.root(gm1_Y, x0 = [starts_re[i_starts_re], starts_im[i_starts_im]], jac = Jacobian_gm1)
                x_im = root.x[1]
                error = np.linalg.norm(gm1_Y(root.x))
                i_starts_im += 1
                found = (x_im > -1e-5 and np.abs(error) < tol)
                #print(zr, root.x, np.abs(error))
            i_starts_re += 1
        g = root.x[0] + 1j*root.x[1]
        if np.imag(g) > 0.91: #FIXME
            print(zr, np.imag(g))
        assert found, "ERROR: No suitable solution found for Delta = "+str(Delta)+" and z = "+str(z)+"."

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
