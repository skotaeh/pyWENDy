from OLS_Solver import OLS_Solver
import numpy as np
from sympy import symbols, diff, lambdify, sympify
from scipy.signal import convolve2d
from scipy.sparse import spdiags, eye
from scipy.linalg import block_diag
from scipy.stats import shapiro
from copy import deepcopy


class IRLS_Solver(OLS_Solver):

    def __init__(self, features, xobs, tobs, type_tf, toggle_SVD, gap, p, S, mu, Mtilde, 
                 diag_reg, trunc, radius, type_rad):
        
        super().__init__(features, xobs, tobs, type_tf, toggle_SVD, gap, p, S, mu, Mtilde, 
                 diag_reg, trunc, radius, type_rad)
        
        self.Jac_mat = None
        self.flag = True

        
    def fit_IRLS(self):        
        if self.flag:
            # we have to always first call an OLS fit in order to proceed
            self.fit_OLS()
            
            # estimate noise variance and build initial covariance
            self._build_Jac_sym()
            param_length_vec = np.array([len(x) for x in self.features])
            L0, L1 = self._get_Lfac(param_length_vec)

            sig_ests = np.array([self._estimate_sigma(self.xobs[:, i]) for i in range(self.d)])
            RT_0 = spdiags(np.kron(sig_ests, np.ones(self.Mp1)), 0, self.Mp1*self.d, self.Mp1*self.d)
            L0 = L0 @ RT_0
            
            s1, s2, s3 = L1.shape
            L1_temp = L1.copy()
            for i in range(s3):
                L1_temp[:, :, i] = L1[:, :, i] @ RT_0 
                L1 = L1_temp        
                
            pvals_list = []
            res = self.G0 @ self.w_hat - self.b0
            _, pvals = shapiro(res)
            pvals_list.append(pvals)

            #w_hat = w0.reshape(-1,1)
            w_hat_its =[self.w_hat]
            iter = 1; check = 1; pval = 1
            w_hat_loc = deepcopy(self.w_hat)
            RT = eye(len(self.b0), format='csc')

            while check > self.iter_diff_tol and iter < self.max_iter and pval > self.pvalmin:
                try:
                    RT, _, _ = self._get_RT(L0, L1, w_hat_loc)
                except np.linalg.LinAlgError:
                    print("Cholesky decomposition failed: matrix not positive definite.")
                    print("Returning initial guess w0.")
                    self.w_hat = self.w_hat_loc
                    
                G = np.linalg.solve(RT, self.G0)
                b = np.linalg.solve(RT, self.b0)

                w_hat_loc = np.linalg.lstsq(G, b, rcond=None)[0].reshape(-1, 1)
                res_n = G @ w_hat_loc  - b
                    
                # check stopping conditions
                _, pvals = shapiro(res_n)
                pvals_list.append(pvals)
                if iter+1 > self.check_pval_it:
                    pval = pvals_list[iter]

                check = np.linalg.norm(w_hat_its[-1] - w_hat_loc)/np.linalg.norm(w_hat_its[-1])
                iter += 1
                w_hat_its.append(w_hat_loc)

            if pval < self.pvalmin:
                print('error: WENDy iterates diverged')
                ind = np.argmax(pvals_list)
                w_hat_loc = w_hat_its[ind]
                w_hat_its.append(w_hat_loc)

            self.w_hat = w_hat_loc
            self.pvals_list = pvals_list
            self.w_hat_its = w_hat_its
            self.flag = False
        else:
            print("Model already built")
        

    def _diff_lambda(self, f, args, var):
        return sympify(diff(f(*args), var))


    def _build_Jac_sym(self):
        M, nstates = self.xobs.shape
        features = [f for f_list in self.features for f in f_list]
        J = len(features)
        self.Jac_mat = np.zeros((J, nstates, M))

        # Create the symbolic variables
        args = symbols('x0:%d' % nstates)
                
        for j in range(J):
            f = features[j]
            for state in range(nstates): 
                g = self._diff_lambda(f, args, args[state])
                G = lambdify(args, g, 'numpy')
                for i in range(M):
                    x_val = self.xobs[i, :]
                    z = G(*x_val)
                    self.Jac_mat[j, state , i] =  z
        

    def _get_Lfac(self, Js):
        _, d, M = self.Jac_mat.shape
        Jac_mat = np.transpose(self.Jac_mat, (1, 2, 0))
        eq_inds = np.where(Js)[0]
        num_eq = len(eq_inds)
        L0 = block_diag(*self.Vp_cell)
        L1 = np.zeros((L0.shape[0], d*M, sum(Js)))
        Ktot = 0
        Jtot = 0
        for i in range(num_eq):
            K, _ = self.V_cell[i].shape
            J = Js[eq_inds[i]]
            for ell in range(d):
                m = np.expand_dims(Jac_mat[ell, :, Jtot+(np.arange(J))].T, axis = 0)
                n = self.V_cell[i][:, :, np.newaxis]
                ixgrid = np.ix_(range(Ktot, Ktot + K), range(ell*M, (ell+1)*M), range(Jtot, Jtot + J))
                L1[ixgrid] = m*n
            Ktot = Ktot + K
            Jtot = Jtot + J
        return L0, L1
    

    def _estimate_sigma(self, f):
        k = 6
        C = self._fdcoeffF(k, 0, np.arange(-k-2, k+3))
        filter = C[:, -1]
        filter = filter / np.linalg.norm(filter, ord=2)
        filter = filter.reshape(1, -1).T
        f = f.reshape(-1, 1)
        sig = np.sqrt(np.mean(np.square(convolve2d(f, filter, mode='valid'))))
        return sig 


    def _fdcoeffF(self, k, xbar, x):
            n = len(x)
            if k >= n:
                raise ValueError('*** length(x) must be larger than k')

            m = k-1  # change to m=n-1 if you want to compute coefficients for all
                    # possible derivatives.  Then modify to output all of C.
            c1 = np.uint32(1)
            c4 = x[0] - xbar
            C = np.zeros((n, m+2), dtype=np.float64)
            C[0, 0] = 1

            for ii in range(n-1):
                i1 = ii+1
                mn = min(ii,m)
                c2 = np.uint32(1)
                c5 = c4
                c4 = x[i1] - xbar      
                for jj in range(-1, ii):
                    j1 = jj+1
                    c3 = x[i1] - x[j1]
                    c2 *= c3
                    if jj == ii-1:
                        for ss in range(mn, -1, -1):
                            C[i1, ss+1] = (c1/c2)*((ss+1)*C[i1-1, ss] - c5*C[i1-1, ss+1])
                        C[i1, 0] = -(c1/c2)*c5*C[i1-1, 0]
                    for ss in range(mn, -1, -1):
                        C[j1, ss+1] = (c4*C[j1, ss+1] - (ss+1)*C[j1, ss])/c3
                    C[j1, 0] = c4*C[j1, 0]/c3
                c1 = c2
            return C

    
    def _get_RT(self, L0, L1, w): 
        dims = L1.shape
        if not np.all(np.all(w == 0)):
            L0 = L0 + np.reshape(np.transpose(L1, (2, 0, 1)).reshape(dims[2], -1).T @ w, (dims[0], -1))
        Cov = L0 @ L0.T
        newCov = (1-self.diag_reg)*Cov + self.diag_reg*np.eye(Cov.shape[0])
        RT = np.linalg.cholesky(newCov)
        return RT, L0, Cov


    