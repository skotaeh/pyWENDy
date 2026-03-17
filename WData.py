import numpy as np
import math

class WData:
    def __init__(self, features, xobs, tobs, type_tf = 0, toggle_SVD = False, gap = 1, p = 10, S = 1, 
                 mu = [1, 2, 1], Mtilde = None, diag_reg = 1e-10, trunc = 0):
        """
        Inputs:
        features: Rhs features
        xobs: data
        tobs: time
        radius: grid point radius in a list (if None - compute this automatically)
        type_rad: 0 - Single-scale Local, 1 - Multi-scale Global
        type_tf: Type of test function: 0 -> L2, 1 ->L_inf
        toggle_SVD: False -> no SVD,  True -> SVD
        gap: gap between test functions
        p: order of poly tf
        S: truncation order of In
        mu: finite difference orders of accuracy
        trunc: truncation method for svd 0 -> corner point, 0 < trunc < 1 trunc% weight of singularvals
        diag_reg:         
        """
        
        self.features = features
        self.xobs = xobs
        self.tobs = tobs
        self.type_tf = type_tf
        self.toggle_SVD = toggle_SVD
        self.gap = gap
        self.p = p

        self.d = self.xobs.shape[1]
        self.dt = self.tobs[1]
        self.T = self.tobs[-1]
        self.M = len(self.tobs) - 1
        self.Mp1 = len(self.tobs)

        self.S = S
        self.mu = mu 
        if Mtilde == None: 
            Mtilde = self.M
        self.Mtilde = Mtilde
        self.trunc = trunc

        #set params
        self.iter_diff_tol = 1e-6
        self.max_iter = 100
        self.diag_reg = diag_reg
        self.pvalmin = 1e-4
        self.check_pval_it = 10
        self.tau = 1e-5

    # at some later date, we can add methods for manipulating data here.  

    def _comb_fun(self, nn):
        tot = 0.
        for kk in range(nn*self.p + 1):
            tot += math.comb(nn*self.p, kk)*(-1)**kk/(2*kk + 1)
        return tot

    