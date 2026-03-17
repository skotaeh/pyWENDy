import numpy as np
from scipy.integrate import solve_ivp

class Simulation:
    
    def __init__(self, features, w_hat, x0, t):
        self.features = features
        self.w_hat = w_hat
        self.x0 = x0
        self.t = t
        

    def simulate(self):
        tol_ode = 1e-15
        w_hat_tolist = []
        count = 0
        for i in range(len(self.features)): 
            a = self.features[i]
            coef = []
            for j in range(len(a)):
                coef.append(self.w_hat[count+j][0])
            count = count + len(a)
            w_hat_tolist.append(coef)        

        rhs_p = lambda tt, x: self._rhs_fun(self.features, w_hat_tolist, x)
        sol = solve_ivp(rhs_p, t_span = np.array([self.t[0], self.t[-1]]), y0=self.x0, t_eval=self.t,  
                        method='BDF', rtol=tol_ode, atol=tol_ode, events=self._blowup_event(1e3))
        if sol.y.shape[1] < len(self.t):
            print('oops')
            final_val = sol.y[:, -1][:, None]
            pad_count = len(self.t) - sol.y.shape[1]
            pad_vals = np.tile(final_val, (1, pad_count))
            y_full = np.hstack([sol.y, pad_vals])
        else:
            y_full = sol.y
        return y_full.T
    

    def _rhs_fun(self,features, params, x):
            nstates = len(x)
            x = tuple(x)
            dx = np.zeros(nstates)
            for i in range(nstates):
                dx[i] = np.sum([f(*x)*p for f, p in zip(features[i], params[i])])
            return dx
    
        
    def _blowup_event(self,thresh):
        def event(t, y):
            return np.linalg.norm(y, ord=np.inf) - thresh
        event.terminal = True
        event.direction = 0
        return event
        
    
