import numpy as np
from scipy.integrate import solve_ivp
import time

import IRLS_Solver as IRLS
import Simulation as Sim


def rhs_fun(features, params, x):
    nstates = len(x)
    x = tuple(x)
    dx = np.zeros(nstates)
    for i in range(nstates):
        dx[i] = np.sum([f(*x)*p for f, p in zip(features[i], params[i])])
    return dx


def lorenz():
    features = [
        [lambda x, y, z: y, lambda x, y, z: x],
        [lambda x, y, z: x, lambda x, y, z: x*z, lambda x, y, z: y],
        [lambda x, y, z: x*y, lambda x, y, z: z]
    ]
    params = [np.array([10, -10]), np.array([28, -1, -1]), np.array([1, -8/3])]

    x0 = np.array([-8, 10, 27])
    t =  np.linspace(0, 10, 501)
    tspan = (t[0], t[-1])
    tol_ode = 1e-15
    rhs_p = lambda t, x: rhs_fun(features, params, x)
    true_vec = np.concatenate(params).reshape(-1, 1)
    options_ode_sim = {"rtol": tol_ode, "atol": tol_ode*np.ones(len(x0))}

    t0 = time.time()
    sol = solve_ivp(rhs_p, t_span = tspan, y0=x0, t_eval=t, rtol=tol_ode, atol=tol_ode)
    #print("sim time =", time.time() - t0)
    x = sol.y.T
    t = sol.t
    #plt.plot(t, x)
    return x, t, params, x0, true_vec, features, rhs_p


def logistic_growth():
    features = [
        [lambda x: x, lambda x: x**2]
    ]
    params = [np.array([ 1, -1])]
    x0 = np.array([ 0.01 ])
    t =  np.linspace(0, 10, 501)
    tspan = (t[0], t[-1])
    tol_ode = 1e-15

    rhs_p = lambda t, x: rhs_fun(features, params, x)
    true_vec = np.concatenate(params).reshape(-1, 1)
    options_ode_sim = {"rtol": tol_ode, "atol": tol_ode*np.ones(len(x0))}

    t0 = time.time()
    sol = solve_ivp(rhs_p, t_span = tspan, y0=x0, t_eval=t, rtol=tol_ode, atol=tol_ode)
    #print("sim time =", time.time() - t0)
    x = sol.y.T
    t = sol.t
    #plt.plot(t, x)
    return x, t, params, x0, true_vec, features, rhs_p


def gen_noise(U_exact, sigma_NR, noise_dist, noise_alg):
    if noise_alg == 0:  # additive
        stdv = np.square(np.sqrt(np.mean(np.square(U_exact))))
    elif noise_alg == 1:  # multiplicative
        stdv = 1
    dims = U_exact.shape
    if noise_dist == 0:  # white noise
        if sigma_NR > 0:
            sigma = sigma_NR * np.sqrt(stdv)
        else:
            sigma = -sigma_NR
        noise = np.random.normal(0, sigma, dims)
    elif noise_dist == 1:  # uniform noise
        if sigma_NR > 0:
            sigma = np.sqrt(3 * np.square(sigma_NR) * stdv)
        else:
            sigma = -sigma_NR
        noise = sigma * (2 * np.random.rand(*dims) - 1)
    if noise_alg == 0:  # additive
        U = U_exact + noise
    elif noise_alg == 1:  # multiplicative
        U = U_exact * (1 + noise)
    noise_ratio_obs = np.linalg.norm(U - U_exact) / np.linalg.norm(U_exact)
    return U, noise, noise_ratio_obs, sigma

x, t, params, x0, true_vec, features,rhs_p = logistic_growth()
subsamp =  1 # subsample data in time
tobs = t[::subsamp]
xobs = x[::subsamp, :]
M, nstates = xobs.shape

#add noise
noise_ratio = 0.1
noise_dist = 0
noise_alg = 0
xobs_n, noise, _, sigma = gen_noise(xobs, noise_ratio, noise_dist, noise_alg)
IRLS_SL_model = IRLS.IRLS_Solver(features, xobs_n, tobs, type_tf = 0, toggle_SVD=False, gap = 1, 
                                 p = 16, S = 1, mu = [1, 2, 1], diag_reg=1e-10, Mtilde = None, 
                                 trunc=0, radius = None, type_rad=0)
IRLS_SL_model.fit_IRLS()



