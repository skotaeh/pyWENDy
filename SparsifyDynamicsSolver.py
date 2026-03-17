import numpy as np

from OLS_Solver import OLS_Solver

class SparsifyDynamicsSolver(OLS_Solver):
    def __init__(self, features, xobs: np.ndarray, tobs: np.ndarray, type_tf, toggle_SVD, gap,
                p, S, mu, Mtilde, diag_reg, trunc, radius, type_rad,):

        super().__init__(features, xobs, tobs, type_tf, toggle_SVD, gap, p, S, mu, Mtilde,
                diag_reg, trunc, radius, type_rad)

        self.Xi_list = None
        self.its_list = None

    def sparsifyDynamics(self, lambda_, gamma=0.0, Mmask=None, max_outer_iter=None):
        # Build weak-form matrices the same way IRLS does: fit_OLS first
        # (this sets V_cell, Vp_cell, etc.)
        self.fit_OLS()

        xobs_cell = [self.xobs[:, i] for i in range(self.d)]
        Theta_cell = [np.vstack([f(*xobs_cell) for f in f_list]).T for f_list in self.features]

        G0_list = [V @ Th for Th, V in zip(Theta_cell, self.V_cell)]
        b0_list = [(-Vp @ x).reshape(-1, 1) for x, Vp in zip(xobs_cell, self.Vp_cell)]

        Xi_list = []
        its_list = []

        for i in range(self.d):
            Theta = np.asarray(G0_list[i], dtype=float)     # (K_i, nn_i)
            dXdt = np.asarray(b0_list[i], dtype=float)      # (K_i, 1)
            nn_i = Theta.shape[1]

            Mi = self._get_state_mask(Mmask, i, nn_i)       # None or (nn_i, 1)

            Xi_i, its_i = self._sparsify_single_equation(
                Theta=Theta,
                dXdt=dXdt,
                lambda_=float(lambda_),
                gamma=float(gamma),
                M=Mi,
                max_outer_iter=max_outer_iter,
            )

            Xi_list.append(Xi_i)
            its_list.append(its_i)

        self.Xi_list = Xi_list
        self.its_list = its_list

        # Stack to match Simulation’s expected ordering
        self.w_hat = np.vstack(Xi_list).reshape(-1, 1)
        return self.w_hat, self.Xi_list, self.its_list

    ###############################################################################################################################
    ###############################################################################################################################

    def _get_state_mask(self, Mmask, state_idx, nn_i):
        if Mmask is None:
            return None

        if isinstance(Mmask, list):
            Mi = np.asarray(Mmask[state_idx], dtype=float).reshape(-1)
        else:
            Mi = np.asarray(Mmask, dtype=float).reshape(-1)

        if Mi.size != nn_i:
            raise ValueError(f"Mmask for state {state_idx} must have length {nn_i}, got {Mi.size}")

        return Mi.reshape(-1, 1)

    def _sparsify_single_equation(self, Theta, dXdt, lambda_, gamma, M, max_outer_iter):
        """
        Direct Python implementation of the MATLAB sparsifyDynamics loop for ONE equation.
        """

        N, nn = Theta.shape
        if dXdt.shape != (N, 1):
            raise ValueError(f"dXdt must have shape {(N, 1)}, got {dXdt.shape}")

        if gamma != 0.0:
            Theta_aug = np.vstack([Theta, gamma * np.eye(nn)])
            dXdt_aug = np.vstack([dXdt, np.zeros((nn, 1))])
        else:
            Theta_aug = Theta
            dXdt_aug = dXdt

        # MATLAB: Xi = Theta \ dXdt
        Xi = np.linalg.lstsq(Theta_aug, dXdt_aug, rcond=None)[0]  # (nn, 1)

        if M is not None:
            M = np.asarray(M, dtype=float).reshape(nn, 1)
            Xi = M * Xi

            d_norm = np.linalg.norm(dXdt_aug, ord=2)
            col_norms = np.linalg.norm(Theta_aug, axis=0, ord=2).reshape(-1, 1)
            col_norms = np.where(col_norms == 0.0, 1e-12, col_norms)

            bnds = (d_norm / col_norms) * M
            LBs = lambda_ * np.maximum(1.0, bnds)
            UBs = (1.0 / lambda_) * np.minimum(1.0, bnds)

        smallinds = np.zeros_like(Xi, dtype=bool)

        it_cap = nn if max_outer_iter is None else int(max_outer_iter)

        for j in range(1, it_cap + 1):
            if M is not None:
                smallinds_new = (np.abs(Xi) < LBs) | (np.abs(Xi) > UBs)

                if np.array_equal(smallinds_new, smallinds):
                    return Xi, j

                smallinds = smallinds_new
                Xi[smallinds] = 0.0

                big = ~smallinds.reshape(-1)
                if np.any(big):
                    Xi_big = np.linalg.lstsq(Theta_aug[:, big], dXdt_aug, rcond=None)[0]
                    Xi[big, :] = M[big, :] * Xi_big

            else:
                smallinds_new = (np.abs(Xi) < lambda_)

                if np.array_equal(smallinds_new, smallinds):
                    return Xi, j

                smallinds = smallinds_new
                Xi[smallinds] = 0.0

                big = ~smallinds.reshape(-1)
                if np.any(big):
                    Xi_big = np.linalg.lstsq(Theta_aug[:, big], dXdt_aug, rcond=None)[0]
                    Xi[big, :] = Xi_big

        return Xi, it_cap