import numpy as np

from OLS_Solver import OLS_Solver


class SparsifyDynamicsSolver(OLS_Solver):
    """
    Sequential-thresholding sparsification solver for WENDy problems.

    Inherits OLS_Solver to access the weak-form system (G0, b0, V_cell, Vp_cell)
    built by fit_OLS(), then applies iterative hard thresholding on each governing
    equation independently to promote sparsity in the recovered coefficient vector. 
    """

    def __init__(self, features, xobs, tobs, type_tf, toggle_SVD, gap,
                 p, S, mu, Mtilde, diag_reg, trunc, radius, type_rad):

        # Inherit all data attributes and OLS machinery from OLS_Solver / WData
        super().__init__(features, xobs, tobs, type_tf, toggle_SVD, gap, p, S, mu, Mtilde,
                         diag_reg, trunc, radius, type_rad)

        # Per-equation sparse coefficient vectors, populated by sparsifyDynamics()
        self.Xi_list = None

        # Per-equation iteration counts at convergence
        self.its_list = None


    def sparsifyDynamics(self, lam_val, gamma=0.0, Mmask=None, max_outer_iter=None):
        """
        Run the sequential-thresholding sparsification across all d state equations.

        Parameters
        ----------
        lam_val : float
            Sparsity threshold. In the unconstrained case, coefficients with
            magnitude below lam_val are zeroed. In the masked case, lam_val
            scales the adaptive lower/upper bounds.
        gamma : float, optional
            Tikhonov regularisation weight. When nonzero, the least-squares
            problem is augmented with gamma * I to penalise large coefficients.
        Mmask : array-like or list of array-like, optional
            Per-coefficient binary mask. If a list, Mmask[ii] applies to
            equation ii. If a single array, it is broadcast to all equations.
            Entries of 0 lock a coefficient to zero; entries of 1 allow it.
        max_outer_iter : int or None, optional
            Cap on thresholding iterations per equation. Defaults to the number
            of candidate library terms (nn) if not specified.

        Returns
        -------
        w_hat : ndarray, shape (total_coeffs, 1)
            Stacked sparse coefficient vector across all equations.
        Xi_list : list of ndarray
            Per-equation coefficient vectors.
        its_list : list of int
            Number of thresholding iterations each equation required.
        """

        # Build weak-form matrices via the inherited OLS fit.
        # This populates self.V_cell, self.Vp_cell, self.Phi, self.PhiP, etc.
        self.fit_OLS()

        # Separate observed states into a list of 1-D arrays, one per dimension
        xobs_cell = [self.xobs[:, ii] for ii in range(self.d)]

        # Evaluate every library function on the observed data to get Theta
        # Theta_cell[ii] has shape (M+1, num_features_for_eq_ii)
        Theta_cell = [np.vstack([ff(*xobs_cell) for ff in f_list]).T
                      for f_list in self.features]

        # Project Theta and the state data through the test functions to get
        # the weak-form linear system:  G0_list[ii] @ Xi_ii  =  b0_list[ii]
        G0_list = [V_mat @ Th for Th, V_mat in zip(Theta_cell, self.V_cell)]
        b0_list = [(-Vp_mat @ xx).reshape(-1, 1)
                   for xx, Vp_mat in zip(xobs_cell, self.Vp_cell)]

        Xi_list = []
        its_list = []

        # Solve each equation independently via sequential thresholding
        for ii in range(self.d):
            Theta_eq = np.asarray(G0_list[ii], dtype=float)
            dXdt_eq = np.asarray(b0_list[ii], dtype=float)
            num_features = Theta_eq.shape[1]

            # Retrieve the mask for this equation (None if unconstrained)
            mask_ii = self._get_state_mask(Mmask, ii, num_features)

            Xi_ii, its_ii = self._sparsify_single_equation(
                Theta=Theta_eq,
                dXdt=dXdt_eq,
                lam_val=float(lam_val),
                gamma=float(gamma),
                mask=mask_ii,
                max_outer_iter=max_outer_iter,
            )

            Xi_list.append(Xi_ii)
            its_list.append(its_ii)

        self.Xi_list = Xi_list
        self.its_list = its_list

        # Stack per-equation vectors into one column to match Simulation's interface
        self.w_hat = np.vstack(Xi_list).reshape(-1, 1)
        return self.w_hat, self.Xi_list, self.its_list


    ###############################################################################################################################
    ###############################################################################################################################


    def _get_state_mask(self, Mmask, state_idx, num_features):
        """
        Extract and validate the coefficient mask for a single equation.

        Parameters
        ----------
        Mmask : None, list, or array-like
            Global mask specification. None means no masking.
        state_idx : int
            Which equation (state dimension) this mask is for.
        num_features : int
            Expected number of library terms for this equation.

        Returns
        -------
        mask : ndarray of shape (num_features, 1) or None
        """
        if Mmask is None:
            return None

        # If Mmask is a list, each entry corresponds to one equation;
        # otherwise the same mask is shared across all equations.
        if isinstance(Mmask, list):
            mask_flat = np.asarray(Mmask[state_idx], dtype=float).reshape(-1)
        else:
            mask_flat = np.asarray(Mmask, dtype=float).reshape(-1)

        if mask_flat.size != num_features:
            raise ValueError(
                f"Mmask for state {state_idx} must have length {num_features}, "
                f"got {mask_flat.size}"
            )

        return mask_flat.reshape(-1, 1)


    def _sparsify_single_equation(self, Theta, dXdt, lam_val, gamma, mask, max_outer_iter):
        """
        Sequential thresholded least-squares for ONE governing equation.

        This is the core loop: solve the (possibly augmented) least-squares
        problem, identify coefficients below the sparsity threshold, lock them
        to zero, re-solve on the remaining "big" terms, and repeat until the
        active set stabilises or the iteration cap is reached.

        Parameters
        ----------
        Theta : ndarray, shape (num_rows, nn)
            Weak-form library matrix for this equation.
        dXdt : ndarray, shape (num_rows, 1)
            Weak-form right-hand side for this equation.
        lam_val : float
            Sparsity-controlling threshold parameter.
        gamma : float
            Tikhonov regularisation weight (0 disables).
        mask : ndarray of shape (nn, 1) or None
            Per-coefficient mask. None means unconstrained thresholding.
        max_outer_iter : int or None
            Maximum number of thresholding sweeps. Defaults to nn.

        Returns
        -------
        Xi : ndarray, shape (nn, 1)
            Sparse coefficient vector for this equation.
        num_iters : int
            Number of iterations performed before termination.
        """

        num_rows, nn = Theta.shape

        # Validate the shape of the right-hand side
        if dXdt.shape != (num_rows, 1):
            raise ValueError(f"dXdt must have shape {(num_rows, 1)}, got {dXdt.shape}")

        # ---- Tikhonov augmentation ----
        # When gamma > 0 we append gamma*I below Theta and zeros below dXdt,
        # which is equivalent to adding a ridge penalty ||gamma * Xi||^2
        if gamma != 0.0:
            Theta_aug = np.vstack([Theta, gamma * np.eye(nn)])
            dXdt_aug = np.vstack([dXdt, np.zeros((nn, 1))])
        else:
            Theta_aug = Theta
            dXdt_aug = dXdt

        # ---- Initial least-squares solve (no thresholding yet) ----
        Xi = np.linalg.lstsq(Theta_aug, dXdt_aug, rcond=None)[0]  # shape (nn, 1)

        # ---- Build adaptive threshold bounds when a mask is supplied ----
        # The mask allows per-coefficient constraints: masked-out coefficients
        # are permanently zeroed, while active ones get adaptive upper/lower
        # bounds that depend on the relative scale of each library column.
        if mask is not None:
            mask = np.asarray(mask, dtype=float).reshape(nn, 1)
            Xi = mask * Xi  # zero out masked coefficients in the initial guess

            # Compute column-wise scaling for adaptive bounds
            rhs_norm = np.linalg.norm(dXdt_aug, ord=2)
            col_norms = np.linalg.norm(Theta_aug, axis=0, ord=2).reshape(-1, 1)
            col_norms = np.where(col_norms == 0.0, 1e-12, col_norms)  # avoid division by zero

            # bnds measures each coefficient's expected scale relative to the rhs
            bnds = (rhs_norm / col_norms) * mask
            lower_bounds = lam_val * np.maximum(1.0, bnds)
            upper_bounds = (1.0 / lam_val) * np.minimum(1.0, bnds)

        # Boolean array tracking which coefficients are currently zeroed out.
        # Initialised to all-False (nothing thresholded yet).
        smallinds = np.zeros_like(Xi, dtype=bool)

        # Default iteration cap is nn (one sweep per candidate term at most)
        it_cap = nn if max_outer_iter is None else int(max_outer_iter)

        # ---- Sequential thresholding loop ----
        for kk in range(1, it_cap + 1):

            # Determine which coefficients fall below the sparsity threshold.
            # Masked case: a coefficient is "small" if it is below the adaptive
            #   lower bound OR above the adaptive upper bound (out-of-range).
            # Unmasked case: a coefficient is "small" if |Xi| < lam_val.
            if mask is not None:
                smallinds_new = (np.abs(Xi) < lower_bounds) | (np.abs(Xi) > upper_bounds)
            else:
                smallinds_new = (np.abs(Xi) < lam_val)

            # Convergence check: if the active set hasn't changed, we're done
            if np.array_equal(smallinds_new, smallinds):
                return Xi, kk

            # Commit the new sparsity pattern (defensive copy to avoid aliasing)
            smallinds = smallinds_new.copy()

            # Zero out all coefficients that are flagged as small
            Xi[smallinds] = 0.0

            # Re-solve least-squares on only the surviving ("big") terms
            biginds = ~smallinds.reshape(-1)
            if np.any(biginds):
                Xi_big = np.linalg.lstsq(Theta_aug[:, biginds], dXdt_aug, rcond=None)[0]

                # In the masked case, re-apply the mask after solving;
                # in the unmasked case, just assign the new values directly.
                if mask is not None:
                    Xi[biginds, :] = mask[biginds, :] * Xi_big
                else:
                    Xi[biginds, :] = Xi_big

        return Xi, it_cap