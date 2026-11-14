from copy import copy

import numpy as np
from t3d.Logbook import info, errr, debug


class TransportSolver():

    def __init__(self, grid, species, geo, physics):

        self.grid = grid
        self.geo = geo
        self.physics = physics

        # number of evolved profiles
        self.N_profiles = species.N_profiles
        self.N_radial = grid.N_radial
        self.edge = slice(self.N_radial-1, None, self.N_radial)

        # initialize an N_prof x N_prof identity matrix
        self.I_mat = np.identity(self.N_profiles*(grid.N_radial))

        y_init = species.get_vec_from_profs()

        self.y_m = y_init
        self.y_old = y_init
        self.y_curr = y_init

        self.y_hist = []
        self.y_hist.append(y_init)
        self.y_error = np.zeros_like(y_init)
        self.chi_error = 0
        self.reset = False
        self.external_source_jacobian = None
        self.balance_closure = False

    def psimat(self, species):

        """Differentiate the actual radial residual with a frozen backend response.

        Flux backends supply local slopes in independent density gradient at
        fixed pressure gradient, and in temperature gradient at fixed density.
        Apply those slopes to changes from the actual compact difference and
        interpolation; use the exact shared-face normalization. No fresh
        transport/source backend job is launched for a Jacobian column.
        """
        y = species.get_vec_from_profs()
        if self.balance_closure:
            return np.zeros((y.size, y.size))
        scratch = species.profile_scratch()
        keys = [(key, False) for key in species.n_evolve_keys]
        keys += [(key, True) for key in species.T_evolve_keys]
        rows = {item.type: i for i, item in enumerate(species.get_species_list())}
        kn0, kT0, _, _ = species.get_grads_on_flux_grid()

        G = np.tile((self.geo.G_fac/self.grid.drho).profile, species.evolve_count)
        local_physics = copy(self.physics)
        local_physics.scale_density_to = None
        attributes = ("pflux", "qflux", "heat")

        def residual(v, key, pressure):
            scratch.get_profs_from_vec(v)
            if not scratch.profiles_are_physical():
                return None
            kn, kT, _, _ = scratch.get_grads_on_flux_grid()
            row = rows[key]
            delta = kT[row]-kT0[row] if pressure else kn[row]-kn0[row]
            suffix = "_dkT" if pressure else "_dkn"
            for base in species.get_species_list():
                item = scratch[base.type]
                for attr in attributes:
                    slope = getattr(base, "d"+attr+suffix)[key]
                    setattr(item, attr, getattr(base, attr) + slope*delta)
            # Internal reactions/heat exchange retain their actual n,p
            # dependence; auxiliary and unsupported external slopes are lagged.
            local_physics.compute_sources(scratch, self.geo, species.norms)
            return G*(self.F_plus_vec(scratch) - self.F_minus_vec(scratch)) - self.Svec(scratch)

        jac = np.zeros((y.size, y.size))
        for i, (key, pressure) in enumerate(keys):
            for j in range(self.N_radial-1):
                k = i*self.N_radial + j
                step = 1e-5*abs(y[k])
                if not np.isfinite(step) or step == 0:
                    raise FloatingPointError("Invalid perturbation of profile in transport Jacobian")
                v = y.copy()
                v[k] += step
                fp = residual(v, key, pressure)
                v[k] = y[k] - step
                fm = residual(v, key, pressure)
                if fp is None or fm is None:
                    f0 = residual(y, key, pressure)
                    if f0 is None or (fp is None and fm is None):
                        raise FloatingPointError("No physical stencil for transport Jacobian")
                    jac[:, k] = (fp - f0)/step if fm is None else (f0 - fm)/step
                else:
                    jac[:, k] = (fp - fm)/(2*step)
        if self.external_source_jacobian is not None:
            jac -= self.external_source_jacobian
        return jac

    def F_plus_vec(self, species):

        Fvec = []
        for s in species.n_evolve_keys:
            Fn = np.zeros(self.N_radial)
            Fn[:-1] = species[s].Fn_plus(species.ref_species, self.geo)
            Fvec.append(Fn)

        for s in species.T_evolve_keys:
            Fp = np.zeros(self.N_radial)
            Fp[:-1] = species[s].Fp_plus(species.ref_species, self.geo)
            Fvec.append(Fp)

        return np.array(Fvec).flatten()

    def F_minus_vec(self, species):

        Fvec = []
        for s in species.n_evolve_keys:
            Fn = np.zeros(self.N_radial)
            Fn[:-1] = species[s].Fn_minus(species.ref_species, self.geo)
            Fvec.append(Fn)

        for s in species.T_evolve_keys:
            Fp = np.zeros(self.N_radial)
            Fp[:-1] = species[s].Fp_minus(species.ref_species, self.geo)
            Fvec.append(Fp)

        return np.array(Fvec).flatten()

    def Svec(self, species):
        Svec = []
        for s in species.n_evolve_keys:
            Sn = np.zeros(self.N_radial)
            Sn[:-1] = species[s].Sn[:-1]
            Svec.append(Sn)

        for s in species.T_evolve_keys:
            Sp = np.zeros(self.N_radial)
            Sp[:-1] = 2./3.*species[s].Sp[:-1]
            Svec.append(Sp)

        return np.array(Svec).flatten()

    def solve(self, species, time):

        alpha = time.alpha
        iter_idx = time.iter_idx
        step_idx = time.step_idx
        time.rms = np.inf
        time.steady_rms = np.inf
        time.implicit_rms = np.inf
        self.y_new = None

        d1 = time.d1(step_idx)
        d0 = time.d0(step_idx)
        dm1 = time.dm1(step_idx)
        debug(f"    2nd order time coeffs: d1 = {d1:.2e}, d0 = {d0:.2e}, d-1 = {dm1:.2e}")
        F_plus = self.F_plus_vec(species)
        F_minus = self.F_minus_vec(species)
        S = self.Svec(species)
        G = np.tile((self.geo.G_fac/self.grid.drho).profile, species.evolve_count)
        spatial_residual = G*(F_plus - F_minus) - S

        if iter_idx == 0:
            if not self.reset:
                self.y_old = self.y_m
                self.y_m = self.y_curr
            # Recompute the old-time closure when retrying, without putting a
            # rejected state into the BDF history.
            self.spatial_residual_m = spatial_residual

        time.steady_rms = self.profile_rms(spatial_residual, self.y_curr)
        # d0 = -d1-dm1. Form the discrete defect from differences, then
        # solve for the correction; this avoids subtracting large O(y/dt)
        # or O(J@y) terms when the timestep is small or the closure is stiff.
        defect = d1*(self.y_curr - self.y_m)
        if dm1 != 0:
            defect -= dm1*(self.y_m - self.y_old)
        if alpha > 0:
            defect += alpha*spatial_residual
        if alpha < 1:
            defect += (1-alpha)*self.spatial_residual_m
        # The fixed edge is an algebraic condition, never extrapolated by BDF.
        defect[self.edge] = 0
        time.implicit_rms = self.profile_rms(defect/d1, self.y_curr)

        try:
            if alpha == 0:
                # Explicit Euler never needs to construct or evaluate a Jacobian.
                self.jacob = self.I_mat*d1
                correction = -defect/d1
            else:
                psi = self.psimat(species)
                self.jacob = self.I_mat*d1 + alpha*psi
                self.jacob[self.edge, :] = 0
                self.jacob[:, self.edge] = 0
                idx = np.arange(self.N_radial-1, self.y_curr.size, self.N_radial)
                self.jacob[idx, idx] = d1
                correction = np.linalg.solve(self.jacob, -defect)
            # Keep the original absolute-state RHS for NPY diagnostics.
            self.rhs = self.jacob @ self.y_curr - defect
            y_new = self.y_curr + correction
        except (np.linalg.LinAlgError, FloatingPointError) as exc:
            errr(f"Error in transport solve: {exc}")
            return

        if not np.all(np.isfinite(y_new)):
            errr("Nonfinite result in linear solve")
            return

        # Test the physical state before accepting or exposing it to a flux
        # closure at the next Newton iteration (including dependent species).
        candidate = species.profile_scratch()
        candidate.get_profs_from_vec(y_new)
        if not candidate.profiles_are_physical():
            errr("Transport candidate has nonpositive or nonfinite density/pressure")
            return
        self.y_new = y_new
        time.rms = self.profile_rms(y_new - self.y_curr, self.y_curr)

    def profile_rms(self, delta, y):
        delta = delta.reshape(-1, self.N_radial)[:, :-1]
        y = y.reshape(-1, self.N_radial)[:, :-1]
        if delta.size == 0:
            return 0.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            relative = np.abs(delta/y)
        scale = np.max(relative)
        if not np.isfinite(scale) or scale == 0.:
            return scale
        # Squaring unscaled ratios can overflow or underflow even though
        # the RMS itself is representable.
        return scale*np.sqrt(np.mean((relative/scale)**2))

    def advance(self, species):
        self.y_curr = self.y_new
        # copy vector data into species profiles
        species.get_profs_from_vec(self.y_curr)
        self.reset = False

    def reset_timestep(self, species, time):
        self.y_curr = self.y_m
        species.get_profs_from_vec(self.y_curr)
        time.dtau_old = time.dtau
        time.dtau = time.dtau/time.dtau_adjust
        time.iter_idx = 0
        self.reset = True
        info(f"*** Resetting timestep at time index {time.step_idx}. new dtau = {time.dtau} ***")
