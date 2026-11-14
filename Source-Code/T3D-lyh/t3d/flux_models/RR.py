import numpy as np
from t3d.FluxModels import FluxModel


class RechesterRosenbluth(FluxModel):
    """Cubic electron heat-flux supplement used by Giacomin et al. (2024)."""

    def __init__(self, pars, grid, time):
        super().__init__(pars, grid, time)
        self.alpha_rr = pars.get('alpha_rr', 1.0)
        self.label = pars.get('label', 'RR')
        if not np.isfinite(self.alpha_rr) or self.alpha_rr < 0:
            raise ValueError("RR alpha_rr must be finite and nonnegative")

    def compute_fluxes(self, species, geo, norms):
        zeros = np.zeros((species.N_species, self.N_fluxtubes))
        if self.alpha_rr == 0:
            species.add_flux(zeros, zeros, label=self.label)
            return

        # Keep the existing closure coefficient, but evaluate T_ref at the
        # same interfaces as collisionality and the inverse scale lengths.
        ns, Ts, nu_ss = species.get_profiles_on_flux_grid(normalize=True, a_minor=geo.a_minor)
        i = species.sidx_electron
        Tref = species.ref_species.T().toFluxProfile().profile
        nu = nu_ss[i]*Tref**1.5
        if not np.all(np.isfinite(nu) & (nu > 0)):
            raise ValueError("RR requires finite positive electron collisionality")
        coefficient = self.alpha_rr*(2/3)*ns[i]*Ts[i]**2/nu

        def evaluate(kT):
            q = zeros.copy()
            q[i] = coefficient*kT[i]**3
            return q

        _, kT, _, _ = species.get_grads_on_flux_grid()
        q0 = evaluate(kT)
        species.add_flux(zeros, q0, label=self.label)
        for key in species.n_evolve_keys:
            _, kT, _, dkap = species.get_grads_on_flux_grid(pert_n=key)
            species.add_dflux_dkn(key, zeros, (evaluate(kT)-q0)/dkap)
        for key in species.T_evolve_keys:
            _, kT, _, dkap = species.get_grads_on_flux_grid(pert_T=key)
            species.add_dflux_dkT(key, zeros, (evaluate(kT)-q0)/dkap)
