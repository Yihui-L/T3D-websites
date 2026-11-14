import numpy as np
from t3d.FluxModels import FluxModel


def ReLU(x, a=1, m=1, p=1, factor=1):
    """Signed power law outside a symmetric critical-gradient interval."""
    x = np.asarray(x)
    excess = np.abs(x) - a
    return np.where(excess >= 0, m*np.sign(x)*np.maximum(excess, 0)**p*factor, 0.)


def Linear(x, a=1, m=1, p=1, factor=1):
    """Signed power law about the zero-flux gradient."""
    return np.sign(x-a)*np.abs(m*(x-a))**p*factor


class ReLU_FluxModel(FluxModel):

    def __init__(self, pars, grid, time):
        super().__init__(pars, grid, time)
        self.D_neo = pars.get('D_neo', 0.0)
        self.label = pars.get('label', 'ReLU')
        self.p = pars.get('power', 1)
        self.noise = pars.get('noise', 0)
        self.seed = pars.get('random_seed', 10)
        if not np.isfinite(self.p) or self.p <= 0:
            raise ValueError("ReLU power must be finite and positive")
        if not np.isfinite(self.noise) or not 0 <= self.noise <= 1:
            raise ValueError("ReLU noise must be in [0, 1]")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("ReLU random_seed must be a nonnegative integer")

    def logfile_info(self) -> str:
        return 'None'

    def compute_fluxes(self, species, geo, norms):
        shape = (species.N_species, self.N_fluxtubes)
        if self.noise:
            # One reproducible draw per channel/species/radius/physical step.
            # Hold it fixed across Newton, retries, and gradient probes;
            # diagnostics do not consume or reset a shared RNG stream.
            step = getattr(self.time, "step_idx", 0)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            particle_factor, heat_factor = 1 + self.noise*rng.uniform(-1, 1, (2, *shape))
        else:
            particle_factor = heat_factor = np.ones(shape)
        species_list = list(species.get_species_list())
        n = [s.n().toFluxProfile().profile for s in species_list]
        pressure = [s.p().toFluxProfile().profile for s in species_list]

        def evaluate(kn, kT, kp):
            particle = np.zeros(shape)
            heat = np.zeros(shape)
            for i, s in enumerate(species_list):
                particle[i] = n[i]*kn[i]*self.D_neo + Linear(
                    kn[i], a=s.n_relu_critical_gradient0+s.n_relu_critical_gradient1*self.rho,
                    m=s.n_relu_flux_slope, p=self.p, factor=particle_factor[i])
                heat[i] = pressure[i]*kp[i]*self.D_neo + ReLU(
                    kT[i], a=s.p_relu_critical_gradient0+s.p_relu_critical_gradient1*self.rho,
                    m=s.p_relu_flux_slope, p=self.p, factor=heat_factor[i])
            return particle, heat

        kn, kT, kp, _ = species.get_grads_on_flux_grid()
        p0, q0 = evaluate(kn, kT, kp)
        species.add_flux(p0, q0, label=self.label)

        for key in species.n_evolve_keys:
            kn, kT, kp, dkap = species.get_grads_on_flux_grid(pert_n=key)
            p, q = evaluate(kn, kT, kp)
            species.add_dflux_dkn(key,(p-p0)/dkap,(q-q0)/dkap)

        for key in species.T_evolve_keys:
            kn, kT, kp, dkap = species.get_grads_on_flux_grid(pert_T=key)
            p, q = evaluate(kn, kT, kp)
            species.add_dflux_dkT(key,(p-p0)/dkap,(q-q0)/dkap)
