from t3d.FluxModels import FluxModel
import numpy as np


class ChangHintonNeo_FluxModel(FluxModel):

    def __init__(self, inputs, grid, time):

        # call base class constructor
        super().__init__(inputs, grid, time)

        self.label = inputs.get('label', 'Chang-Hinton')  # label for diagnostics
        self.vt_sqrt_2 = None

    def compute_fluxes(self, species, geo, norms):
        """Add fluxes to each species
        """
        # in the following, A_sjk indicates A[s, j, k] where
        # s is species index
        # j is rho index
        # k is perturb index

        if geo.geo_option != 'miller':
            raise ValueError("Error: Chang-Hinton analytic neoclassical model requires geo_option = 'miller'")

        rho_j = self.rho  # r/a
        rmajor = geo.AspectRatio  # R/a
        rmajor_prime = geo.rmajor_prime
        eps = rho_j/rmajor  # r/R
        qsf = geo.qsf  # q: safety factor
        B_pol = eps/(qsf*(1 + (eps/qsf)**2)**0.5)

        # computations on the geometry which are species independent
        B02_over_B2 = _compute_avg_B_0_sqr_over_B_sqr(shafranov_shift_R_0_prime=rmajor_prime,
                                                      inverse_aspect_ratio=eps)
        B2_over_B02_inv = _compute_avg_B_sqr_over_B_0_sqr_inv(shafranov_shift_R_0_prime=rmajor_prime,
                                                              inverse_aspect_ratio=eps)
        pfirsch_schluter_F = _compute_pfirsch_schluter_F(inverse_aspect_ratio=eps,
                                                         avg_B_0_sqr_over_B_sqr=B02_over_B2,
                                                         avg_B_sqr_over_B_0_sqr_inverse=B2_over_B02_inv)
        K_2_star = _compute_K_2_star(inverse_aspect_ratio=eps, averaged_B_0_sqr_over_B_sqr=B02_over_B2)

        _flux_shape = (species.N_species, self.N_fluxtubes)
        pflux0_sj = np.zeros(_flux_shape)
        qflux0_sj = np.zeros(_flux_shape)
        pflux_sj = np.zeros(_flux_shape)
        qflux_sj = np.zeros(_flux_shape)

        # find index of 'first ion type'
        first_ion_idx = None
        for i, s in enumerate(species.get_species_list()):
            if s.type == species.first_ion_type:
                first_ion_idx = i

        # Bundled-up function to calculate qflux
        def _heat_flux(*, nu_ss, kT_sj):
            # note that this is a closure over the common and fixed parameters:
            # qsf, rmajor, eps, pfirsch_schluter_F, vtfac, and B_pol
            nu_star_i = self._compute_nustar(nu_ss=nu_ss,
                                             q_safety_factor=qsf,
                                             rmajor=rmajor,
                                             inverse_aspect_ratio=eps)
            K_2 = _compute_K_2(K_2_star=K_2_star,
                               nu_star_i=nu_star_i,
                               inverse_aspect_ratio=eps,
                               pfirsch_schluter_F=pfirsch_schluter_F)
            qflux = self._compute_qflux_sj(K_2=K_2,
                                           inverse_aspect_ratio=eps,
                                           kT_sj=kT_sj,
                                           nu_ss=nu_ss,
                                           vtfac=species.norms.vtfac,
                                           B_pol=B_pol)
            return qflux

        # base profiles case
        # get gradient values on flux grid
        # these are 2d arrays, e.g. kn_sj = kap_n[s, j]
        _, kT0_sj, _, _ = species.get_grads_on_flux_grid(pert_n=None, pert_T=None)
        _, _, nu_ss = species.get_profiles_on_flux_grid(normalize=True, a_minor=geo.a_minor)

        qflux0_sj[first_ion_idx, :] = _heat_flux(nu_ss=nu_ss[first_ion_idx, :],
                                                 kT_sj=kT0_sj[first_ion_idx, :])
        # print(f'\n  qflux_neo = {qflux0_sj}\n')
        species.add_flux(pflux0_sj, qflux0_sj, label=self.label)

        # perturbed density cases
        # for each evolved density species, need to perturb density
        for stype in species.n_evolve_keys:
            _, kT_sj, _, dkap = species.get_grads_on_flux_grid(pert_n=stype, pert_T=None)
            qflux_sj[first_ion_idx, :] = _heat_flux(nu_ss=nu_ss[first_ion_idx, :],
                                                    kT_sj=kT_sj[first_ion_idx, :])
            dpflux_dkn_sj = (pflux_sj - pflux0_sj) / dkap
            dqflux_dkn_sj = (qflux_sj - qflux0_sj) / dkap
            species.add_dflux_dkn(stype, dpflux_dkn_sj, dqflux_dkn_sj)

        # perturbed temperature cases
        # for each evolved temperature species, need to perturb temperature
        for stype in species.T_evolve_keys:
            _, kT_sj, _, dkap = species.get_grads_on_flux_grid(pert_n=None, pert_T=stype)
            qflux_sj[first_ion_idx, :] = _heat_flux(nu_ss=nu_ss[first_ion_idx, :],
                                                    kT_sj=kT_sj[first_ion_idx, :])
            dpflux_dkT_sj = (pflux_sj - pflux0_sj) / dkap
            dqflux_dkT_sj = (qflux_sj - qflux0_sj) / dkap
            # print(f'\n  qflux_neo = {qflux_sj}\n')
            species.add_dflux_dkT(stype, dpflux_dkT_sj, dqflux_dkT_sj)

    def _compute_qflux_sj(self, *, K_2, inverse_aspect_ratio, kT_sj, nu_ss, vtfac, B_pol):
        # Where does this formula come from?
        pi = np.pi
        eps = inverse_aspect_ratio
        return K_2 * (8 / pi)**0.5 / 3 * eps**0.5 * kT_sj * nu_ss / (0.5 * vtfac * B_pol**2)

    def _compute_nustar(self, *, nu_ss, q_safety_factor, rmajor, inverse_aspect_ratio):
        # Where does this formula come from?
        """Quantity ν_*

        Parameters
        ----------
        nu_ss : array_like
            ?collisionality between two species?
        q_safety_factor
            Local safety factor. Unitless.
        rmajor
            Major radius
        eps
            ε, inverse aspect ratio. unitless.
        """
        pi = np.pi
        eps = inverse_aspect_ratio
        q = q_safety_factor
        nu_star = 4 / (3 * np.sqrt(pi)) * nu_ss * q * rmajor / eps**1.5
        return nu_star


########################################################################
# Formulas from the paper
#   Chang, C. S., and F. L. Hinton. "Effect of Finite Aspect Ratio
#   on the Neoclassical Ion Thermal Conductivity in the Banana Regime."
#   The Physics of Fluids 25, no. 9 (September 1, 1982): 1493–94.
#   https://doi.org/10.1063/1.863934.
# Implemented in order as in the paper.
########################################################################

def _compute_K_2_star(*,
                      inverse_aspect_ratio,
                      averaged_B_0_sqr_over_B_sqr):
    """Quantity K₂^*

    Parameters
    -----------
    inverse_aspect_ratio
        ε, unitless.
    averaged_B_0_sqr_over_B_sqr: float.
        <B₀² / B²>, unitless quantity defined after Equation (3).

    Notes
    -----
    As Equation (3), but using epsilon instead of δ.
    """
    eps = inverse_aspect_ratio
    return (0.66 + 1.88 * eps**(1/2) - 1.54 * eps) * averaged_B_0_sqr_over_B_sqr


def _compute_avg_B_0_sqr_over_B_sqr(*, shafranov_shift_R_0_prime, inverse_aspect_ratio):
    """Quantity <B₀² / B²>

    Parameters
    ----------
    shafranov_shift_R_0_prime
        R₀', unitless. Defined as dR₀/dr.
    inverse_aspect_ratio
        ε, unitless.

    Notes
    -----
    Formula given after Equation (3) of the Chang-Hinton (1982) paper,
    but using epsilon instead of δ.
    """
    eps = inverse_aspect_ratio
    R_0_prime = shafranov_shift_R_0_prime
    numerator = 1 + (3/2) * (eps**2 + eps * R_0_prime) + (3/8) * R_0_prime * eps**3
    denominator = 1 + (1/2) * eps * R_0_prime
    return numerator / denominator


def _compute_K_2(*,
                 K_2_star,
                 nu_star_i,
                 inverse_aspect_ratio,
                 pfirsch_schluter_F):
    """Quantity K₂

    Parameters
    ----------
    K_2_star
        K₂*, unitless.
    nu_star_i
        ν_{*,i}. Normalized neoclassical ion collisionality?
    inverse_aspect_ratio
        ε, unitless.
    pfirsch_schluter_F: float. unitless.
        F. Unitless quantity defined after Equation (4).

    Notes
    -----
    Equation (4) of the Chang-Hinton (1982) paper,
    but using ε `eps` instead of δ.
    """
    eps = inverse_aspect_ratio
    # constants listed in the paper just below the formula
    K_2_0 = 0.66
    a_2 = 1.03
    b_2 = 0.31
    c_2 = 0.74
    term_1 = (K_2_star/0.66) / (1 + a_2 * nu_star_i**(1/2) + b_2 * nu_star_i)
    term_2 = (c_2**2 / b_2) * nu_star_i * eps**(3/2) / (1 + c_2 * nu_star_i * eps**(3/2))
    K_2 = K_2_0 * (term_1 + term_2 * pfirsch_schluter_F)
    return K_2


def _compute_pfirsch_schluter_F(*,
                                inverse_aspect_ratio,
                                avg_B_0_sqr_over_B_sqr,
                                avg_B_sqr_over_B_0_sqr_inverse):
    """Quantity F, the Pfirsch-Schluter factor

    Parameters
    ----------
    inverse_aspect_ratio
        ε, unitless.
    avg_B_0_sqr_over_B_sqr:
        <B₀²/B²>, unitless quantity defined after Equation (3).
    avg_B_sqr_over_B_0_sqr_inverse
        <B²/B₀²>⁻¹, unitless quantity defined after Equation (4).

    Notes
    -----
    Appears after Equation (4), but using epsilon instead of δ.
    """
    eps = inverse_aspect_ratio
    return (1/2) * eps**(-1/2) * (avg_B_0_sqr_over_B_sqr - avg_B_sqr_over_B_0_sqr_inverse)


def _compute_avg_B_sqr_over_B_0_sqr_inv(*,
                                        shafranov_shift_R_0_prime,
                                        inverse_aspect_ratio):
    """Quantity <B² / B₀²>⁻¹

    Args
    ----
    shafranov_shift_R_0_prime
        R₀', unitless. Defined as dR₀/dr.
    inverse_aspect_ratio
        ε, unitless.

    Notes
    -----
    Formula given after Equation (4) of the Chang-Hinton (1982) paper,
    but using epsilon instead of δ.
    """
    eps = inverse_aspect_ratio
    R_0_prime = shafranov_shift_R_0_prime
    numerator = (1 - eps**2)**(1/2) * (1 + (1/2) * eps * R_0_prime)
    denominator = 1 + (R_0_prime/eps)*((1 - eps**2)**(1/2) - 1)
    return numerator / denominator
