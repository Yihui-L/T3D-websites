import unittest
from types import SimpleNamespace

import numpy as np

from t3d.Grid import Grid
from t3d.Profiles import GridProfile
from t3d.Species import SpeciesDict
from t3d.flux_models.GX import GX_FluxModel
from t3d.flux_models.ReLU import ReLU


class TestFluxConsistency(unittest.TestCase):
    def test_density_gradient_perturbation_preserves_charge_balance(self):
        grid = Grid({"grid": {"N_radial": 5}})
        k = np.ones(grid.N_radial - 1)
        # n_e = 6 n_C + n_D; the independent e gradient perturbation
        # must induce a different *logarithmic* gradient change in D.
        charges = np.array([-1.0, 6.0, 1.0])
        densities = np.array([1.0, 0.1, 0.4])
        species = SpeciesDict.__new__(SpeciesDict)
        species.N_species = len(charges)
        species.N_radial = grid.N_radial
        species.species_dict = {}
        for kind, charge, density in zip(("electron", "carbon", "deuterium"), charges, densities):
            profile = GridProfile(density, grid)
            species.species_dict[kind] = SimpleNamespace(
                type=kind, Z=charge, rel_step=0.0, abs_step=0.5,
                n=lambda p=profile: p, get_fluxgrads=lambda: (k, 2*k, 3*k),
            )
        species.qneut_species = species["deuterium"]
        kn, kt, kp, step = species.get_grads_on_flux_grid(pert_n="electron")
        np.testing.assert_allclose(step, -0.5)
        np.testing.assert_allclose(kn[-1, :], -0.25)
        np.testing.assert_allclose((charges * densities) @ kn, 0, atol=1e-14)
        np.testing.assert_allclose(kn + kt, kp)

    def test_signed_relu_power_on_reverse_gradient(self):
        x = np.array([-3.0, 3.0])
        for power in (1, 2, 1.5):
            with self.subTest(power=power):
                np.testing.assert_allclose(ReLU(x, a=1, m=1, p=power),
                                           [-2**power, 2**power])

    def gx_proxy(self, kn_e, kp_e, base_qe=2.0, ion_first=True, full=False):
        # Align the evolved row in tests that isolate other defects.
        order = slice(None) if ion_first else slice(None, None, -1)
        electron_row = 1 if ion_first else 0

        def get_grads(**kwargs):
            kn = np.array([[0.0], [kn_e]])[order]
            kp = np.array([[1.0], [kp_e]])[order]
            return kn, kp - kn, kp, None

        captured = {}
        species = SimpleNamespace(
            n_evolve_keys=["electron"],
            T_evolve_keys=["electron"],
            get_species_list=lambda: [
                SimpleNamespace(type="deuterium", is_adiabatic=False),
                SimpleNamespace(type="electron", is_adiabatic=False),
            ][order],
            get_grads_on_flux_grid=get_grads,
            add_flux=lambda particle, heat, source=None, label=None: captured.update(
                particle=particle, heat=heat, source=source, label=label),
            add_dflux_dkn=lambda species, particle, heat, source=None: captured.update(dkn=heat, dHkn=source),
            add_dflux_dkT=lambda species, particle, heat, source=None: captured.update(dkp=heat, dHkp=source),
        )
        gx = SimpleNamespace(
            adiabatic_species=None, no_perturb_temperature=[], N_fluxtubes=1,
            kn0_sj=np.array([[0.0], [0.0]])[order], kp0_sj=np.array([[1.0], [2.0]])[order],
            kT0_sj=np.array([[1.0], [2.0]])[order],
            pflux0_sj=np.array([[0.0], [2.0]])[order],
            qflux0_sj=np.array([[0.0], [base_qe]])[order],
            dpflux_dkn_sj={"electron": np.array([[0.0], [3.0]])[order]},
            dqflux_dkn_sj={"electron": np.array([[0.0], [3.0]])[order]},
            dpflux_dkT_sj={"electron": np.array([[0.0], [4.0]])[order]},
            dqflux_dkT_sj={"electron": np.array([[0.0], [4.0]])[order]},
            heat0_sj=np.array([[0.0], [1.25]])[order],
            dheat_dkn_sj={"electron": np.array([[0.0], [-2.0]])[order]},
            dheat_dkT_sj={"electron": np.array([[0.0], [0.5]])[order]},
            label="GX regression",
            info=lambda _: None,
        )
        GX_FluxModel.compute_fluxes_ReLU(gx, species, None, None)
        keys = captured if full else ("particle", "heat", "dkn", "dkp")
        return {key: (captured[key][electron_row, 0] if isinstance(captured[key], np.ndarray)
                     else captured[key]) for key in keys}

    def test_gx_surrogate_identifies_evolved_species_in_full_gradient_array(self):
        self.assertEqual(self.gx_proxy(kn_e=0, kp_e=3),
                         dict(particle=6, heat=6, dkn=3, dkp=4))

    def test_gx_surrogate_uses_fixed_pressure_density_derivative(self):
        self.assertEqual(self.gx_proxy(kn_e=1, kp_e=2, ion_first=False),
                         dict(particle=5, heat=5, dkn=3, dkp=4))

    def test_gx_surrogate_uses_derivative_of_clamped_heat_flux(self):
        self.assertEqual(self.gx_proxy(kn_e=0, kp_e=1, ion_first=False),
                         dict(particle=-2, heat=2, dkn=3, dkp=0))
        h = 1e-5
        for kp in (1.0, 3.0):
            center = self.gx_proxy(kn_e=0, kp_e=kp, ion_first=False)
            plus = self.gx_proxy(kn_e=0, kp_e=kp+h, ion_first=False)
            minus = self.gx_proxy(kn_e=0, kp_e=kp-h, ion_first=False)
            self.assertAlmostEqual(center["dkp"], (plus["heat"] - minus["heat"])/(2*h))
        self.assertEqual(self.gx_proxy(kn_e=0, kp_e=3, base_qe=-6),
                         dict(particle=6, heat=1e-16, dkn=0, dkp=0))

    def test_gx_surrogate_preserves_signed_turbulent_exchange_and_label(self):
        base = self.gx_proxy(kn_e=0, kp_e=2, full=True)
        trial = self.gx_proxy(kn_e=1, kp_e=1, full=True)
        self.assertEqual(base["source"], 1.25)
        self.assertEqual(trial["source"], -1.25)
        self.assertEqual(trial["dHkn"], -2.0)
        self.assertEqual(trial["dHkp"], 0.5)
        self.assertEqual(trial["label"], "GX regression")

    def test_duplicate_flux_diagnostic_labels_are_rejected(self):
        from t3d.FluxModels import FluxModelDict
        grid = Grid({"grid": {"N_radial": 5}})
        with self.assertRaisesRegex(ValueError, "Duplicate flux model label 'ReLU'"):
            FluxModelDict({"model": [{"model": "ReLU"}, {"model": "ReLU"}]}, grid, None)
        models = FluxModelDict({"model": [{"model": "ReLU", "label": "analytic A"},
                                          {"model": "ReLU", "label": "analytic B"}]}, grid, None)
        self.assertEqual([model.label for model in models.get_models_list()],
                         ["analytic A", "analytic B"])


if __name__ == "__main__":
    unittest.main()
