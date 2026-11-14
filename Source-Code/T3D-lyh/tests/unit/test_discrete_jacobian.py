import unittest
from types import SimpleNamespace

import numpy as np

from t3d.Grid import Grid
from t3d.Physics import Physics, Xsec
from t3d.Profiles import FluxProfile, GridProfile
from t3d.Solver import TransportSolver
from t3d.Species import SpeciesDict
from t3d.Time import Time


def fixture(ions=None, evolve_density=True, **physics_inputs):
    g = Grid({"grid": {"N_radial": 9}})
    ions = ["deuterium"] if ions is None else ions
    spec = [{"type": kind, "density": {"evolve": evolve_density}}
            for kind in ions + ["electron"]]
    sp = SpeciesDict({"species": spec}, g)
    norms = SimpleNamespace(t_ref=1.0, P_ref_MWm3=2.0)
    sp.norms = norms
    for s in sp.get_species_list():
        density = 1.0 if s.type == "electron" else 1.0/len(ions)
        s.n_init = GridProfile(density, g)
        s.n_prof = GridProfile(density, g)
        pressure = (2.0 - 0.4*g.rho**2) if s.type != "electron" else 1.0+0.1*g.rho
        s.p_prof = GridProfile(pressure, g)
        s.T_init = s.T()
        s.norms = norms
    geo = SimpleNamespace(G_fac=GridProfile(1.0, g), area=FluxProfile(1.0, g), Btor=1.0)
    physics = Physics.__new__(Physics)
    physics.grid, physics.geo, physics.norms = g, geo, norms
    for name in ("collisions", "alpha_heating", "radiation", "bremsstrahlung", "turbulent_exchange"):
        setattr(physics, name, physics_inputs.get(name, False))
    physics.scale_density_to = None
    return sp, geo, physics, TransportSolver(g, sp, geo, physics)


def finite_column(species, rhs, column, step=1e-6):
    y = species.get_vec_from_profs()
    delta = np.zeros_like(y)
    delta[column] = step
    species.get_profs_from_vec(y + delta)
    plus = rhs()
    species.get_profs_from_vec(y - delta)
    minus = rhs()
    species.get_profs_from_vec(y)
    rhs()
    return (plus - minus)/(2*step)


def residual(sp, solver):
    G = np.tile((solver.geo.G_fac/solver.grid.drho).profile, sp.evolve_count)
    return G*(solver.F_plus_vec(sp) - solver.F_minus_vec(sp)) - solver.Svec(sp)


class TestDiscreteJacobian(unittest.TestCase):
    def test_gradient_jacobian_matches_actual_compact_operator_and_handles_zero_flux(self):
        sp, geo, physics, solver = fixture(evolve_density=False)
        ion = sp["deuterium"]
        kappa_at_zero = ion.get_fluxgrads()[1][3]
        def rhs():
            ion.qflux = FluxProfile(.7*(ion.get_fluxgrads()[1].profile-kappa_at_zero), sp.grid)
            ion.dqflux_dkT[ion.type] = FluxProfile(.7, sp.grid)
            return residual(sp, solver)
        rhs()
        jac = solver.psimat(sp)
        self.assertEqual(ion.qflux[3], 0.)
        self.assertTrue(np.isfinite(jac).all())
        numeric = finite_column(sp, rhs, 1)
        self.assertGreater(abs(numeric[6]), .1)
        np.testing.assert_allclose(jac[:, 1], numeric, atol=2e-6, rtol=2e-7)
        t = Time({"time":{"dtau": .01}})
        solver.solve(sp, t)
        self.assertTrue(np.isfinite(t.rms))

    def test_quasineutral_reference_density_is_in_jacobian(self):
        sp, geo, physics, solver = fixture()
        self.assertEqual(sp.ref_species.type,sp.qneut_species.type)
        for s in sp.get_species_list():
            s.pflux = FluxProfile(1.0, sp.grid)
            s.qflux = FluxProfile(1.0, sp.grid)
            for t in sp.get_species_list():
                s.dpflux_dkn[t.type] = FluxProfile(0., sp.grid)
                s.dqflux_dkn[t.type] = FluxProfile(0., sp.grid)
        jac = solver.psimat(sp)
        numeric = finite_column(sp, lambda: residual(sp, solver), 3)
        self.assertGreater(np.max(abs(numeric)), 1)
        np.testing.assert_allclose(jac[:, 3], numeric, atol=2e-6, rtol=2e-7)

    def test_actual_collision_term_and_quasineutral_source_derivative(self):
        sp, geo, physics, solver = fixture(collisions=True)
        def rhs():
            physics.compute_sources(sp,geo,sp.norms)
            return residual(sp,solver)
        rhs()
        jac = solver.psimat(sp)
        for column in (3, 12, 21):
            numeric = finite_column(sp,rhs,column)
            np.testing.assert_allclose(jac[:,column],numeric,atol=2e-6,rtol=2e-7)
        np.testing.assert_allclose((sp["deuterium"].Sp+sp["electron"].Sp).profile,0,atol=2e-12)

    def test_actual_alpha_source_directional_derivative(self):
        sp, geo, physics, solver = fixture(ions=["deuterium","tritium"],alpha_heating=True)
        physics.f_helium = 0.
        physics.alpha_efficiency = .63
        physics.xs = {"D-T": Xsec.read_xsec("D_T_-_a_n.txt")}
        for item in sp.get_species_list():
            item.p_prof *= 5
        def rhs():
            physics.compute_sources(sp, geo, sp.norms)
            return residual(sp, solver)
        rhs()
        jac = solver.psimat(sp)
        # Distinct directions: independent D/e densities, D/e pressures;
        # all of them alter deposition, including dependent T density.
        for column in (3, 12, 21, 39):
            numeric = finite_column(sp,rhs,column)
            self.assertGreater(np.max(abs(numeric)), 1e-6)
            np.testing.assert_allclose(jac[:,column],numeric,atol=1e-6,rtol=3e-7)

    def test_positive_independent_state_can_make_dependent_ion_negative(self):
        sp, geo, physics, solver = fixture(ions=["deuterium","tritium"])
        self.assertEqual(sp.qneut_species.type,"tritium")
        self.assertEqual(sp.n_evolve_keys,["deuterium","electron"])
        self.assertTrue(sp.profiles_are_physical())
        sp["deuterium"].Sn = GridProfile(3.0, sp.grid)
        t = Time({"time":{"dtau":.2,"alpha":0}})
        solver.solve(sp,t)
        self.assertTrue(np.isinf(t.rms))
        self.assertIsNone(solver.y_new)
        self.assertTrue(sp.profiles_are_physical())
        self.assertAlmostEqual(sp["tritium"].n()[0],.5)

    def test_constraints_handle_density_and_temperature_chains(self):
        g = Grid({"grid":{"N_radial":5}})
        inputs = {"species":[
            {"type":"tritium","density":{"equal_to":"0.5 * deuterium"},
             "temperature":{"equal_to":"2 * deuterium"}},
            {"type":"deuterium","density":{"evolve":True}},
            {"type":"electron","temperature":{"evolve":False}}]}
        sp=SpeciesDict(inputs,g)
        for s in sp.get_species_list():
            s.n_init = GridProfile(.5 if s.type=="deuterium" else 1.,g)
            s.n_prof=GridProfile(s.n_init.profile.copy(),g)
            s.p_prof=GridProfile(1.,g)
            s.T_init=GridProfile(1.,g)
        sp.get_profs_from_vec(sp.get_vec_from_profs())
        np.testing.assert_allclose(sp["deuterium"].n().profile,2./3)
        np.testing.assert_allclose(sp["tritium"].n().profile,1./3)
        np.testing.assert_allclose(sp["tritium"].T().profile,2*sp["deuterium"].T().profile)
        kn0, kt0, _, _=sp.get_grads_on_flux_grid()
        kn, kt, _, _=sp.get_grads_on_flux_grid(pert_n="electron")
        # Equal profiles respond by the same relative log-density perturbation.
        np.testing.assert_allclose((kn-kn0)[0],(kn-kn0)[1])
        np.testing.assert_allclose((kt-kt0)[0],(kt-kt0)[1])
        np.testing.assert_allclose((kt-kt0)[2],0.)
        charge_dkn = sum(s.Z*s.n().toFluxProfile().profile*row
                          for s,row in zip(sp.get_species_list(),kn-kn0))
        np.testing.assert_allclose(charge_dkn,0.,atol=1e-14)

    def test_banded_compact_gradient_matches_original_dense_form(self):
        rng = np.random.default_rng(141)
        for count in (3, 4, 5, 9, 64, 129):
            grid = Grid({"grid": {"N_radial": count}})
            x, h = grid.rho, grid.drho
            lower = np.ones(count - 1)
            upper = np.ones(count - 1)
            lower[-1] = upper[0] = .5
            diagonal = np.full(count, 3.0)
            diagonal[[0, -1]] = .5
            diagonal[[1, -2]] = 4.0
            dense = np.diag(lower, -1) + np.diag(diagonal) + np.diag(upper, 1)
            inverse = np.linalg.inv(dense)
            for f in (np.full(count, 2.0), x, np.sin(4*x).tolist(), rng.normal(size=count)):
                rhs = np.zeros(count)
                rhs[0] = (f[1] - f[0])/h
                rhs[1] = 3*(f[2] - f[0])/h
                for j in range(2, count - 2):
                    rhs[j] = (.25*(f[j+2] - f[j-2]) + 7*(f[j+1] - f[j-1]))/(3*h)
                rhs[-2] = 3*(f[-1] - f[-3])/h
                rhs[-1] = (f[-1] - f[-2])/h
                got = GridProfile(f, grid).gradient_5pt().profile
                np.testing.assert_allclose(got, inverse@rhs, atol=2e-13, rtol=2e-13)
                np.testing.assert_allclose(dense@got, rhs, atol=3e-13, rtol=2e-13)

    def test_minimum_valid_grid(self):
        for count in (0,1,2):
            with self.assertRaisesRegex(ValueError,"at least 3"):
                Grid({"grid":{"N_radial":count}})
        g=Grid({"grid":{"N_radial":3}})
        np.testing.assert_allclose(GridProfile(g.rho,g).gradient_5pt().profile,1.)
