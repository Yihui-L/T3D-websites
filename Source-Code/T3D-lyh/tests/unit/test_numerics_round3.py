import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from t3d.Grid import Grid
from t3d.Profiles import GridProfile, FluxProfile
from t3d.Time import Time
from t3d.Species import SpeciesDict
from t3d.flux_models.ReLU import ReLU_FluxModel
from t3d.flux_models.RR import RechesterRosenbluth
from tests.unit.test_discrete_jacobian import fixture
from tests.unit.test_lyh_time_control import engine


class TestNumericsAudit(unittest.TestCase):
    def test_three_nodes_exact_quadratic_derivatives_and_linear_interpolation(self):
        g = Grid({"grid": {"N_radial":3}})
        prof = GridProfile(2 + 3*g.rho + 4*g.rho**2,g)
        np.testing.assert_allclose(prof.gradient_4pt().profile, 3 + 8*g.rho)
        np.testing.assert_allclose(prof.gradient_as_FluxProfile().profile, 3 + 8*g.midpoints)
        for size in (3, 5, 9):
            grid = Grid({"grid":{"N_radial":size}})
            np.testing.assert_allclose(FluxProfile(2+3*grid.midpoints,grid).toGridProfile(2).profile,
                                       2+3*grid.rho, atol=1e-14)

    def test_profile_equality_is_not_any_matching_node(self):
        g = Grid({"grid":{"N_radial":3}})
        self.assertNotEqual(GridProfile([1,2,3],g), GridProfile([1,5,9],g))
        self.assertEqual(GridProfile([1,2,3],g),GridProfile([1,2,3],g))
        self.assertNotEqual(GridProfile([1,2,3],g),
                            GridProfile([1,2,3],Grid({"grid":{"N_radial":3,"rho_edge":.5}})))

    def test_floor_for_positive_midpoint_does_not_allow_interpolated_zero(self):
        g = Grid({"grid":{"N_radial":3}})
        f = GridProfile([1.,1.,9.],g).toFluxProfile()
        self.assertEqual(f[0],1.)
        self.assertTrue(np.all(f.profile > 0.))

    def test_flux_faces_telescope_and_have_exact_axis_zero(self):
        sp, geo, _, _ = fixture()
        key = sp.n_evolve_keys[0]
        item = sp[key]
        item.pflux = FluxProfile(np.linspace(6,-5,sp.N_radial-1),sp.grid)
        plus = item.Fn_plus(sp.ref_species,geo).profile
        minus = item.Fn_minus(sp.ref_species,geo).profile
        self.assertEqual(minus[0],0.)
        np.testing.assert_array_equal(minus[1:], plus[:-1])
        self.assertAlmostEqual(sum(plus-minus),plus[-1], places=12)
        sp.clear_fluxes()
        self.assertEqual(max(np.abs(item.Fn_plus(sp.ref_species,geo).profile)),0.)

    def test_newton_difference_resolves_stiff_tiny_correction(self):
        e = engine(dtau=1.,alpha=1.,N_steps=1)
        e.solver.psimat = lambda _: np.diag([1e12,1e12,0.])
        e.solver.Svec = lambda _: np.array([.001,.001,0.])
        e.solver.solve(e.species,e.time)
        # The update is physically representable, but an absolute-state
        # solve subtracts J*y (1e12) and loses precision on a 1e-3 defect.
        expected = 1 + .001/(1+1e12)
        np.testing.assert_array_equal(e.solver.y_new,[expected,expected,1.])
        self.assertLess(e.time.rms, 1e-14)
        self.assertEqual(e.time.implicit_rms,.001)

    def test_variable_bdf_step_after_edge_change_preserves_dirichlet_state(self):
        e = engine(dtau=.1,alpha=1.)
        t=e.time
        t.step_idx=1
        t.dtau_previous_accepted=.1
        e.solver.y_m=np.array([1.,1.,.5])
        e.solver.y_curr=np.array([1.1,1.1,2.])
        e.solver.solve(e.species,t)
        np.testing.assert_allclose(e.solver.y_new,[1.2,1.2,2.],atol=1e-14)
        t.dtau_previous_accepted=None
        self.assertEqual((t.d1(7),t.d0(7),t.dm1(7)),(10.,-10.,0.))

    def test_theta_retry_recomputes_current_explicit_closure_without_shifting_history(self):
        e = engine(dtau=.1,alpha=.5,N_steps=1)
        e.solver.Svec = Mock(side_effect=[np.array([-20.,-20.,0.]),np.array([2.,2.,0.])])
        scalar_profile = e.solver.y_m.copy()
        e.solver.solve(e.species,e.time)
        self.assertIsNone(e.solver.y_new)
        e.solver.reset_timestep(e.species,e.time)
        e.solver.solve(e.species,e.time)
        np.testing.assert_array_equal(e.solver.y_m,scalar_profile)
        np.testing.assert_allclose(e.solver.y_new,[1.1,1.1,1.],atol=1e-14)

    def test_invalid_time_controls_and_radial_coordinate(self):
        for option in ({"max_newton_iter":1.5},{"N_steps":True},{"max_timestep_retries":float('nan')},
                       {"newton_tolerance":float('nan')},{"rms_converged":-1.}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                Time({"time":option})
        with self.assertRaises(ValueError):
            Time({"time":{"use_SI":True}}).normalize_inputs(0.)
        for option in ({"rho_edge":1.05},{"flux_label":"not-a-coordinate"}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                Grid({"grid":option})

    def test_species_selection_rejects_ambiguity_and_honors_explicit_charge_constraint(self):
        g = Grid({})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            SpeciesDict({"species":[{"type":"electron"},{"type":"deuterium"},{"type":"electron"}]},g)
        with self.assertRaisesRegex(ValueError,"Only one"):
            SpeciesDict({"species":[{"type":"electron","qneut_species":True},
                                    {"type":"deuterium","qneut_species":True}]},g)
        sp = SpeciesDict({"species":[{"type":"electron","qneut_species":True},
                                    {"type":"deuterium"}]},g)
        self.assertIs(sp.qneut_species,sp.electron)
        self.assertEqual(sp.n_evolve_keys,["deuterium"])
        sp = SpeciesDict({"species":[{"type":"electron"},{"type":"deuterium"},
                      {"type":"tritium","density":{"equal_to":"0.5 * deuterium"}}]},g)
        self.assertEqual(sp.qneut_species.type,"deuterium")
        self.assertEqual(sp.n_equalto_keys,["tritium"])
        with self.assertRaisesRegex(ValueError,"cannot also"):
            SpeciesDict({"species":[{"type":"electron"},
                        {"type":"deuterium","qneut_species":True,"density":{"equal_to":"electron"}}]},g)

    def test_integer_profile_can_receive_physical_fraction_and_bad_shape_rejected(self):
        g = Grid({"grid":{"N_radial":3}})
        p = GridProfile(np.array([1,2,3]),g)
        p[1] = 1.125
        np.testing.assert_array_equal(p.profile,[1.,1.125,3.])
        with self.assertRaises(ValueError):
            FluxProfile(np.zeros((2,1)),g)

    def test_gradient_probe_requires_finite_nonzero_step(self):
        sp, _, _, _ = fixture()
        e = sp[sp.n_evolve_keys[0]]
        e.rel_step = 0.
        with self.assertRaisesRegex(ValueError,"finite and nonzero"):
            sp.get_grads_on_flux_grid(pert_n=e.type,abs_step=0.)

    def test_relu_noise_preserves_nonzero_base_and_zero_unrelated_jacobian(self):
        g = Grid({"grid":{"N_radial":5}})
        t = Time({})
        model = ReLU_FluxModel({"noise":.9, "random_seed":15},g,t)
        prof=GridProfile(1.,g)
        s = SimpleNamespace(n=lambda: prof,p=lambda: prof,
                            n_relu_critical_gradient0=0.,n_relu_critical_gradient1=0.,
                            p_relu_critical_gradient0=0.,p_relu_critical_gradient1=0.,
                            n_relu_flux_slope=1.,p_relu_flux_slope=1.)
        base = np.ones((1,g.N_radial-1))
        step = np.full((g.N_radial-1,),.01)
        def grads(pert_T=None):
            return base,base if pert_T is None else base+step,2*base,step
        sp = SimpleNamespace(N_species=1,n_evolve_keys=[],T_evolve_keys=["D"],
                             get_species_list=lambda: [s],get_grads_on_flux_grid=grads,
                             add_flux=Mock(),add_dflux_dkT=Mock())
        model.compute_fluxes(sp,None,None)
        particle,heat=sp.add_flux.call_args.args
        np.testing.assert_array_equal(sp.add_dflux_dkT.call_args.args[1],0.)
        np.testing.assert_allclose(sp.add_dflux_dkT.call_args.args[2],heat,atol=2e-14)
        t.iter_idx=2
        t.retry_generation=1
        ReLU_FluxModel({"noise":.1, "random_seed":11},g,t)
        model.compute_fluxes(sp,None,None)
        np.testing.assert_array_equal(sp.add_flux.call_args.args[0],particle)
        np.testing.assert_array_equal(sp.add_flux.call_args.args[1],heat)
        t.step_idx=1
        model.compute_fluxes(sp,None,None)
        self.assertFalse(np.array_equal(sp.add_flux.call_args.args[1],heat))

    def test_rr_evaluates_reference_at_interfaces_and_probes_density(self):
        sp, geo, _, _ = fixture()
        geo.a_minor = 1.
        model = RechesterRosenbluth({},sp.grid,Time({}))
        elec = sp.electron
        ref = sp.ref_species.T().toFluxProfile().profile
        ns = np.full((sp.N_species,sp.N_radial-1), 2.)
        Ts = np.full_like(ns,3.)
        nu = np.full_like(ns,4.)
        old = sp.get_grads_on_flux_grid
        sp.get_profiles_on_flux_grid = Mock(return_value=(ns,Ts,nu))
        _, kT, _, _ = old()
        _, kTn, _, dkn = old(pert_n=sp.n_evolve_keys[0])
        model.compute_fluxes(sp,geo,sp.norms)
        i=sp.sidx_electron
        coeff = (2/3)*ns[i]*Ts[i]**2/(nu[i]*ref**1.5)
        q = coeff*kT[i]**3
        np.testing.assert_allclose(elec.qflux.profile,q)
        expected = (coeff*kTn[i]**3-q)/dkn
        self.assertGreater(np.max(np.abs(expected)),.01)
        np.testing.assert_allclose(elec.dqflux_dkn[sp.n_evolve_keys[0]].profile,expected)
