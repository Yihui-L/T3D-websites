"""Manufactured quasineutral face state and local transport probe regressions."""
import unittest

import numpy as np

from t3d.Grid import Grid
from t3d.Profiles import FluxProfile, GridProfile
from t3d.Species import SpeciesDict

from tests.unit.test_discrete_jacobian import fixture, finite_column, residual


class TestFaceChargeConstraints(unittest.TestCase):
    def make_mixture(self, steep=False):
        sp, geo, physics, solver = fixture(ions=['deuterium','tritium'])
        g=sp.grid
        x=g.rho
        nD=np.array([1.,1.,10.,1.,1.,1.,1.,1.,1.]) if steep else .2+.55*x**3
        nT=1+0*x if steep else .75-.25*x
        for key, n in [('deuterium',nD),('tritium',nT),('electron',nD+nT)]:
            item=sp[key]
            item.n_prof=GridProfile(n,g)
            item.n_init=GridProfile(n,g)
        sp.get_profs_from_vec(sp.get_vec_from_profs())
        return sp, geo, physics, solver

    def assert_charge_neutral(self, sp, face_n, kn):
        weights=np.array([item.Z for item in sp.get_species_list()])[:,None]
        np.testing.assert_allclose(np.sum(weights*face_n,axis=0),0.,atol=1e-14)
        np.testing.assert_allclose(np.sum(weights*face_n*kn,axis=0),0.,atol=1e-13)

    def test_positive_fallback_uses_one_stencil_for_all_species(self):
        sp, _, _, _ = self.make_mixture(steep=True)
        face_n=sp.get_profiles_on_flux_grid()[0]
        np.testing.assert_allclose(face_n[:,0],[1.,1.,2.])
        self.assertTrue(np.all(face_n>0))
        kn,_,_,_=sp.get_grads_on_flux_grid()
        self.assert_charge_neutral(sp,face_n,kn)

    def test_varying_mixture_preserves_charge_in_base_and_all_density_probes(self):
        sp, _, _, _ = self.make_mixture()
        face_n=sp.get_profiles_on_flux_grid()[0]
        kn0, kt0, kp0, _=sp.get_grads_on_flux_grid()
        # Only the dependent density gradient changes relative to the
        # independent finite-difference/interpolation convention.
        for key in sp.n_evolve_keys:
            i=list(sp.species_dict).index(key)
            np.testing.assert_allclose(kn0[i],sp[key].get_fluxgrads()[0].profile)
        for key in [None,*sp.n_evolve_keys]:
            kn,kt,kp,dkap=sp.get_grads_on_flux_grid(pert_n=key)
            self.assert_charge_neutral(sp,face_n,kn)
            np.testing.assert_allclose(kn+kt,kp,atol=2e-15)
            np.testing.assert_allclose(kp,kp0,atol=2e-15)  # independently evolved p
            if key:
                i=list(sp.species_dict).index(key)
                np.testing.assert_allclose(kn[i]-kn0[i],dkap)

    def test_equal_temperature_and_temperature_gradient_in_base_and_probes(self):
        g=Grid({'grid':{'N_radial':5}})
        sp=SpeciesDict({'species':[
            {'type':'electron','qneut_species':True},
            {'type':'tritium','temperature':{'equal_to':'2.0 * deuterium'}},
            {'type':'deuterium'}]},g)
        x=g.rho
        for key,n in [('electron',np.ones_like(x)),
                       ('tritium',.1+.1*x*x),('deuterium',.6-.2*x*x)]:
            item=sp[key]
            item.n_prof=GridProfile(n,g)
            item.n_init=GridProfile(n,g)
            item.T_init=GridProfile(1.,g)
            item.p_prof=item.n_prof*GridProfile(2+3*x*x,g)
        sp.get_profs_from_vec(sp.get_vec_from_profs())
        for params in ({},{'pert_n':'tritium'},{'pert_n':'deuterium'},{'pert_T':'deuterium'}):
            kn,kt,kp,_=sp.get_grads_on_flux_grid(**params)
            # The physical temperature link holds for logarithmic gradients,
            # irrespective of n ratios or the probe direction.
            np.testing.assert_allclose(kt[1],kt[2],rtol=0,atol=1e-15)
            np.testing.assert_allclose(kn+kt,kp,atol=1e-15)

    def test_inherently_neutral_linked_mixture_does_not_need_projection_root(self):
        g=Grid({'grid':{'N_radial':5}})
        sp=SpeciesDict({'species':[
            {'type':'electron'},
            {'type':'tritium','density':{'equal_to':'0.6 * electron'}},
            {'type':'deuterium','density':{'equal_to':'0.4 * electron'}}]},g)
        self.assertIsNone(sp.qneut_species)
        self.assertEqual(sp.n_evolve_keys,['electron'])
        for item in sp.get_species_list():
            item.n_prof=GridProfile(1.0+g.rho**3,g)
            item.n_init=item.n_prof
            item.p_prof=GridProfile(2.0-g.rho**2,g)
        sp.get_profs_from_vec(sp.get_vec_from_profs())
        face_n=sp.get_profiles_on_flux_grid()[0]
        for options in ({},{'pert_n':'electron'},{'pert_T':'electron'}):
            kn,kt,kp,_=sp.get_grads_on_flux_grid(**options)
            self.assert_charge_neutral(sp,face_n,kn)
            np.testing.assert_allclose(kn+kt,kp,atol=1e-15)

    def test_projected_pressure_jacobian_matches_recomputed_residual(self):
        sp, geo, physics, solver = self.make_mixture()
        t = sp['tritium']
        e = sp['electron']
        # One analytic synthetic closure with sensitivity to the dependent T
        # gradient; only the corresponding pressure-column derivative matters.
        def rhs():
            _,kt,_,_=sp.get_grads_on_flux_grid()
            e.qflux=FluxProfile(.4+.7*kt[1],sp.grid)
            e.dqflux_dkT[t.type]=FluxProfile(.7,sp.grid)
            return residual(sp,solver)
        rhs()
        jac=solver.psimat(sp)
        column=(sp.N_dens_profiles+sp.T_evolve_keys.index(t.type))*sp.N_radial+3
        expected=finite_column(sp, rhs, column)
        self.assertGreater(np.max(np.abs(expected)),1.)
        np.testing.assert_allclose(jac[:,column],expected,atol=2e-6,rtol=3e-7)


class TestProfileGeometrySafety(unittest.TestCase):
    def test_profile_never_aliases_input_or_coordinates(self):
        g=Grid({'grid':{'N_radial':3}})
        original=g.rho.copy()
        p=GridProfile(g.rho,g)
        p[0]=2
        np.testing.assert_array_equal(g.rho,original)
        original=g.midpoints.copy()
        f=FluxProfile(g.midpoints,g)
        f[1]=3
        np.testing.assert_array_equal(g.midpoints,original)

    def test_profile_arithmetic_rejects_incompatible_radial_axes(self):
        import operator
        g=Grid({'grid':{'N_radial':3}})
        h=Grid({'grid':{'N_radial':3,'rho_edge':.5}})
        i=Grid({'grid':{'N_radial':3,'flux_label':'rminor'}})
        for op in [operator.add,operator.sub,operator.mul,operator.truediv,operator.pow]:
            for other in [GridProfile(2.,h),GridProfile(2.,i)]:
                with self.assertRaisesRegex(ValueError,'different radial grids'):
                    op(GridProfile(1.,g),other)
        self.assertNotEqual(GridProfile(1.,g),GridProfile(1.,i))

    def test_numpy_scalar_operand_add_multiply_divide_both_orders(self):
        g=Grid({'grid':{'N_radial':3}})
        p=GridProfile(2.,g)
        scalar=np.array(4.)
        for expected,result in [(6.,p+scalar),(6.,scalar+p),(8.,p*scalar),(8.,scalar*p),
                                (.5,p/scalar),(2.,scalar/p),(-2.,p-scalar),(2.,scalar-p)]:
            np.testing.assert_array_equal(result.profile,expected)


class TestManufacturedTimeConvergence(unittest.TestCase):
    @staticmethod
    def march(alpha, step_count):
        # Smooth positive y'=lambda*y with exact fixed value at the last
        # radial degree of freedom. Use alternating accepted steps.
        from tests.unit.test_lyh_time_control import engine
        e=engine(alpha=alpha)
        slope=-.7
        e.solver.Svec=lambda _: np.r_[slope*e.solver.y_curr[:-1],0.]
        e.solver.psimat=lambda _: np.diag([-slope,-slope,0.])
        for i in range(step_count):
            e.time.dtau=(.7 if i%2==0 else 1.3)/step_count
            e.time.iter_idx=0
            e.solver.solve(e.species,e.time)
            assert e.solver.y_new is not None
            e.solver.advance(e.species)
            if alpha > 0.:
                e.time.iter_idx=1
                e.solver.solve(e.species,e.time)
                assert e.time.implicit_rms < 3e-16
                e.solver.advance(e.species)
            e.time.dtau_previous_accepted=e.time.dtau
            e.time.step_idx+=1
            e.time.time+=e.time.dtau
        assert abs(e.time.time-1.) < 2e-14
        assert e.solver.y_curr[-1]==1.
        return abs(e.solver.y_curr[0]-np.exp(slope))

    def test_variable_step_refinement_exhibits_expected_order(self):
        for alpha, minimum, maximum in [(0.,1.8,2.2),(.5,3.8,4.2),(1.,3.5,4.5)]:
            with self.subTest(alpha=alpha):
                coarse,fine=[self.march(alpha,count) for count in (32,64)]
                ratio=coarse/fine
                self.assertGreater(ratio,minimum)
                self.assertLess(ratio,maximum)

    def test_rms_reduces_large_and_small_finite_relative_errors(self):
        from tests.unit.test_lyh_time_control import engine
        solver=engine().solver
        for magnitude in (1e-200,1e200):
            self.assertEqual(solver.profile_rms(np.array([magnitude,-magnitude,77.]),np.ones(3)),
                             magnitude)
        self.assertEqual(solver.profile_rms(np.array([]),np.array([])),0.)
