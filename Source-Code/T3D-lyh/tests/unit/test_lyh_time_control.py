import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from t3d.Grid import Grid
from t3d.Profiles import GridProfile
from t3d.Solver import TransportSolver
from t3d.Time import Time
from t3d.trinity_lib import TrinityEngine


class ConstantRateSolver(TransportSolver):
    """u' = 1 in one evolved cell, fixed value at the edge."""

    def F_plus_vec(self, species):
        return np.zeros(self.N_radial)

    F_minus_vec = F_plus_vec

    def Svec(self, species):
        return np.array([1.0, 1.0, 0.0])

    def psimat(self, species):
        return np.zeros((self.N_radial, self.N_radial))


def engine(**options):
    e = TrinityEngine.__new__(TrinityEngine)
    e.time = Time({"time": options})
    e.grid = Grid({"grid": {"N_radial": 3}})
    e.geometry = SimpleNamespace(evolve_equilibrium=False, G_fac=GridProfile(1.0, e.grid))
    def profile_scratch():
        candidate = SimpleNamespace()
        def unpack(vec):
            candidate.values = vec
        candidate.get_profs_from_vec = unpack
        candidate.profiles_are_physical = lambda: np.all(np.isfinite(candidate.values) & (candidate.values > 0))
        return candidate
    e.species = SimpleNamespace(
        N_profiles=1, evolve_count=1, get_vec_from_profs=lambda: np.array([1.0, 1.0, 1.0]),
        get_profs_from_vec=Mock(), clear_fluxes=Mock(), clear_sources=Mock(),
        profiles_are_physical=lambda: True,
        profile_scratch=profile_scratch)
    e.physics = SimpleNamespace(
        update_edge=False, update_source=False, scale_density_to=None, compute_sources=Mock())
    e.solver = ConstantRateSolver(e.grid, e.species, e.geometry, e.physics)
    e.flux_models = SimpleNamespace(get_models_list=list)
    e.source_models = SimpleNamespace(get_sources_list=list)
    e.norms = SimpleNamespace(t_ref=1.0)
    e.logger = Mock(n_write=1000)
    e.trinity_infile = "unused.in"
    return e


@patch("t3d.trinity_lib.check_for_stop_file", return_value=False)
class TestTimeControl(unittest.TestCase):
    def test_variable_step_bdf2_and_terminal_refresh(self, stop):
        e = engine(dtau=0.1, dtau_max=0.2, newton_threshold=10, dtau_increase_threshold=10,
                   t_max=0.5, N_steps=9)
        e.evolve_profiles()
        self.assertAlmostEqual(e.time.time, 0.5)
        self.assertEqual(e.time.step_idx, 3)
        self.assertAlmostEqual(e.time.dtau_previous_accepted, 0.2)
        np.testing.assert_allclose(e.solver.y_curr, [1.5, 1.5, 1.0], rtol=0, atol=1e-14)
        self.assertEqual(e.physics.compute_sources.call_count, 4)
        self.assertTrue(e.time.diagnostics_current)

    def test_terminal_time_ignores_roundoff_only_extra_step(self, stop):
        e = engine(dtau=0.1, dtau_max=0.1, t_max=1.0, alpha=0)
        e.evolve_profiles()
        self.assertEqual(e.time.step_idx,10)
        self.assertAlmostEqual(e.time.time,1.0,places=15)

    def test_retry_keeps_previous_accepted_bdf2_history(self, stop):
        e = engine(dtau=0.1, dtau_max=0.2, newton_threshold=10, dtau_increase_threshold=10,
                   t_max=1, N_steps=2)
        genuine_solve = e.solver.solve
        attempts = []

        def solve(species, time):
            attempts.append((time.time, time.dtau, time.dtau_previous_accepted, time.retry_generation))
            genuine_solve(species, time)
            if len(attempts) == 2:
                time.rms = np.inf

        e.solver.solve = solve
        e.evolve_profiles()
        self.assertEqual(attempts, [(0, 0.1, None, 0), (0.1, 0.2, 0.1, 0), (0.1, 0.1, 0.1, 1)])
        self.assertEqual(e.time.retry_generation, 0)
        self.assertAlmostEqual(e.time.time, 0.2)
        np.testing.assert_allclose(e.solver.y_curr, [1.2, 1.2, 1.0], rtol=0, atol=1e-14)

    def test_small_second_newton_update_does_not_end_time_evolution(self, stop):
        e = engine(dtau=0.1, dtau_max=0.1, N_steps=2, t_max=1)
        e.evolve_profiles()
        self.assertEqual(e.time.step_idx, 2)
        self.assertAlmostEqual(e.time.time, 0.2)
        np.testing.assert_allclose(e.solver.y_curr, [1.2, 1.2, 1.0], rtol=0, atol=1e-14)

    def test_singular_implicit_attempt_cannot_advance_and_retries_are_bounded(self, stop):
        e = engine(dtau=0.1, alpha=1, max_timestep_retries=2)
        e.solver.I_mat[:] = 0.0
        with self.assertRaisesRegex(RuntimeError, "exhausted 2 timestep retries"):
            e.evolve_profiles()
        self.assertEqual(e.time.time, 0)
        self.assertEqual(e.time.step_idx, 0)
        self.assertTrue(np.isinf(e.time.rms))
        self.assertIsNone(e.solver.y_new)
        self.assertEqual(e.physics.compute_sources.call_count, 3)

    def test_explicit_skips_jacobian_and_initial_step_respects_cap(self, stop):
        e = engine(dtau=.1, dtau_max=.01, alpha=0, N_steps=1)
        e.solver.psimat = Mock(side_effect=AssertionError("explicit solve must not build a Jacobian"))
        e.evolve_profiles()
        self.assertEqual(e.time.time,.01)
        np.testing.assert_allclose(e.solver.y_curr, [1.01, 1.01, 1.0])

    def test_nonpositive_explicit_candidate_retries_without_advancing(self, stop):
        e = engine(dtau=.1, alpha=0, N_steps=1)
        e.solver.Svec = lambda _: np.array([-20., -20., 0.])
        e.evolve_profiles()
        self.assertEqual(e.time.time,.025)
        np.testing.assert_allclose(e.solver.y_curr, [.5, .5, 1.0])

    def test_bdf_history_cancels_newton_update_but_not_residual(self, stop):
        e = engine(dtau=.125, dtau_max=.125, N_steps=2, newton_threshold=10.)
        e.solver.Svec = lambda _: np.array([1., 1., 0.]) if e.time.time == 0 else np.array([-.5, -.5, 0.])
        e.evolve_profiles()
        self.assertEqual(e.time.time,.25)
        self.assertEqual(e.time.step_idx,2)
        self.assertEqual(e.time.rms,0.0)
        self.assertGreater(e.time.steady_rms,.4)

    def test_zero_duration_with_external_closure_never_launches_model(self, stop):
        e = engine(t_max=0)
        backend = Mock(label="external")
        e.flux_models.get_models_list = lambda: [backend]
        e.evolve_profiles()
        self.assertEqual(e.time.time, 0)
        self.assertEqual(e.time.step_idx, 0)
        self.assertTrue(np.isnan(e.time.rms))
        self.assertFalse(e.time.diagnostics_current)
        backend.compute_fluxes.assert_not_called()
        e.logger.save.assert_called_once_with(e.species, e.solver, e.time, final=True)

    def test_zero_duration_local_closure_evaluates_only_diagnostics(self, stop):
        e = engine(t_max=0)
        e.solver.solve = Mock()
        e.evolve_profiles()
        self.assertTrue(e.time.diagnostics_current)
        e.physics.compute_sources.assert_called_once()
        e.solver.solve.assert_not_called()

    def test_si_normalization_keeps_dimensionless_controls(self, stop):
        t = Time({"time": dict(use_SI=True, dtau=0.1, dtau_max=0.4, t_max=2,
                               dtau_adjust=3, dtau_increase_threshold=0.012)})
        t.normalize_inputs(0.2)
        self.assertEqual((t.dtau, t.dtau_max, t.t_max), (0.5, 2, 10))
        self.assertEqual((t.dtau_adjust, t.dtau_increase_threshold), (3, 0.012))
