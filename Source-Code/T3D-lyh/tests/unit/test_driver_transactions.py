import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from netCDF4 import Dataset
from t3d.Geometry import Geometry
from t3d.Grid import Grid
from t3d.Import import TrinityReader
from t3d.Profiles import FluxProfile, GridProfile
from t3d.TrinityIO import Input
from tests.unit import test_source_physics_io, test_lyh_time_control


class TestDriverTransactions(unittest.TestCase):
    @patch("t3d.trinity_lib.check_for_stop_file", return_value=False)
    def test_feedback_runs_only_at_accepted_boundary_and_resets_order(self, stop):
        e = test_lyh_time_control.engine(dtau=.125, N_steps=2, newton_threshold=10., dtau_max=.125)
        calls = []
        e.physics.scale_density_to = "sudo_line"
        e.physics.scale_to_sudo_density = Mock(return_value=True)
        e.species.get_vec_from_profs = lambda: e.solver.y_curr*1.1
        old_solve = e.solver.solve

        def solve(species, time):
            calls.append(time.dtau_previous_accepted)
            return old_solve(species,time)
        e.solver.solve = solve
        e.evolve_profiles()
        self.assertEqual(calls,[None,None])
        self.assertEqual(e.physics.scale_to_sudo_density.call_count,2)
        self.assertAlmostEqual(e.solver.y_curr[0], (1.125*1.1+.125)*1.1)

    @patch("t3d.trinity_lib.check_for_stop_file", side_effect=[False, True])
    def test_stop_mid_iteration_restores_state_and_does_not_publish_stale_flux(self, stop):
        e = test_lyh_time_control.engine(dtau=.125, N_steps=2, newton_threshold=1e-8)
        external = Mock(label="external", after_sources=False)
        e.flux_models.get_models_list = lambda:[external]
        e.evolve_profiles()
        self.assertEqual(e.time.step_idx,0)
        self.assertFalse(e.time.diagnostics_current)
        np.testing.assert_array_equal(e.solver.y_curr,[1,1,1])
        self.assertEqual(external.compute_fluxes.call_count,1)

    @patch("t3d.trinity_lib.check_for_stop_file", return_value=False)
    def test_no_unknowns_only_writes_diagnostics(self, stop):
        e = test_lyh_time_control.engine()
        e.solver.N_profiles = 0
        e.solver.solve = Mock()
        e.evolve_profiles()
        e.solver.solve.assert_not_called()
        self.assertEqual(e.time.step_idx,0)
        e.physics.compute_sources.assert_called_once()

    @patch("t3d.trinity_lib.check_for_stop_file", return_value=False)
    def test_write_cadence_counts_accepted_steps_not_newton_iterations(self, stop):
        e = test_lyh_time_control.engine(dtau=.125, N_steps=3, newton_threshold=1e-8,
                                         dtau_max=.125)
        e.logger.n_write=2
        rows = []
        e.logger.export.side_effect=lambda: rows.append((e.time.step_idx,e.time.iter_idx))
        e.evolve_profiles()
        self.assertEqual(rows, [(2,0),(3,0)])


class TestInputOutputAndGeometry(unittest.TestCase):
    def test_file_parse_error_is_not_masked_by_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.in"
            p.write_text("[data]\nwrong : 3\n")
            with self.assertRaisesRegex(ValueError, "line 2"):
                Input(p)
            with self.assertRaises(FileNotFoundError):
                Input(p.with_name("missing.in"))
        self.assertEqual(Input("[inline]\na = 1").input_dict["inline"]["a"],1)

    def test_distinct_output_paths_and_preserve_prior_netcdf_on_failed_write(self):
        h = test_source_physics_io.TestSourcePhysicsAndOutput()
        h.setUp()
        sp, physics, time = h.species(), h.physics(), test_source_physics_io.Time(h.inputs)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            h.inputs["log"]["f_save"] = str(directory / "results.nc")
            logger = h.logger(sp, physics, time)
            self.assertEqual(set((logger.nc_save,logger.bp_save,logger.f_save)), set(map(str,
                             [directory/"results.nc",directory/"results.bp",directory/"results.log.npy"])))
            logger.save(sp,SimpleNamespace(),time,final=True)
            logger.export()
            infile = directory/"input.in"
            infile.write_text("# valid source input copy")
            logger.write_netCDF4(infile.as_posix())
            with Dataset(logger.nc_save) as f:
                self.assertEqual(list(f["time"]["t_geometry_file"][:]),[""])
            saved = Path(logger.nc_save).read_bytes()
            with self.assertRaises(FileNotFoundError):
                logger.write_netCDF4(str(directory/"absent.in"))
            self.assertEqual(Path(logger.nc_save).read_bytes(),saved)
            self.assertEqual(len(list(directory.glob(".results*"))),0)
        h.inputs["log"]["n_write"] = 0
        with self.assertRaisesRegex(ValueError,"positive integer"):
            h.logger(sp, physics, time)

    def test_saved_derivative_uses_accepted_history_and_correct_order(self):
        from t3d.tools.source_sink_plot import profile_time_derivative
        # Newton row is not the accepted value. Variable-step BDF2 integrates
        # a quadratic exactly, and the first-order step is explicitly marked.
        times = np.array([0., 0., .2, .2, .5])
        values = np.array([0., 99., .04, 99., .25])
        accepted = [0,2,4]
        self.assertTrue(np.isnan(profile_time_derivative(values,times,accepted,0,0)))
        self.assertAlmostEqual(profile_time_derivative(values,times,accepted,2,1),.2)
        self.assertAlmostEqual(profile_time_derivative(values,times,accepted,4,2),1.)
        self.assertTrue(np.isnan(profile_time_derivative(values,times,accepted,4,0)))

    def test_missing_data_is_not_a_finite_netcdf_fill_value(self):
        g = Grid({"grid":{"N_radial":4}})
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp)/"masked.nc"
            with Dataset(filename,"w") as f:
                f.createDimension("r",4)
                f.createDimension("t",1)
                f.createGroup("grid").createVariable("rho",float,("r",))[:]=g.rho
                f.createGroup("species").createVariable("n_H",float,("t","r"),fill_value=42.)
            with self.assertRaisesRegex(ValueError,"unavailable at frame"):
                TrinityReader(g,filename).get_density(tag="H")

    def test_geqdsk_uses_capability_not_label_and_line_average_has_no_length_units(self):
        g=Grid({"grid":{"N_radial":5,"flux_label":"rminor"}})
        def geqdsk(geo):
            geo.Btor_eq=2.
            geo.a_minor_eq=.4
            geo.R_major_eq=4.
            geo.area=FluxProfile(g.midpoints,g)
            geo.grho=FluxProfile(1.,g)
        provider=SimpleNamespace(label="custom", get_gs2_eqdsk_geometry=geqdsk)
        models=SimpleNamespace(get_models_list=lambda:[provider])
        geo=Geometry({"geometry":{"geo_option":"geqdsk"}},g,models)
        self.assertEqual((geo.Btor,geo.a_minor,geo.AspectRatio),(2.,.4,4.))
        np.testing.assert_allclose(geo.line_average(GridProfile(3.,g),profile=True).profile,3.)
        self.assertAlmostEqual(geo.line_average(GridProfile(3.,g)),3.)
        with self.assertRaisesRegex(ValueError,"exactly one|requires one"):
            Geometry({"geometry":{"geo_option":"geqdsk"}},g,SimpleNamespace(get_models_list=list))

    def test_vmec_lcfs_no_explicit_field_uses_wout_and_custom_boozer_hook(self):
        g=Grid({"grid":{"N_radial":5}})
        wout=Path(__file__).resolve().parents[1]/"data"/"wout_w7x.nc"
        prep=Mock()
        models=SimpleNamespace(get_models_list=lambda:[SimpleNamespace(label="custom",make_booz=prep)])
        geo=Geometry({"geometry":{"geo_option":"vmec","geo_file":str(wout)},
                      "rescale":{"B_def":"LCFS","B_mult":1.2}},g,models)
        self.assertAlmostEqual(geo.Btor,1.2*geo.Ba_vmec)
        prep.assert_called_once_with(str(wout))
