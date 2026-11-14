import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import f90nml
import h5py
import numpy as np
from scipy.constants import proton_mass

from t3d.SourceModels import SourceModelDict
from t3d.Time import Time
from tests.unit.test_discrete_jacobian import fixture


class TestBeamInterface(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        app = self.root / "xbeams3d"
        app.touch(mode=0o700)
        self.vmec = self.root / "wout_test.nc"
        self.vmec.touch()
        f90nml.write({"beams3d_input": {"mass_beams": 2*proton_mass}}, self.root/"input.test")
        self.sp, _, self.phys, self.solver = fixture(ions=["deuterium", "tritium"])
        self.sp.norms.Sn_ref_SI20 = 3.
        self.grid = self.sp.grid
        self.geo = SimpleNamespace(geo_file=str(self.vmec))
        self.pars = {"source":"BEAMS3D", "beams3d_path":str(self.root),
                     "beams3d_outputs":str(self.root/"out"), "beams3d_template":str(self.root/"input.test"),
                     "particle_source":True, "power_source":True}

    def model(self, **kw):
        return SourceModelDict({"source": self.pars | kw}, self.grid, Time({}))["BEAMS3D"]

    def test_namelist_units_axis_dimensions_and_distinct_trials(self):
        paths = []
        def finish(argv, cwd, **kw):
            paths.append(cwd)
            self.assertEqual(argv[-3:], ["-vmec", "t3d", "-collisions"])
            self.assertTrue(kw["check"])
            self.assertEqual((cwd/"wout_t3d.nc").resolve(), self.vmec)
            values = f90nml.read(cwd/"input.t3d")["beams3d_input"]
            np.testing.assert_allclose(values["ne_aux_s"], np.r_[0, self.grid.rho**2, 1])
            np.testing.assert_allclose(values["ne_aux_f"], 1e20)
            np.testing.assert_allclose(np.array(values["te_aux_f"])[1:-1], 1e3*self.sp.electron.T().profile)
            np.testing.assert_allclose(np.array(values["ti_aux_f"])[1:-1], 1e3*self.sp.bulk_ion.T().profile)
            np.testing.assert_allclose(values["ni_aux_f"], 5e19)
            self.assertEqual(np.shape(values["ni_aux_f"]), (self.grid.N_radial+2, 2))
            np.testing.assert_allclose(np.asarray(values["ni_aux_m"])/proton_mass, self.sp.get_masses(drop="electron"))
            (cwd/"beams3d_t3d.h5").touch()
        model = self.model()
        n,T,_ = self.sp.get_profiles_on_grid()
        with patch("t3d.source_models.BEAMS3D.subprocess.run", side_effect=finish):
            model.run_beams3d(n,T,self.sp,self.geo)
            model.time.retry_generation = 1
            model.run_beams3d(n,T,self.sp,self.geo)
        self.assertNotEqual(paths[0], paths[1])
        self.assertIn("-r1-",paths[1].name)

    def test_overwrites_fortran_profile_slice_starting_indices(self):
        template = self.root/"input.indexed"
        template.write_text("&beams3d_input\n ni_aux_s(2:3)=.2,.4\n ni_aux_f(2,4)=2.\n"
                            " ne_aux_f(3)=1.5e20\n te_aux_s(4)=.5\n /\n")
        model = self.model(beams3d_template=str(template))
        n, T, _ = self.sp.get_profiles_on_grid()
        def finish(argv, cwd, **kw):
            p = f90nml.read(cwd/"input.t3d")["beams3d_input"]
            self.assertEqual(p.start_index["ni_aux_f"], [1, 1])
            for key in ("ni_aux_s", "ne_aux_f", "te_aux_s"):
                self.assertEqual(p.start_index[key], [1])
            np.testing.assert_allclose(p["ni_aux_s"], np.r_[0., self.grid.rho**2, 1.])
            (cwd/"beams3d_t3d.h5").touch()
        with patch("t3d.source_models.BEAMS3D.subprocess.run", side_effect=finish):
            model.run_beams3d(n, T, self.sp, self.geo)

    def test_beam_isotope_check_respects_sparse_beam_line_indices(self):
        template = self.root/"input.indexed"
        n, T, _ = self.sp.get_profiles_on_grid()
        mass = self.sp.bulk_ion.mass*proton_mass
        template.write_text(f"&beams3d_input\n power_beams(4)=1.\n mass_beams(3)={mass}\n /\n")
        with self.assertRaisesRegex(ValueError, "beam 4 particle_source"):
            self.model(beams3d_template=str(template)).run_beams3d(n, T, self.sp, self.geo)
        template.write_text(f"&beams3d_input\n power_beams(4)=1.\n mass_beams(2)={3*proton_mass}\n"
                            f" mass_beams(4)={mass}\n /\n")
        def finish(argv, cwd, **kw):
            (cwd/"beams3d_t3d.h5").touch()
        with patch("t3d.source_models.BEAMS3D.subprocess.run", side_effect=finish):
            self.model(beams3d_template=str(template)).run_beams3d(n, T, self.sp, self.geo)

    def test_reads_radial_times_beam_and_enabled_channels(self):
        h5 = self.root/"deposition.h5"
        with h5py.File(h5,"w") as handle:
            handle["dist_rhoaxis"]=[.05,.55,.95]
            handle["ndot_prof"]=np.array([[1,2],[2,3],[3,4]])*1.e20
            handle["epower_prof"]=np.array([[1,2],[2,3],[3,4]])*1.e6
            handle["ipower_prof"]=np.array([[1,2],[2,3],[3,4]])*2.e6
        model = self.model()
        Sn, Sp=model.read_beams3d_deposition(h5,self.sp,self.sp.norms)
        electron=self.sp.sidx_electron
        bulk=list(self.sp.get_species_list()).index(self.sp.bulk_ion)
        expected=np.interp(self.grid.rho,[.05,.55,.95],[3,5,7])
        np.testing.assert_allclose(Sn[bulk],expected/3)
        np.testing.assert_allclose(Sn[electron],expected/3)
        np.testing.assert_allclose(Sp[electron],expected/2)
        np.testing.assert_allclose(Sp[bulk],expected)
        Sn,_=self.model(particle_source=False).read_beams3d_deposition(h5,self.sp,self.sp.norms)
        np.testing.assert_array_equal(Sn,0)

    def test_independent_np_jacobian_reaches_discrete_residual(self):
        sp=self.sp
        norms=sp.norms
        model=self.model(implicit=True)
        def run(ns, Ts, state, geo):
            np.testing.assert_allclose(ns[sp.sidx_electron],
                                       sum(item.Z*ns[i] for i,item in enumerate(state.get_species_list())
                                           if item.type!="electron"))
            return ns,Ts
        with patch.object(model,"run_beams3d",side_effect=run), patch.object(
                model,"read_beams3d_deposition",side_effect=lambda out,*a: (out[0],3*out[0]*out[1])):
            model.compute_sources(sp,self.geo,norms)
        jac=model.source_jacobian
        n=sp.N_radial
        np.testing.assert_allclose(np.diag(jac),
            np.tile(np.r_[np.ones(n-1),0.],sp.N_dens_profiles).tolist() +
            np.tile(np.r_[np.full(n-1,2.),0.],sp.N_profiles-sp.N_dens_profiles).tolist(),
            rtol=1e-11,atol=1e-11)
        base=self.solver.psimat(sp)
        self.solver.external_source_jacobian=jac
        np.testing.assert_allclose(self.solver.psimat(sp),base-jac)

    def test_missing_and_failed_child_outputs_not_reused(self):
        m=self.model()
        n,T,_=self.sp.get_profiles_on_grid()
        with patch("t3d.source_models.BEAMS3D.subprocess.run",return_value=None):
            with self.assertRaisesRegex(RuntimeError, "did not produce"):
                m.run_beams3d(n,T,self.sp,self.geo)
        with patch("t3d.source_models.BEAMS3D.subprocess.run",
                   side_effect=subprocess.CalledProcessError(2,"xbeams3d")):
            with self.assertRaises(subprocess.CalledProcessError):
                m.run_beams3d(n,T,self.sp,self.geo)
