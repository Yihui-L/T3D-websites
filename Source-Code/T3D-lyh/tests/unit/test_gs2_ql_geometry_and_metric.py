import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import f90nml
import numpy as np
from netCDF4 import Dataset

from t3d.Grid import Grid
from t3d.flux_models.GS2_QL import GS2_QL_FluxModel, a, b


class TestGS2IO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = GS2_QL_FluxModel.__new__(GS2_QL_FluxModel)
        self.model.grid = Grid({"grid": {"N_radial": 4, "rho_edge": 0.8}})
        self.model.rho = self.model.grid.midpoints
        self.model.N_fluxtubes = len(self.model.rho)
        self.model.out_dir = str(self.root)
        self.model.time = SimpleNamespace(step_idx=0, iter_idx=0)
        self.model.label = "GS2"
        self.model.overwrite = False
        self.model.use_toroidal_n = True
        self.model.nth0 = 3
        self.model.th0 = np.array([0., 0.4, np.pi])
        self.model.electromagnetic = False
        self.model.inputs = f90nml.Namelist({
            "knobs":{"fapar": 1, "fbpar":1},
            "kt_grids_single_parameters":{"n0":12},
            "theta_grid_parameters":{"nperiod": 3}, "theta_grid_eik_knobs":{},
            "species_knobs":{}, "gs2_diagnostics_knobs":{}
        })

    def test_geqdsk_dry_initialization_and_radial_output(self):
        m = self.model
        eq = self.root/"g file"
        eq.write_text("header\nparameters\n     0.35000D+01\n")
        geo = SimpleNamespace(geo_option="geqdsk", geo_file=str(eq))

        def run(fins):
            self.assertEqual(len(fins), m.N_fluxtubes)
            for i, path in enumerate(fins):
                nml = f90nml.read(path)
                self.assertEqual(nml["theta_grid_eik_knobs"]["eqfile"], str(eq))
                self.assertEqual(nml["theta_grid_eik_knobs"]["irho"], 1)
                self.assertEqual(nml["theta_grid_parameters"]["nperiod"], 1)
                self.assertEqual(nml["knobs"]["nstep"], 1)
                self.assertEqual(nml["species_parameters_2"]["z"], -1)
                self.assertAlmostEqual(nml["theta_grid_parameters"]["rhoc"], m.rho[i])
                self.assertNotIn("n0", nml["kt_grids_single_parameters"])
                with Dataset(path[:-3]+".out.nc", "w") as f:
                    f.createDimension("theta", 5)
                    th = f.createVariable("theta", "f8", ("theta",))
                    th[:] = np.linspace(-np.pi, np.pi, 5)
                    g = f.createVariable("grho", "f8", ("theta",))
                    g[:] = i+1
                    jac = f.createVariable("jacob", "f8", ("theta",))
                    jac[:] = .25
                    for key,val in (("bref",3.),("aref",.5),("shat",.1*(i+1))):
                        f.createVariable(key,"f8").assignValue(val)
        m.submit_gs2_job = run
        m.get_gs2_eqdsk_geometry(geo)
        self.assertEqual((geo.Btor_eq,geo.a_minor_eq,geo.R_major_eq),(3.,.5,7.))
        np.testing.assert_allclose(geo.grho.profile, [1,2,3])
        np.testing.assert_allclose(geo.area.profile, np.pi**2*np.arange(1,4))
        np.testing.assert_allclose(geo.shat.profile, [.1,.2,.3])
        self.assertEqual(m.inputs["theta_grid_parameters"]["nperiod"],3)
        self.assertEqual(m.inputs["kt_grids_single_parameters"]["n0"],12)

    def mode_nc(self, growth, q=(3.,-1.), name="mode.nc"):
        path = self.root/name
        with Dataset(path,"w") as f:
            for key,num in (("t",5),("theta",3),("ri",2),("species",2),("ky",1)):
                f.createDimension(key,num)
            def var(key,dims,data):
                f.createVariable(key,"f8",dims)[:] = data
            tt=np.arange(5.)
            var("t",("t",),tt)
            var("phi2",("t",),np.exp(2*growth*tt))
            var("phi",("ky","theta","ri"),[[[1.,.25],[.5,.5],[.25,1.]]])
            var("apar",("ky","theta","ri"),np.zeros((1,3,2)))
            var("theta",("theta",),[-1,0,1])
            var("kperp2",("ky","theta"),[[2.,2.,2.]])
            var("jacob",("theta",),[1,1,1])
            var("ky",("ky",),[.5])
            var("es_heat_flux",("t","species"),np.tile(q,(5,1)))
            var("es_part_flux",("t","species"),np.tile((1.,4.),(5,1)))
            var("apar_heat_flux",("t","species"),np.zeros((5,2)))
            var("apar_part_flux",("t","species"),np.zeros((5,2)))
        return path

    def test_electrostatic_and_zero_amp_em_preserve_signed_species_weights(self):
        path = self.mode_nc(.4)
        for apar in (False, True):
            gam,ky,lam,heat,part = self.model.compute_ql_metric(path,apar,False)
            np.testing.assert_allclose([gam,ky,lam],[.4,.5,.2])
            np.testing.assert_allclose(heat,[.3,-.1])
            np.testing.assert_allclose(part,[.1,.4])
            self.assertAlmostEqual(sum(heat),lam)

    def test_damped_or_singular_mode(self):
        path=self.mode_nc(-.4)
        gam,_,lam,heat,part = self.model.compute_ql_metric(path,False,False)
        self.assertLess(gam,0)
        self.assertEqual(lam,0)
        np.testing.assert_array_equal(heat, np.zeros(2))
        np.testing.assert_array_equal(part, np.zeros(2))
        path=self.mode_nc(.4,q=(1.,-1.),name="singular.nc")
        with self.assertRaisesRegex(ValueError,"zero total heat"):
            self.model.compute_ql_metric(path,False,False)

    def test_exact_ballooning_integration_and_zero_shear_flow_limit(self):
        ang = np.array([0., .25, .5, np.pi])
        vals = np.array([[1.-value/4. for value in ang]])
        avg=self.model._ballooning_average
        np.testing.assert_allclose(avg(vals,ang,1.,0.,1.), [1.])
        np.testing.assert_allclose(avg(vals,ang,-1.,.4,1.), [1.-.4/8.])
        np.testing.assert_allclose(avg(vals,ang,0.,.4,1.), [1.-np.pi/8.])

    def test_spectral_aggregation_conserves_model_heat(self):
        m=self.model
        m.N_fluxtubes=1
        m.nky=2
        m.nth0=2
        m.th0=np.array([0.,1.])
        m.gammaE=0.
        m.facnorm=2**(a-1.5)
        m.compute_ql_metric=Mock(side_effect=[
            (.4,.5,.2,np.array([.3,-.1]),np.array([.1,.4])),
            (.5,.5,9,np.array([0.,9]),np.array([0.,0.])),
            (.4,1.,.2,np.array([.3,-.1]),np.array([.1,.4])),
            (.5,1.,9,np.array([0.,9]),np.array([0.,0.])),
        ])
        _,q,_=m.read_gs2_fluxes(["n"]*4,SimpleNamespace(N_species=2),SimpleNamespace(shat=[.4]))
        self.assertAlmostEqual(np.sum(q),m.facnorm*b*.1**a)
        self.assertLess(q[1,0],0.)
