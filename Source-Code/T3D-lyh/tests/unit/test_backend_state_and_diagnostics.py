from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock
import copy

import h5py
from netCDF4 import Dataset
import numpy as np

from t3d.Grid import Grid
from t3d.Profiles import GridProfile, FluxProfile
from t3d.Species import SpeciesDict
from t3d.flux_models.GX import GX_FluxModel
from t3d.flux_models.SFINCS import SFINCS_FluxModel
from t3d.trinity_lib import TrinNormalizations


class TestBackendStateAndDiagnostics(TestCase):
    def setUp(self):
        self.dir = TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.grid = Grid({"grid": {"N_radial": 4}})
        self.time = SimpleNamespace(step_idx=0, iter_idx=0)

    def model_gx(self):
        model = GX_FluxModel.__new__(GX_FluxModel)
        model.__dict__.update(grid=self.grid, time=self.time, rho=self.grid.midpoints,
                              N_fluxtubes=3, out_dir=str(self.root), label="GX", gx_index=0,
                              log=Mock(), info=Mock(), wait=Mock(), zeta_centers=[0.],
                              adiabatic_species=None, build_ReLU=False, no_perturb_temperature=[],
                              dkap_n=.5, dkap_T=.5, _restart_outputs={}, _restart_geometry=None,
                              _gx_pending_geometry=None, nstep_gx=100, electromagnetic=False,
                              tprim_shift=0., collisionality_scaling_factor=1.,
                              target_chie_over_chii=0., qflux_scaling_factor=1.,
                              theta_zeta_fluxes=False, abort_on_nans=True, no_restarts=True)
        model.inputs = {key:{} for key in ("Geometry","Time","Dimensions","Restart","species","Boltzmann","Physics")}
        model.submit_gx_job = lambda tag,path: str(Path(path,tag))
        return model

    def test_gx_probes_full_dependent_mixture_but_writes_kinetic_rows_only(self):
        inputs = {"species":[{"type":"ion_primary", "mass":2, "Z":1, "reference":True},
                             {"type":"adiabatic_link", "mass":3, "Z":1,
                              "density":{"equal_to":"0.5*ion_primary"}},
                             {"type":"ion_3", "mass":5, "Z":1, "density":{"evolve": True}},
                             {"type":"electron", "density":{"evolve":True}}]}
        s=SpeciesDict(inputs,self.grid)
        norms=TrinNormalizations()
        q=s.qneut_species
        self.assertEqual(q.type,"ion_3")
        for item in s.get_species_list():
            item.norms=norms
            item.n_prof=GridProfile({"ion_primary":1.,"adiabatic_link":.5,"ion_3":.5,"electron":2.}[item.type],self.grid)
            item.p_prof=GridProfile(1.,self.grid)
        model=self.model_gx()
        model.build_ReLU=True
        # Removing ion_primary (which is the target of equal_to) from the old
        # SpeciesDict copy failed when computing the charge-constrained probe.
        model.adiabatic_species="ion_primary"
        s.ref_species=s["ion_3"]
        geo=SimpleNamespace(geo_option="vmec",geo_file=str(self.root/"wout.nc"),a_minor=1.,Btor=2.)
        prior=copy.deepcopy(model.inputs)
        model.compute_fluxes_gx(s,geo,norms)
        self.assertEqual(model.inputs,prior)
        self.assertEqual(len(model.outs_jk),1+len(s.n_evolve_keys)+len(s.T_evolve_keys))
        pos=s.n_evolve_keys.index("electron")+1
        point=Path(model.outs_jk[pos][0]+".in").read_text()
        expected,_,_,_=s.get_grads_on_flux_grid(pert_n="electron",abs_step=.5)
        fprim=eval(next(line.split("=",1)[1] for line in point.splitlines() if "fprim" in line),
                   {"__builtins__":{}})
        np.testing.assert_allclose(fprim, expected[[1,2,3],0])
        self.assertNotEqual(fprim[1],0.)
        self.assertEqual(model.kn0_sj.shape,(4,3))

    def test_gx_eqdsk_initializes_once_per_radius_regardless_of_zeta_angles(self):
        model=self.model_gx()
        model.zeta_centers=[.1,.2]
        def run(*args,**kwargs):
            self.assertEqual(model.zeta_centers,[.1])
            stems = [str(self.root/f"s{i}") for i in range(model.N_fluxtubes)]
            for i,stem in enumerate(stems):
                with Dataset(stem+".out.nc","w") as f:
                    g=f.createGroup("Inputs")
                    for key,val in (("B_ref",3.),("a_ref",.4),("grhoavg",1.+i),("surfarea",3.+i)):
                        g.createVariable(key,"f8").assignValue(val)
                    f.createGroup("Geometry").createVariable("rmaj","f8").assignValue(6.)
            return stems
        model.run_gx_fluxtubes=run
        geo=SimpleNamespace()
        model.get_gx_eqdsk_geometry(geo)
        np.testing.assert_allclose(geo.grho.profile,[1,2,3])
        np.testing.assert_allclose(geo.area.profile,[3,4,5])
        self.assertEqual((geo.Btor_eq,geo.a_minor_eq,geo.R_major_eq),(3.,.4,6.))
        self.assertEqual(model.zeta_centers,[.1,.2])

    def test_gx_single_sample_and_missing_or_nonfinite_flux(self):
        m=self.model_gx()
        m.N_fluxtubes=1
        self.assertEqual(m.median_estimator([2.]),2.)
        with self.assertRaisesRegex(ValueError,"unmasked"):
            m.median_estimator(np.ma.array([2.],mask=[True]))
        path=str(self.root/"finite")
        def write(values):
            with Dataset(path+".out.nc","w") as f:
                g=f.createGroup("Diagnostics")
                g.createDimension("t",1)
                g.createDimension("s",1)
                for key,val in zip(("ParticleFlux_st","HeatFlux_st","TurbulentHeating_st"),values):
                    g.createVariable(key,"f8",("t","s"))[:] = [[val]]
        write([1.,2.,-.25])
        cols=m.read_gx_fluxes([path],[SimpleNamespace(type="electron")])
        np.testing.assert_allclose(np.asarray(cols).ravel(),[1.,2.,-.25])
        write([np.inf,2.,-.25])
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            m.read_gx_fluxes([path],[SimpleNamespace(type="electron")])

    def test_sfincs_rejects_truncated_and_unconverged_hdf5(self):
        m=SFINCS_FluxModel.__new__(SFINCS_FluxModel)
        m.__dict__.update(info=Mock(),log=Mock(),label="SFINCS",N_fluxtubes=3,namelist={"physicsparameters":{}})
        item=SimpleNamespace(n=lambda:GridProfile(1.,self.grid), T=lambda:GridProfile(1.,self.grid))
        species=SimpleNamespace(N_species=1,ref_species=item,get_species_list=lambda:[item])
        geo=SimpleNamespace(Btor=1.,grho=FluxProfile(1.,self.grid))
        norms=SimpleNamespace(rho_star=1.,a_meters=1.)
        path=self.root/"partial.h5"
        def write(finished=None, converged=None, nan=False):
            with h5py.File(path,"w") as f:
                if finished is not None:
                    f["finished"]=finished
                if converged is not None:
                    f["didNonlinearCalculationConverge"]=converged
                f["includePhi1"]=1
                f["particleFlux_vd_rN"]=[[2.,3.]]
                f["heatFlux_vd_rN"]=[[1., np.nan if nan else 5.]]
                f["NIterations"]=2
        write()
        with self.assertRaisesRegex(RuntimeError,"not marked finished"):
            m.read_sfincs_fluxes([path]*3,species,geo,norms)
        write(1,0)
        with self.assertRaisesRegex(RuntimeError,"did not converge"):
            m.read_sfincs_fluxes([path]*3,species,geo,norms)
        write(1,1,nan=True)
        with self.assertRaisesRegex(ValueError,"Nonfinite"):
            m.read_sfincs_fluxes([path]*3,species,geo,norms)
        write(1,1)
        p,q=m.read_sfincs_fluxes([path]*3,species,geo,norms)
        np.testing.assert_allclose(p,3*np.sqrt(2))
        np.testing.assert_allclose(q,5*np.sqrt(8))
