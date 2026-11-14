import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch
import unittest

import numpy as np

from t3d.Import import TranspReader, Import
from t3d.Grid import Grid
from t3d.TrinityIO import Input, tomllib


class TestInputRoundTrip(unittest.TestCase):
    def test_toml_roundtrip_mixed_order_arrays_and_strings(self):
        raw = """
debug = true
title = "shot's file"
rho = [0.1, 0.3, 0.7]
labels = ["a", "b"]
[log]
output_netcdf = true
[nested]
empty = {}
when = 2024-09-01T12:34:56Z
[[species]]
type = "electron"
temperature = {evolve = true, label = "edge's value", fit = [1.0, 0.2]}
[[species]]
type = "hydrogen"
"""
        data = Input(raw)
        with TemporaryDirectory() as directory:
            file = Path(directory)/"case.in"
            data.write(file)
            self.assertEqual(Input(file).input_dict, data.input_dict)
            backed = file.read_bytes()
            data.input_dict["debug"] = False
            data.write(file, overwrite=False)
            self.assertEqual(file.read_bytes(), backed)
            data.write(file, overwrite=True)
            self.assertFalse(Input(file).input_dict["debug"])
        printed = io.StringIO()
        with redirect_stdout(printed):
            data.pretty_print("log")
        self.assertEqual(tomllib.loads(printed.getvalue()), {"log":{"output_netcdf":True}})


class TestTRANSPSourceIdentity(unittest.TestCase):
    def fixture(self):
        r = TranspReader.__new__(TranspReader)
        r.flux_label = "torflux"
        rho = np.array([0., .5, 1.])
        r.data = {"NM1A":"3.0", "NM1Z":"1.0","NM2A":"3.0", "NM2Z":"2.0",
                  "NE":1e20*np.array([10.,10.,10.]),
                  "NM1":1e20*np.array([6.,6.,6.]), "NM2":1e20*np.array([2.,2.,2.]),
                  "SNBII":1e20*np.array([2.,3.,4.]), "DNER":1e20*np.array([.1,.2,.3]),
                  "QNBII":1e6*np.array([4.,6.,8.]), "DWIR":1e6*np.array([1.,2.,3.])}
        r.meta = {key:SimpleNamespace(rho=rho) for key,val in r.data.items() if isinstance(val,np.ndarray)}
        return r

    def test_transp_species_match_charge_and_missing_is_not_electron(self):
        r=self.fixture()
        self.assertEqual(r.find_ion_species_index(3.,2.),2)
        with self.assertRaisesRegex(ValueError,"ambiguous"):
            r.find_ion_species_index(3.)
        np.testing.assert_allclose(r.get_density(species_mass=3.,species_charge=2.)[1],2.)
        with self.assertRaisesRegex(ValueError,"matching"):
            r.get_density(species_mass=4.,species_charge=2.)

    def test_transp_units_and_missing_channels(self):
        r=self.fixture()
        coords,n=r.get_density_source(keys="SNBII")
        np.testing.assert_allclose(coords,[0,.5,1])
        np.testing.assert_allclose(n,[2,3,4])
        coords,p=r.get_pressure_source(keys=["QNBII","DWIR"],norms=SimpleNamespace(P_ref_MWm3=2))
        np.testing.assert_allclose(p,[1.5,2.,2.5])
        np.testing.assert_allclose(r.get_pressure_source(species_mass=3.,species_charge=2.)[1],0.)
        np.testing.assert_allclose(r.get_density_source(species_mass=3.,species_charge=2.)[1],0.)
        with self.assertRaisesRegex(ValueError,"not found"):
            r.get_density_source(keys=["NOT_A_CHANNEL"])
        with self.assertRaisesRegex(ValueError,"radial grid"):
            r.meta["DWIR"].rho=np.array([0.,.6,1.])
            r.get_pressure_source(keys=["QNBII","DWIR"])

    def test_profile_import_rejects_unsorted_invalid_coordinates(self):
        grid=Grid({"grid":{"N_radial":3}})
        imp=Import({},grid)
        imp.imported=SimpleNamespace(get_density=lambda **kw: ([0., 1., .4], [1.,2.,3.]))
        with self.assertRaisesRegex(ValueError,"increasing"):
            imp.get_density()


class TestEquilibriumAndEntryPoint(unittest.TestCase):
    def test_native_desc_volume_is_interpolated_directly(self):
        from t3d.Geometry import Geometry
        from scipy.interpolate import interp1d
        axis = np.linspace(0, 1, 6)
        desc = SimpleNamespace(rho=axis, area=axis+2*axis**3, dVdrho=axis+axis**2,
                               aminor=.6, Rmajor=4., B_ref=2., B0=2., iota23=.4,
                               out_vmec=None)
        desc.grad_rho = np.r_[0., desc.area[1:]/desc.dVdrho[1:]]
        cfg = {"grid":{"N_radial":4,"rho_edge":.83},
               "geometry":{"geo_option":"desc","geo_file":"native.h5"}}
        g = Grid(cfg)
        with patch("t3d.Geometry.DescHandler",return_value=desc):
            geo = Geometry(cfg,g)
        np.testing.assert_allclose(geo.area_grid.profile/geo.grho_grid.profile*geo.a_minor**3,
                                   interp1d(axis,desc.dVdrho)(g.rho),rtol=1e-14)
        cfg["grid"]["flux_label"]="rminor"
        with self.assertRaisesRegex(ValueError,"DESC requires rho"):
            Geometry(cfg,Grid(cfg))

    def test_cli_source_checkout_and_invalid_filename(self):
        from importlib import metadata
        from t3d import __main__ as cli
        with patch.object(cli.log,"set_handlers") as setup, patch.object(cli.log,"finalize") as finish, patch(
                "t3d.__main__.metadata.version",side_effect=metadata.PackageNotFoundError), patch(
                "t3d.__main__.TrinityEngine") as engine, redirect_stdout(io.StringIO()):
            with patch("sys.argv",["t3d","official.in"]):
                cli.main()
            engine.assert_called_once_with("official.in")
            engine.return_value.evolve_profiles.assert_called_once()
            finish.assert_called_once()
            setup.reset_mock()
            with patch("sys.argv",["t3d","not-an-input.txt"]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    cli.main()
            self.assertNotEqual(cm.exception.code,0)
            setup.assert_not_called()


class TestIndependentIOSweep(unittest.TestCase):
    def test_snapshots_geometry_metric_for_each_frame(self):
        from netCDF4 import Dataset
        from tests.unit.test_source_physics_io import TestSourcePhysicsAndOutput
        h=TestSourcePhysicsAndOutput()
        h.setUp()
        species,physics=h.species(),h.physics()
        from t3d.Time import Time
        clock=Time(h.inputs)
        logger=h.logger(species,physics,clock)
        first = h.geo.G_fac.profile.copy()
        logger.save(species,SimpleNamespace(),clock)
        h.geo.G_fac.profile *= 2
        clock.step_idx = 1
        logger.save(species,SimpleNamespace(),clock,final=True)
        with TemporaryDirectory() as directory:
            fixture=Path(__file__).resolve().parents[1]/"regression"/"test-w7x-relu.in"
            output=Path(directory)/"test.nc"
            logger.write_netCDF4(str(fixture),output)
            with Dataset(output) as ds:
                np.testing.assert_allclose(ds["time"]["t_G_fac"][:],[first,2*first])
                np.testing.assert_allclose(ds["geo"]["G_fac"][:],first)

    def test_miller_restart_does_not_restore_old_runtime(self):
        from t3d.Import import TrinityReader
        from t3d.Profiles import FluxProfile
        g=Grid({"grid":{"N_radial":4,"flux_label":"rminor"}})
        old=Grid({"grid":{"N_radial":4,"flux_label":"rminor"}})
        old_model=object()
        new_model=object()
        stored=SimpleNamespace(geo_option="miller",grid=old,flux_models=old_model,
                               geo_outputs="old",area=FluxProfile(2.,old),Btor=2.,
                               a_minor=1.,AspectRatio=3.,beta_prime_const=False)
        target=SimpleNamespace(grid=g,flux_models=new_model,geo_outputs="new",
                               force_beta_prime_species=True,beta_prime_const=True)
        reader=TrinityReader.__new__(TrinityReader)
        reader.output_format=reader.PICKLE
        reader.rho=g.rho
        reader.frame=0
        reader.data={"geo_obj":stored,"species":[SimpleNamespace(beta_prime=[7]),SimpleNamespace(beta_prime=[9])]}
        reader.get_geometry(target,g)
        self.assertIs(target.flux_models,new_model)
        self.assertEqual(target.geo_outputs,"new")
        self.assertTrue(target.beta_prime_const)
        self.assertIs(target.area.grid,g)
        np.testing.assert_array_equal(target.beta_prime,[7])

    def test_jetto_masked_value_is_not_read_as_physics(self):
        from netCDF4 import Dataset
        with TemporaryDirectory() as directory:
            p=Path(directory)/"jetto.nc"
            with Dataset(p,"w") as ds:
                ds.createDimension("t",1)
                ds.createDimension("r",3)
                ds.createVariable("X","f8",("t","r"))[0]=[0,.5,1]
                ds.createVariable("TE","f8",("t","r"),fill_value=1000)[0]=[3000,1000,2000]
            g=Grid({"grid":{"N_radial":3}})
            imp=Import({"import":{"type":"jetto","file":str(p)}},g)
            with self.assertRaisesRegex(ValueError,"nonfinite values"):
                imp.get_temperature(key="TE")


class TestModelRegistry(unittest.TestCase):
    def test_flux_singleton_uses_same_label_and_count_as_array(self):
        from t3d.FluxModels import FluxModelDict
        from t3d.Time import Time
        grid=Grid({"grid":{"N_radial":3}})
        config={"model":"ReLU","label":"my_relu"}
        for forms in (config,[config]):
            registry=FluxModelDict({"model":forms},grid,Time({}))
            self.assertEqual(registry.N_models,1)
            self.assertEqual(registry["my_relu"].label,"my_relu")
        self.assertEqual(FluxModelDict({"model":[]},grid,Time({})).N_models,0)
        with self.assertRaisesRegex(ValueError,"Unknown flux model"):
            FluxModelDict({"model":{"model":"made-up"}},grid,Time({}))


class TestVolumeInterpolation(unittest.TestCase):
    def test_constant_circular_integral_exact_on_nodes_and_faces(self):
        from t3d.Geometry import Geometry
        from t3d.Profiles import FluxProfile,GridProfile
        cfg={"grid":{"N_radial":5,"rho_edge":.81, "flux_label":"rminor"},
             "geometry":{"AspectRatio":4},"rescale":{"a_minor":.8}}
        g=Grid(cfg)
        geo=Geometry(cfg,g)
        for cls in (GridProfile,FluxProfile):
            f=cls(2.,g)
            expected=2*geo.volume*f.rho**2
            np.testing.assert_allclose(geo.volume_integrate(f,profile=True,interpolate=True).profile,expected)
            self.assertAlmostEqual(geo.volume_integrate(f,interpolate=True),expected[-1])
        alien=Grid({"grid":{"N_radial":5,"rho_edge":.9}})
        with self.assertRaisesRegex(ValueError, "different radial grids"):
            geo.volume_integrate(GridProfile(1.,alien))
