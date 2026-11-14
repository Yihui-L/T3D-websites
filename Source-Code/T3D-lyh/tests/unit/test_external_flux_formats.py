from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest import TestCase
from unittest.mock import Mock

import f90nml
import numpy as np

from t3d.Grid import Grid
from t3d.Profiles import GridProfile, FluxProfile
from t3d.flux_models.GX import GX_FluxModel, _checked_gx_array, _gx_periodic_surface_average, tomllib
from t3d.flux_models.KNOSOS import KNOSOS_FluxModel
from t3d.flux_models.SFINCS import SFINCS_FluxModel
from t3d.flux_models.neo import neo_FluxModel


class TestExternalFluxFormats(TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name)
        self.grid = Grid({"grid": {"N_radial": 3}})
        self.rho = self.grid.midpoints

    def test_gx_toml_preserves_quoted_newline_and_nested_template_values(self):
        src = self.path / "template.in"
        dst = self.path / "out.in"
        src.write_text('name = "test=a#b" # this is not part of the value\n'
                       '    [Geometry]\n  geo_file = "x # y=z.nc"\n'
                       '    [Diagnostics.options]\n  enable = false\n'
                       '    [Dimensions]\n  nspecies = 2\n'
                       '    [species]\n  z = [1.0, -1.0]\n')
        gx = GX_FluxModel.__new__(GX_FluxModel)
        gx.__dict__.update(log=Mock(), info=Mock())
        gx.read_input(src)
        self.assertEqual(gx.inputs["Geometry"]["geo_file"], "x # y=z.nc")
        gx.inputs["Geometry"]["beta"] = np.float64(2e-4)
        gx.inputs["Geometry"]["path"] = 'slash\\quote"'
        gx.write_input(dst)
        self.assertEqual(tomllib.loads(dst.read_text()), gx.inputs)

    def test_gx_periodic_average_has_no_spurious_zero_or_constant_flux(self):
        zeta, theta = np.meshgrid([-1.25,-.02,.81,2.25], [-2.51,-1.1,.04,1.8])
        zeta, theta = zeta.ravel(), theta.ravel()
        weight = 2. + np.cos(theta)
        fluxes = [0.*weight, -3.*weight, 5.*weight]
        avg = _gx_periodic_surface_average(zeta, theta, weight, fluxes, nfp=1)
        np.testing.assert_allclose(avg, [0.,-3.,5.], atol=1e-12)
        with self.assertRaisesRegex(ValueError, "unmasked"):
            _checked_gx_array(np.ma.array([1.,0.],mask=[True,False]), "heat")
        m = GX_FluxModel.__new__(GX_FluxModel)
        m.__dict__.update(info=Mock(), log=Mock(), theta_zeta_fluxes=True, zeta_centers=[0.])
        with self.assertRaisesRegex(ValueError, "multiple field lines"):
            m.read_gx_fluxes([], [])

    def test_sfincs_typed_input_preserves_physical_charges_and_resets_indices(self):
        src = self.path / "original.namelist"
        src.write_text("&speciesparameters\n Zs(2:4)=1, -1, 6 ! placeholder\n/\n"
                       "&geometryparameters\n equilibriumFile='eq! with space=a.nc'\n/\n")
        geom = NS(geo_option="vmec", geo_file="unused.nc")
        mass = np.array([2., 1./1836., 12.])
        charges = np.array([1., -1., 6.])
        for ref in (0, 1):
            with self.subTest(reference=ref):
                m = SFINCS_FluxModel.__new__(SFINCS_FluxModel)
                m.__dict__.update(info=Mock(), log=Mock(), grid=self.grid, rho=self.rho)
                m.read_sfincs_input(src)
                m.run_sfincs = Mock(return_value=[])
                species = NS(get_charges=lambda normalize: charges/charges[ref] if normalize else charges,
                             get_masses=lambda normalize: mass/mass[ref] if normalize else mass,
                             n_evolve_keys=[], T_evolve_keys=[],
                             get_grads_on_flux_grid=lambda **kw: (None,None,None,None))
                m.compute_fluxes(species, geom, NS(a_meters=1., rho_star=.01, vtfac=1.))
                dst = self.path / f"sfincs-{ref}.namelist"
                m.write_sfincs_input(dst, m.namelist)
                read = f90nml.read(dst)
                np.testing.assert_array_equal(read["speciesparameters"]["zs"], charges)
                np.testing.assert_allclose(read["speciesparameters"]["mhats"], mass/mass[ref])
                self.assertEqual(read["geometryparameters"]["equilibriumfile"], "eq! with space=a.nc")
                self.assertEqual(read["geometryparameters"]["inputradialcoordinateforgradients"], 3)
                self.assertNotIn("zs", read["speciesparameters"].start_index)
                with self.assertRaisesRegex(ValueError, "requires VMEC"):
                    m.compute_fluxes(species, NS(geo_option="miller"), NS())

    def test_neo_locally_normalized_input_charges_and_gyroradius(self):
        src = self.path / "template.neo"
        src.write_text("# initial notes\n N_SPECIES = 7 # overwritten\n SIM_MODEL=2\n")
        model = neo_FluxModel.__new__(neo_FluxModel)
        model.__dict__.update(rho=self.rho, N_fluxtubes=2, out_dir=str(self.path), qmax=0,
                              bunit_over_b0=np.ones(2)*.7, time=NS(step_idx=0, iter_idx=0))
        model.read_input(src)
        bare = np.ones(2)
        geo = NS(a_minor=1., Btor=2., qsf=bare, shat=bare, rmajor_prime=bare,
                 kappa=bare, s_kappa=bare, tri=bare, s_delta=bare, rmaj=bare)
        temperature = np.array([4., 9.])
        norm = NS(rho_star=.05, vtfac=2.)
        charge = np.array([1., -1., 6.])
        mass = np.array([2., 1./1836., 12.])
        for ref in (0, 1):
            with self.subTest(reference=ref):
                species = NS(ref_species=NS(T=lambda: NS(toFluxProfile=lambda: NS(profile=temperature))),
                             N_species=3, get_charges=lambda normalize: charge/charge[ref] if normalize else charge,
                             get_masses=lambda normalize: mass/mass[ref] if normalize else mass,
                             get_profiles_on_flux_grid=lambda **kw: (np.ones((3,2)),)*3)
                dirs = model.run_neo(self.rho, np.ones((3,2)), np.ones((3,2)), species, geo, ref, norm)
                for i, path in enumerate(dirs):
                    data = dict(line.split("=") for line in (Path(path)/"input.neo").read_text().splitlines())
                    self.assertEqual(data["N_SPECIES"], "3")
                    np.testing.assert_array_equal([float(data[f"Z_{j}"]) for j in (1,2,3)], charge)
                    expected = .05 * np.sqrt(temperature[i]/2) / (2*.7)
                    self.assertAlmostEqual(float(data["RHO_STAR"]), expected)
                    self.assertAlmostEqual(float(data["NU_1"]), np.sqrt(2.))

    def test_neo_parser_conserves_signed_flux_and_rejects_malformed_data(self):
        model = neo_FluxModel.__new__(neo_FluxModel)
        model.__dict__.update(rho=self.rho, N_fluxtubes=2, qmax=0)
        norm = NS(rho_star=.02, vtfac=2.)
        geo = NS(Btor=3.)
        n = GridProfile([2.,3.,5.],self.grid)
        p = GridProfile([7.,4.,2.],self.grid)
        ref = NS(n=lambda:n,p=lambda:p,T=lambda:p/n)
        ns = n.toFluxProfile().profile
        species = NS(N_species=1,ref_species=ref,get_species_list=lambda:[ref],
                     get_densities_on_flux_grid=lambda:np.array([ns]))
        files = [self.path / f"face-{i}" for i in range(2)]
        for i,f in enumerate(files):
            f.write_text(f"{self.rho[i]} 0 0 0 0 -3D0 5 0 0 0 0 0 0\n")
        gamma, q, _ = model.read_neo_fluxes(files,species,geo,norm)
        norm_t = (p/n).toFluxProfile().profile
        scale = geo.Btor**2 / norm.rho_star**2 * ns * np.sqrt(norm_t/norm.vtfac)
        np.testing.assert_allclose(gamma[0]*(p**1.5/n**.5).toFluxProfile().profile, -3*scale)
        np.testing.assert_allclose(q[0]*p.toFluxProfile().profile**2.5/ns**1.5, 5*scale*norm_t)
        with self.assertRaisesRegex(ValueError, "one output"):
            model.read_neo_fluxes(files[:1],species,geo,norm)
        with self.assertRaisesRegex(ValueError, "radius"):
            model.read_neo_fluxes(files[::-1],species,geo,norm)
        files[0].write_text(files[0].read_text().replace("5 0", "nan 0"))
        with self.assertRaisesRegex(ValueError, "Invalid NEO"):
            model.read_neo_fluxes(files,species,geo,norm)

    def test_knosos_predictive_output_matches_impurity_order_and_face_radius(self):
        m = KNOSOS_FluxModel.__new__(KNOSOS_FluxModel)
        m.__dict__.update(info=Mock(), log=Mock(), rho=self.rho, N_fluxtubes=2, label="KNOSOS")
        ref = NS(n=lambda:GridProfile(1.,self.grid),T=lambda:GridProfile(1.,self.grid))
        bulk = NS(type="bulk")
        carbon = NS(type="carbon")
        species = NS(N_species=3,ref_species=ref,bulk_ion=bulk,other_ions_dict={"carbon": carbon},
                     get_species_list=lambda:[NS(type="electron"),carbon,bulk])
        ns = NS(P_ref_MWm3=1e-6, Sn_ref_SI20=1e-20)
        geo = NS(Btor=1.,a_minor=1.)
        paths = []
        for j,r in enumerate(self.rho):
            path = self.path / f"knosos-{j}"
            path.write_text(f"rho Er Q_i Q_e Q_Z Gam_i Gam_e Gam_Z\n{r:.5e} 0 {1+j} 10 100 2 20 200\n")
            paths.append(path)
        p, q = m.read_knosos_fluxes(paths[::-1],species,geo,ns)
        np.testing.assert_allclose(q, [[10,10],[100,100],[1,2]])
        np.testing.assert_allclose(p, [[20,20],[200,200],[2,2]])
        with self.assertRaisesRegex(ValueError, "radius"):
            m.read_knosos_fluxes(paths[:1]*2,species,geo,ns)
        paths[0].write_text("header\n0 1 2\n")
        with self.assertRaisesRegex(ValueError, "columns"):
            m.read_knosos_fluxes(paths,species,geo,ns)
