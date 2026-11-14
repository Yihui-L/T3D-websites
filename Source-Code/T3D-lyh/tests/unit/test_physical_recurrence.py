import unittest

import numpy as np

from t3d.Profiles import GridProfile
from t3d.Species import SpeciesDict
from tests.unit import test_source_physics_io


class TestPhysicalRecurrence(unittest.TestCase):
    def setUp(self):
        self.fixture = test_source_physics_io.TestSourcePhysicsAndOutput()
        self.fixture.setUp()

    def test_two_maxwellian_reactivity_uses_relative_speed_distribution(self):
        f = self.fixture
        species = f.species()
        physics = f.physics()
        d, t = species["deuterium"], species["tritium"]
        equal = physics.compute_alpha_birth_power(species).profile
        t.p_prof *= 2
        unequal = physics.compute_alpha_birth_power(species).profile
        self.assertTrue(np.all(unequal > equal))
        # Reduced-mass times the variance of relative Maxwellian velocities.
        mu = 1/(1/d.mass + 1/t.mass)
        d.p_prof *= mu*(1/d.mass + 2/t.mass)
        t.p_prof = t.n()*d.T()
        np.testing.assert_allclose(physics.compute_alpha_birth_power(species).profile, unequal,
                                   rtol=2e-15, atol=0)
        with self.assertRaisesRegex(ValueError, "f_helium"):
            f.physics(f_helium=1.1)
        with self.assertRaisesRegex(ValueError, "f_tritium"):
            f.physics(f_tritium=-.1)

    def test_mixed_background_temperature_and_particle_charge(self):
        f = self.fixture
        configs = [
            dict(type="electron", density={"core":3,"edge":2}),
            dict(type="deuterium", density={"init_to":"0.9*electron"}),
            dict(type="helium", density={"init_to":"0.05*electron","evolve":False}),
            dict(type="carbon", Zeff=1.4)]
        sp = SpeciesDict({"species":configs}, f.grid)
        sp.init_profiles_and_sources(f.geo,f.norms)
        ne = sp.electron.n().profile
        background = (6 - 1.4)/(6 - 1.1)
        np.testing.assert_allclose(sp.Zeff().profile,1.4)
        np.testing.assert_allclose(sp["carbon"].n().profile/ne,(1-background)/6)
        np.testing.assert_allclose(sp["deuterium"].n().profile/ne,.9*background)
        np.testing.assert_allclose(sp["helium"].n().profile/ne,.05*background)
        configs[-1]["Zeff"] = 1.05
        with self.assertRaisesRegex(ValueError,"between"):
            SpeciesDict({"species":configs},f.grid).init_profiles_and_sources(f.geo,f.norms)

    def test_sudo_is_explicit_and_survives_profile_unpack(self):
        f = self.fixture
        sp = f.species(names=("hydrogen","electron"))
        phy = f.physics(alpha_heating=False, scale_density_to=".5e0 * sudo_line")
        before = [s.T().profile.copy() for s in sp.get_species_list()]
        old_profiles = sp.get_vec_from_profs()
        with self.assertRaisesRegex(ValueError,"positive prescribed"):
            phy.scale_to_sudo_density(sp)
        np.testing.assert_array_equal(sp.get_vec_from_profs(),old_profiles)
        C = f.geo.B0/(16*(f.geo.AspectRatio*f.geo.a_minor)*f.geo.a_minor**2)
        target=2.
        P=target**2/(.5**2*C)
        sp.Paux = GridProfile(P/f.norms.P_ref_MWm3/f.geo.volume_integrate(GridProfile(1.,f.grid),active=True),f.grid)
        phy.compute_sources(sp,f.geo,f.norms)
        np.testing.assert_array_equal(sp.get_vec_from_profs(),old_profiles)
        self.assertTrue(phy.scale_to_sudo_density(sp))
        sp.get_profs_from_vec(sp.get_vec_from_profs())
        np.testing.assert_allclose(sp["electron"].n().profile,target)
        np.testing.assert_allclose(sp["hydrogen"].n().profile,target)
        for original,s in zip(before,sp.get_species_list()):
            np.testing.assert_allclose(s.T().profile,original)
        self.assertFalse(phy.scale_to_sudo_density(sp))
        for invalid in ("sudo_bad", "0 * sudo_line", "sudo_line junk"):
            with self.assertRaisesRegex(ValueError,"[Ss]udo"):
                f.physics(scale_density_to=invalid)

    def test_edge_caps_and_rebuilds_all_constraints(self):
        f = self.fixture
        inputs={"species":[
            {"type":"electron","density":{"core":5,"edge":1,"evolve":False},
                               "temperature":{"core":6,"edge":2,"evolve":False}},
            {"type":"deuterium","density":{"core":3,"edge":.5},
                                "temperature":{"equal_to":"0.5 * electron"}},
            {"type":"tritium","temperature":{"core":7,"edge":3,"evolve":False}},
        ]}
        sp=SpeciesDict(inputs,f.grid)
        sp.init_profiles_and_sources(f.geo,f.norms)
        settings=f.physics()
        settings.edge_p_ratio = settings.edge_n_ratio = 1.01
        settings.edge_p_multiplier = settings.edge_n_multiplier = 5.
        settings.edge_n_max = 1.4
        settings.edge_T_max = 2.3
        before = sp.get_vec_from_profs()
        with self.assertRaisesRegex(ValueError, 'nonphysical'):
            sp.update_edge(settings)
        np.testing.assert_array_equal(sp.get_vec_from_profs(), before)
        settings.edge_n_multiplier = 1.5
        interiors=[(s.n().profile[:-1].copy(),s.p().profile[:-1].copy())
                   for s in sp.get_species_list()]
        self.assertTrue(sp.update_edge(settings))
        sp.get_profs_from_vec(sp.get_vec_from_profs())
        for (n,p),s in zip(interiors,sp.get_species_list()):
            np.testing.assert_allclose(s.n().profile[:-1],n)
            np.testing.assert_allclose(s.p().profile[:-1],p)
        self.assertAlmostEqual(sp["electron"].n()[-1],1.4)
        self.assertAlmostEqual(sp["electron"].T()[-1],2.3)
        self.assertAlmostEqual(sp["deuterium"].T()[-1],1.15)
        self.assertAlmostEqual(sp["tritium"].T()[-1],3)
        np.testing.assert_allclose(sp["electron"].n().profile,sp["deuterium"].n().profile+sp["tritium"].n().profile)

    def test_manually_tabulated_density_and_temperature_without_numpy_matrix(self):
        f = self.fixture
        cfg={"species":[
            {"type":"electron",
             "density":{"shape":"manual","profstr":"0, 1; 2, 1"},
             "temperature":{"shape":"manual","profstr":"0 1; 5 2"}},
            {"type":"hydrogen"}]}
        sp=SpeciesDict(cfg,f.grid)
        sp.init_profiles_and_sources(f.geo,f.norms)
        rho=f.grid.rho/f.grid.rho_edge
        np.testing.assert_allclose(sp.electron.n().profile,2-rho)
        np.testing.assert_allclose(sp.electron.T().profile,5-3*rho)
        cfg["species"][0]["temperature"]["profstr"] = "0 0.8; 5 2"
        with self.assertRaisesRegex(ValueError,"covering the grid"):
            SpeciesDict(cfg,f.grid).init_profiles_and_sources(f.geo,f.norms)

    def test_disabled_source_labels_do_not_leak_into_power_diagnostic(self):
        f = self.fixture
        physics=f.physics(bremsstrahlung=True)
        sp=f.species()
        physics.compute_sources(sp,f.geo,f.norms)
        self.assertTrue(np.all(sp.electron.Sp_labeled["rad"].profile < 0))
        physics.bremsstrahlung=False
        physics.alpha_heating=False
        physics.compute_sources(sp,f.geo,f.norms)
        np.testing.assert_array_equal(sp.electron.Sp_labeled["rad"].profile,0.)
        self.assertEqual(physics.compute_Pheat(sp),0.)
