"""Independent physics falsification after the round-4 fixes."""
import itertools
import unittest

import numpy as np
from scipy.integrate import quad

from t3d.Profiles import GridProfile
from t3d.Species import SpeciesDict
from tests.unit import test_source_physics_io


class TestPostRepairPhysics(unittest.TestCase):
    def setUp(self):
        self.fixture = test_source_physics_io.TestSourcePhysicsAndOutput()
        self.fixture.setUp()

    def test_pairwise_collisions_and_alpha_partition_across_positive_states(self):
        rng=np.random.default_rng(10301)
        f=self.fixture
        cfg=[{"type":"deuterium"},{"type":"tritium"},{"type":"carbon","density":{"evolve":False}},
             {"type":"electron","qneut_species":True}]
        sp=SpeciesDict({"species":cfg},f.grid)
        # Use individually positive profiles but impose charge balance directly
        # so the test does not assume the new initialization logic is correct.
        phy=f.physics(alpha_efficiency=.72)
        for _ in range(24):
            ions=(sp["deuterium"],sp["tritium"],sp["carbon"])
            for s in ions:
                n=rng.uniform(.08,1.6,f.grid.N_radial)
                if s.type=="carbon":
                    n*=.02
                s.n_prof=GridProfile(n,f.grid)
                s.p_prof=s.n_prof*GridProfile(rng.uniform(.4,24.,f.grid.N_radial),f.grid)
                s.norms=f.norms
            sp.electron.n_prof=sum((s.Z*s.n() for s in ions),GridProfile(0.,f.grid))
            sp.electron.p_prof=sp.electron.n()*GridProfile(rng.uniform(1.,30.,f.grid.N_radial),f.grid)
            sp.electron.norms=f.norms
            for i,j in itertools.combinations(sp.get_species_list(),2):
                diff=j.T()-i.T()
                gi=i.n()*i.nu_equil(j)*diff
                gj=j.n()*j.nu_equil(i)*diff
                np.testing.assert_allclose(gi.profile,gj.profile,rtol=4e-15,atol=1e-13)
                self.assertTrue(np.all(gi.profile*diff.profile >= 0))
            species=list(sp.get_species_list())
            deposited=phy.compute_total_alpha_heating(sp).profile
            by_species=np.stack([phy.compute_alpha_heating(s,sp).profile for s in species])
            self.assertTrue(np.all(by_species>0))
            np.testing.assert_allclose(deposited, f.norms.P_ref_MWm3*by_species.sum(axis=0),
                                       rtol=2e-15,atol=0.)
            Ec = 4*14.8*(sum(s.n().profile*s.Z**2/s.mass for s in ions)/sp.electron.n().profile)**(2./3.)
            x = 3500./(Ec*sp.electron.T().profile)
            # Ion drag as a fraction of total drag, averaged over slowing energy.
            analytic=np.array([quad(lambda z: 1./(1.+(xi*z)**1.5),0.,1.,epsabs=1e-12)[0] for xi in x])
            np.testing.assert_allclose(phy.calc_alpha_ion_heating_fraction(sp),analytic,rtol=5e-11)

    def test_resolved_neutral_roots_agree_under_mixed_link_permutations(self):
        f=self.fixture
        species=[
            {"type":"electron", "density":{"core":2.1,"edge":1.5}},
            {"type":"deuterium", "density":{"equal_to":"0.65 * electron"}},
            {"type":"tritium", "density":{"core":.25,"edge":.15}},
            {"type":"helium","density":{"equal_to":"0.1 * tritium"}},
        ]
        for order in itertools.permutations(species):
            s=SpeciesDict({"species":order},f.grid)
            s.init_profiles_and_sources(f.geo,f.norms)
            np.testing.assert_allclose(s["tritium"].n().profile,s.electron.n().profile*.35/1.2)
            np.testing.assert_allclose(s["helium"].n().profile,.1*s["tritium"].n().profile)
            np.testing.assert_allclose(sum(i.Z*i.n().profile for i in s.get_species_list()),0.,atol=1e-15)
