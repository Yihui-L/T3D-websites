"""Manufactured composition and external source edge cases."""
import itertools
import unittest

import numpy as np

from t3d.Physics import Physics
from t3d.Species import SpeciesDict
from tests.unit.test_source_physics_io import TestSourcePhysicsAndOutput as PhysicalFixture


class TestPhysicsRound4(unittest.TestCase):
    def setUp(self):
        self.f = PhysicalFixture()
        self.f.setUp()

    def initialize(self, configs):
        s = SpeciesDict({"species": configs}, self.f.grid)
        s.init_profiles_and_sources(self.f.geo, self.f.norms)
        return s

    def test_initial_density_and_temperature_dependency_graph_is_order_independent(self):
        definitions = (
            {"type":"electron", "qneut_species":True},
            {"type":"tritium", "density":{"init_to":"4.e-1*deuterium"},
             "temperature":{"init_to":"2 * deuterium"}},
            {"type":"deuterium", "density":{"init_to":" .5 * hydrogen "},
             "temperature":{"equal_to":"3 * hydrogen"}},
            {"type":"hydrogen","density":{"core":1., "edge":1.},
             "temperature":{"core":1., "edge":1.}},
        )
        for sequence in itertools.permutations(definitions):
            s = self.initialize(sequence)
            np.testing.assert_allclose(s["deuterium"].n().profile,.5)
            np.testing.assert_allclose(s["tritium"].n().profile,.2)
            np.testing.assert_allclose(s["tritium"].T().profile,6.)
            np.testing.assert_allclose(s.electron.n().profile,1.7)
            v=s.get_vec_from_profs()
            v[s.n_evolve_keys.index("deuterium")*s.N_radial]+=.1
            s.get_profs_from_vec(v)
            self.assertAlmostEqual(s["tritium"].n()[0],.2)  # init_to releases that constraint
            self.assertAlmostEqual(s.electron.n()[0],1.8)

    def test_density_link_which_already_imposes_quasineutrality_keeps_free_density(self):
        h={"type":"hydrogen","density":{"equal_to":"electron"}}
        e={"type":"electron","density":{"core":1., "edge":1.}}
        s=self.initialize([h,e])
        self.assertIsNone(s.qneut_species)
        self.assertEqual(s.n_evolve_keys,["electron"])
        v=s.get_vec_from_profs()
        v[0]+=.1
        s.get_profs_from_vec(v)
        np.testing.assert_allclose(s["hydrogen"].n().profile,s.electron.n().profile)
        self.assertAlmostEqual(s.electron.n()[0],1.1)
        kn,_,_,_=s.get_grads_on_flux_grid(pert_n="electron")
        np.testing.assert_allclose(kn[0],kn[1])

    def test_zero_charge_root_does_not_mask_other_charge_constraint(self):
        config=[
            {"type":"electron","density":{"core":1., "edge":1., "evolve":False}},
            {"type":"hydrogen","density":{"equal_to":"electron"}},
            {"type":"deuterium"},
        ]
        with self.assertRaisesRegex(ValueError,"positive"):
            self.initialize(config)  # D cannot be positive because H exhausts the charge budget
        with self.assertRaisesRegex(ValueError,"zero independent charge"):
            self.initialize([{"type":"electron","qneut_species":True},config[1]])
        # The published e/D/T/C example uses ion charge 0.495+0.495+6*0.01=1.05
        # per electron and cannot be neutralized without changing its input.
        with self.assertRaisesRegex(ValueError,"Inconsistent charge in density links"):
            self.initialize([{"type":"electron"},
                             {"type":"deuterium","density":{"equal_to":"0.495 * electron"}},
                             {"type":"tritium","density":{"equal_to":"0.495 * electron"}},
                             {"type":"carbon","density":{"equal_to":"0.01 * electron"}}])

    def test_cycle_ambiguous_or_nonfinite_profile_inputs_are_rejected(self):
        e={"type":"electron"}
        for density, pattern in (
            ({"init_to":"deuterium"},"Cycle"),
            ({"init_to":"not_a_species"},"Unknown species"),
            ({"init_to":"1e999 * electron"},"finite positive"),
            ({"init_to":"2 * electron trailing"},"Invalid density"),
            ({"init_to":"electron","equal_to":"electron"},"cannot both"),
        ):
            with self.subTest(density=density):
                with self.assertRaisesRegex(ValueError,pattern):
                    self.initialize([e,{"type":"deuterium","density":density},
                                    {"type":"hydrogen","qneut_species":True}])
        for attribute,value in (("mass",0.),("mass",float("inf")),("Z",0.),("Z",-2.)):
            with self.subTest(attribute=attribute,value=value):
                with self.assertRaisesRegex(ValueError,"positive"):
                    self.initialize([e,{"type":"deuterium",attribute:value}])

    def test_invalid_aux_controller_cannot_reduce_power_on_request_to_raise_it(self):
        for params in ({"p_down_multiplier":-1.},
                       {"p_up_multiplier":.9},
                       {"Q_threshold_low":11., "Q_threshold_high":8.},
                       {"p_aux_min_MW":10., "p_aux_max_MW":3.},
                       {"Q_threshold_low":float("nan")}):
            with self.subTest(params=params):
                with self.assertRaisesRegex(ValueError,"Auxiliary source control"):
                    Physics({"time_dependence":{"update_source":True, **params}},
                            self.f.grid,self.f.geo,self.f.norms)


    def test_custom_initialization_honors_both_boundary_and_averaged_targets(self):
        configs=[
            {"type":"electron","density":{"shape":"custom","func":"def init(rho): return 5-rho**2",
                                             "lineavg":3., "edge":1.2, "evolve":False},
             "temperature":{"shape":"custom","func":"def init(rho): return 8-rho**2",
                            "volavg":4., "edge":2.3}},
            {"type":"hydrogen"},
        ]
        s=self.initialize(configs)
        self.assertAlmostEqual(s.electron.n()[-1],1.2)
        self.assertAlmostEqual(np.mean(s.electron.n().profile),3.)
        self.assertAlmostEqual(s.electron.T()[-1],2.3)
        self.assertAlmostEqual(self.f.geo.volume_average(s.electron.T()),4.)
        configs[0]["temperature"]["func"]="def init(rho): return 8."
        with self.assertRaisesRegex(ValueError,"Cannot scale a custom temperature"):
            self.initialize(configs)

    def test_negative_or_out_of_domain_Zeff_cannot_manufacture_heating(self):
        sp=self.f.species()
        for invalid in (-1.,float("nan")):
            with self.assertRaisesRegex(ValueError,"positive Zeff"):
                self.f.physics(bremsstrahlung=True,Zeff=invalid).compute_bremsstrahlung(sp.electron,sp)
        with self.assertRaisesRegex(ValueError,"covering the grid"):
            self.f.physics(Zeff={"shape":"manual","profstr":"0 .2; 1 2"})
        with np.errstate(over="ignore"):
            with self.assertRaisesRegex(ValueError,"zero or invalid radial weight"):
                self.f.physics(Zeff={"shape":"gaussian","center":2., "width":1e-6, "integrated":5.})

    def test_external_source_reset_shape_and_atomic_finiteness(self):
        sp=self.f.species()
        shape=(sp.N_species,sp.N_radial)
        p=np.ones(shape)
        sp.add_source(p,2*p,label="beam")
        sp.add_source(p,2*p,label="beam")
        np.testing.assert_array_equal(sp.electron.Sp_labeled["beam"].profile,4.)
        power=sp.electron.Sp.profile.copy()
        invalid=p.copy()
        invalid[sp.sidx_electron,-1]=np.nan
        with self.assertRaisesRegex(ValueError,"finite particle and power arrays"):
            sp.add_source(p,invalid,label="beam")
        np.testing.assert_array_equal(sp.electron.Sp.profile,power)
        sp.clear_sources()
        self.assertNotIn("beam",sp.electron.Sn_labeled)
        self.assertNotIn("beam",sp.electron.Sp_labeled)
        np.testing.assert_array_equal(sp.electron.Sp.profile,0.)
        np.testing.assert_array_equal(sp.electron.Sn.profile,0.)
