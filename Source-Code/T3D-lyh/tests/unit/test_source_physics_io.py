import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from netCDF4 import Dataset
import numpy as np

from t3d.Geometry import Geometry
from t3d.Grid import Grid
from t3d.Import import Import
from t3d.Physics import Physics
from t3d.Profiles import GridProfile
from t3d.Species import SpeciesDict
from t3d.Time import Time
from t3d.TrinityIO import Logger
from t3d.trinity_lib import TrinNormalizations


class TestSourcePhysicsAndOutput(unittest.TestCase):

    def setUp(self):
        self.inputs = {'grid': {'N_radial': 5, 'rho_edge': .8, 'flux_label': 'rminor'},
                       'rescale': {'a_minor': 1., 'Ba': 2.},
                       'log': {'output_netcdf': True, 'output_adios2': False}}
        self.grid = Grid(self.inputs)
        self.geo = Geometry(self.inputs, self.grid)
        self.norms = TrinNormalizations(Btor=2, a_meters=1)

    def species(self, names=('deuterium', 'tritium', 'electron'), tag=None):
        configs = []
        for name, density in zip(names, [.45, .55 if len(names) == 3 else 1, 1.]):
            configs.append({'type': name, 'tag': tag,
                            'density': {'core': density, 'edge': density, 'evolve': False},
                            'temperature': {'core': 10., 'edge': 10., 'evolve': True}})
        result = SpeciesDict({'species': configs}, self.grid)
        result.init_profiles_and_sources(self.geo, self.norms)
        return result

    def physics(self, **kwargs):
        return Physics({'physics': kwargs}, self.grid, self.geo, self.norms)

    def logger(self, species, physics, time, source_models=None):
        flux_models = SimpleNamespace(get_models_list=lambda: [])
        return Logger('unused.in', self.inputs, self.grid, time, species, self.geo,
                      physics, self.norms, flux_models, source_models=source_models)

    def test_alpha_birth_and_deposition_are_distinct_and_partitioned(self):
        species = self.species()
        physic = self.physics(alpha_efficiency=.25)
        birth = physic.compute_alpha_birth_power(species)
        deposited = physic.compute_total_alpha_heating(species)
        self.assertTrue(np.all(birth.profile > 0.))
        np.testing.assert_allclose(deposited.profile, .25*birth.profile)
        np.testing.assert_allclose(physic.compute_fusion_power(species).profile, 5.03*birth.profile)
        deposited_sum = sum(physic.compute_alpha_heating(s, species).profile
                            for s in species.get_species_list())*self.norms.P_ref_MWm3
        np.testing.assert_allclose(deposited_sum, deposited.profile)
        physic.alpha_efficiency = 0.
        np.testing.assert_allclose(physic.compute_fusion_power(species).profile, 5.03*birth.profile)
        np.testing.assert_array_equal(physic.compute_total_alpha_heating(species).profile, 0.)
        feedback_off = self.physics(alpha_heating=False)
        np.testing.assert_allclose(feedback_off.compute_fusion_power(species).profile, 5.03*birth.profile)
        np.testing.assert_array_equal(feedback_off.compute_total_alpha_heating(species).profile, 0.)

    def test_default_hydrogen_only_and_nonpositive_steady_state_power(self):
        species = self.species(names=('hydrogen', 'electron'))
        physic = self.physics()
        physic.compute_sources(species, self.geo, self.norms)
        np.testing.assert_array_equal(physic.compute_alpha_birth_power(species).profile, 0.)
        np.testing.assert_array_equal(physic.compute_alpha_pressure(species).profile, 0.)
        self.assertTrue(np.isnan(physic.compute_tauE(species)))
        self.assertTrue(np.isnan(physic.compute_ISS04(species)))
        with self.assertRaisesRegex(ValueError, 'alpha_efficiency'):
            self.physics(alpha_efficiency=1.1)

    def test_radiation_precedence_and_zeff_input_models(self):
        species = self.species()
        with patch('t3d.Physics.warn') as warning:
            both = self.physics(alpha_heating=False, radiation=True, bremsstrahlung=True)
        self.assertIn('double counting', warning.call_args.args[0])
        both.compute_sources(species, self.geo, self.norms)
        np.testing.assert_array_equal(species['electron'].Sp_labeled['rad'].profile, 0.)
        standalone = self.physics(alpha_heating=False, bremsstrahlung=True)
        standalone.compute_sources(species, self.geo, self.norms)
        self.assertTrue(np.all(species['electron'].Sp_labeled['rad'].profile < 0.))
        gaussian = self.physics(alpha_heating=False, Zeff={'shape': 'gaussian', 'height': 2., 'width': .2})
        np.testing.assert_allclose(gaussian.Zeff_prof.profile, 2.*np.exp(-.5*(self.grid.rho/.2)**2))
        manual = self.physics(alpha_heating=False, Zeff={'shape': 'manual', 'profstr': '0 1; 2 3'})
        np.testing.assert_allclose(manual.Zeff_prof.profile, 2.+self.grid.rho/self.grid.rho_edge)

    def test_controller_uses_birth_and_controls_only_prescribed_auxiliary(self):
        species = self.species()
        physic = self.physics(alpha_efficiency=.05)
        fusion = self.geo.volume_integrate(physic.compute_fusion_power(species), active=True)
        conversion = self.geo.volume_integrate(GridProfile(1., self.grid), active=True)*self.norms.P_ref_MWm3
        expected_aux = fusion/20
        species['deuterium'].Sp_labeled['aux'] = GridProfile(expected_aux/conversion, self.grid)
        species['electron'].Sp_labeled['beam'] = GridProfile((expected_aux/4)/conversion, self.grid)
        physic.source_p_aux_min_MW = expected_aux
        species.update_source(physic, self.geo)
        value = species['deuterium'].Sp_labeled['aux'][0]*conversion
        self.assertAlmostEqual(value, .9*expected_aux)
        self.assertAlmostEqual(species['electron'].Sp_labeled['beam'][0]*conversion, expected_aux/4)
        self.assertAlmostEqual(species.Paux[0]*conversion, value)

    def test_netCDF_multi_ion_time_alignment_units_and_fusion_metrics(self):
        species = self.species()
        physic = self.physics(alpha_efficiency=.4, collisions=False)
        time = Time(self.inputs)
        model = SimpleNamespace(label='beam', particle_source=True, power_source=True)
        source_models = SimpleNamespace(get_sources_list=lambda: [model])
        logger = self.logger(species, physic, time, source_models)
        for step in range(2):
            species.clear_sources()
            physic.compute_sources(species, self.geo, self.norms)
            source = (step+1)*np.repeat([[1.], [2.], [4.]], len(self.grid.rho), axis=1)
            species.add_source(source, source, label='beam')
            time.diagnostics_current = True
            logger.save(species, SimpleNamespace(), time)
        time.diagnostics_current = False
        logger.save(species, SimpleNamespace(), time)
        self.assertEqual(len(logger.log['Sp_beam_i']), 3)
        fixture = Path(__file__).resolve().parents[1] / 'regression' / 'test-w7x-relu.in'
        with tempfile.TemporaryDirectory() as tmp:
            file = str(Path(tmp) / 'test.nc')
            logger.write_netCDF4(str(fixture), f_save=file)
            with Dataset(file) as nc:
                sp = nc['species']
                z = nc['zeroD']
                np.testing.assert_allclose(sp['Sp_beam_i'][:2], np.repeat([[3.], [6.]], self.grid.N_radial, axis=1))
                np.testing.assert_allclose(sp['Sn_tot_i'][:2], sp['Sn_beam_i'][:2])
                self.assertTrue(np.all(np.isnan(sp['Sp_beam_i'][2])))
                self.assertEqual(sp['Sn_beam_int_SI20_D'].units, '10^20 s^-1')
                self.assertEqual(sp['Sn_tot_cumint_SI20_D'].units, '10^20 s^-1')
                self.assertEqual(z['N_SI20_D'].units, 'particles')
                np.testing.assert_allclose(z['Pfus_MW'][:], 5.03*z['Palpha_birth_int_MW'][:])
                np.testing.assert_allclose(z['Palpha_int_MW'][:], .4*z['Palpha_birth_int_MW'][:])
                np.testing.assert_allclose(z['Qfus'][:2], z['Pfus_MW'][:2]/z['Paux_MW'][:2])
                self.assertGreater(z['Paux_MW'][0], 0.)
                self.assertTrue(np.isnan(z['Qfus'][2]))

    def test_reserved_duplicate_labels(self):
        sp = self.species(tag='i')
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.logger(sp, self.physics(), Time(self.inputs))
        sp = self.species()
        model = SimpleNamespace(label='coll', particle_source=True, power_source=True)
        with self.assertRaisesRegex(ValueError, 'source label'):
            self.logger(sp, self.physics(), Time(self.inputs), SimpleNamespace(get_sources_list=lambda: [model]))
        sp['tritium'].type_tag = 'int_MW_D'
        with self.assertRaisesRegex(ValueError, 'names collide'):
            self.logger(sp, self.physics(), Time(self.inputs))


class TestSourceRestart(unittest.TestCase):

    def setUp(self):
        self.old = Grid({'grid': {'N_radial': 4, 'rho_edge': .7}})
        self.norms = SimpleNamespace(P_ref_MWm3=4., Sn_ref_SI20=20.)

    def fixture(self, path, *, refs=True, flux_label='torflux'):
        with Dataset(path, 'w') as f:
            f.createDimension('rho', 4)
            f.createDimension('frame', 2)
            g = f.createGroup('grid')
            g.createVariable('rho', 'f8', ('rho',))[:] = self.old.rho
            g.createVariable('flux_label', str)[...] = flux_label
            norm = f.createGroup('norms')
            norm.createVariable('t_ref', 'f8')[...] = .1
            if refs:
                norm.createVariable('P_ref_MWm3', 'f8')[...] = 8.
                norm.createVariable('Sn_ref_SI20', 'f8')[...] = 5.
            s = f.createGroup('species')
            values = [[2., 3., 7., 99.], [np.nan]*4]
            s.createVariable('Sp_aux_D', 'f8', ('frame', 'rho'))[:] = values
            s.createVariable('Sn_aux_D', 'f8', ('frame', 'rho'))[:] = values

    def test_native_grid_alias_source_old_and_new_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'source.nc')
            self.fixture(path)
            imp = Import({'import': {'type': '  T3D ', 'file': path, 'frame': 0}}, self.old)
            self.assertEqual(imp.type, 'trinity')
            np.testing.assert_array_equal(imp.get_pressure_source(tag='D', norms=self.norms), [4., 6., 14., 198.])
            np.testing.assert_array_equal(imp.get_density_source(tag='D', norms=self.norms), [.5, .75, 1.75, 24.75])
            self.assertAlmostEqual(imp.get_pressure_source(tag='D')[0], 16.)

    def test_restart_remap_has_no_fixed_node_or_outside_domain_input(self):
        new = Grid({'grid': {'N_radial': 8, 'rho_edge': .9}})
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'source.nc')
            self.fixture(path)
            imp = Import({'import': {'type': 'Trinity', 'file': path, 'frame': 0}}, new)
            result = imp.get_pressure_source(tag='D', norms=self.norms)
            self.assertEqual(result.size, len(new.rho))
            self.assertEqual(result[-1], 0.)
            self.assertLessEqual(np.max(result), 14.)
            old_volumes = np.diff(np.r_[0., self.old.midpoints])
            new_volumes = np.diff(np.r_[0., new.midpoints])
            self.assertAlmostEqual(np.dot(result[:-1], new_volumes), np.dot([4., 6., 14.], old_volumes))

    def test_restart_legacy_tref_and_nonavailable_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'source.nc')
            self.fixture(path, refs=False)
            inp = {'import': {'type': 'trinity', 'file': path, 'frame': 0}}
            imp = Import(inp, self.old)
            self.assertAlmostEqual(imp.get_density_source(tag='D', norms=self.norms)[0], 1.)
            self.assertAlmostEqual(imp.get_pressure_source(tag='D', norms=self.norms)[0], 2*.01602176634/.1/4)
            inp['import']['frame'] = -1
            with self.assertRaisesRegex(ValueError, 'unavailable at frame'):
                Import(inp, self.old).get_pressure_source(tag='D', norms=self.norms)

    def test_restart_flux_label_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'source.nc')
            self.fixture(path, flux_label='rminor')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                Import({'import': {'type': 'trinity', 'file': path}}, self.old)

    def test_bp_backend_metadata_contract(self):
        attributes = {'grid/flux_label': ['torflux'], 'norms/P_ref_MWm3': np.array([8.])}
        variables = {'grid/rho': self.old.rho, 'species/Sp_aux_D': np.array([[2., 3., 7., 0.]])}
        bp = SimpleNamespace(available_attributes=lambda: attributes,
                             read_attribute=attributes.__getitem__,
                             available_variables=lambda: variables,
                             read=variables.__getitem__, close=lambda: None)
        with patch('t3d.Import.adios2.FileReader', return_value=bp):
            imp = Import({'import': {'type': 'trinity', 'file': 'mock.bp'}}, self.old)
            np.testing.assert_array_equal(imp.get_pressure_source(tag='D', norms=self.norms), [4., 6., 14., 0.])

    def test_jetto_timing_alias_and_torflux_coordinate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'jetto.nc')
            with Dataset(path, 'w') as f:
                f.createDimension('time', 2)
                f.createDimension('radius', 4)
                f.createVariable('time', 'f8', ('time',))[:] = [0., 1.]
                f.createVariable('X', 'f8', ('time', 'radius'))[:] = [self.old.rho, self.old.rho]
                f.createVariable('PE', 'f8', ('time', 'radius'))[:] = [[4.e6]*4, [8.e6]*4]
            for key in ('jetto_time', 'jett0_time'):
                imp = Import({'import': {'type': 'JETTO', 'file': path, key: .1}}, self.old)
                self.assertEqual(imp.imported.time_index, 0)
                np.testing.assert_allclose(imp.get_pressure_source(keys='PE', norms=self.norms), 1.)

    def test_ascii_source_keys_are_summed_and_validated(self):
        with patch('builtins.open', return_value=io.StringIO('# Header\n\n0 2 4\n  1 3 7 # inline\n')):
            imp = Import({'import': {'type': 'COLUMNS', 'file': 'in-memory',
                                     'columns': ['rho', 'a', 'b'], 'divide_by': [1, 1, 2]}}, self.old)
        np.testing.assert_allclose(imp.get_pressure_source(keys=['a', 'b'], interpolate=False, norms=self.norms), [1, 1.625])
        np.testing.assert_allclose(imp.get_density_source(keys=['a', 'b'], interpolate=False, norms=self.norms), [.2, .325])
        expected = (4 + 2.5*self.old.rho) / 4.
        np.testing.assert_allclose(imp.get_pressure_source(keys=['a', 'b'], norms=self.norms), expected)
        with self.assertRaisesRegex(ValueError, 'Missing ASCII source'):
            imp.get_pressure_source(keys=['x'])
