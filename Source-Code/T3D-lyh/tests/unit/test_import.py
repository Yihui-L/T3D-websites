import unittest
from t3d.TrinityIO import Input
from t3d.Species import SpeciesDict
from t3d.Geometry import Geometry
from t3d.trinity_lib import TrinNormalizations
from t3d.Grid import Grid
from t3d.Import import Import
from netCDF4 import Dataset
import numpy as np


class TestImport(unittest.TestCase):

    def test_ufile(self):
        inputs = Input('tests/unit/test-import-ufile.in').input_dict
        grid = Grid(inputs)
        imported = Import(inputs, grid)
        geo = Geometry(inputs, grid, imported=imported)

        species = SpeciesDict(inputs, grid)
        norms = TrinNormalizations(Btor=geo.Btor,a_meters=geo.a_minor,
                                   vt_sqrt_2=False,
                                   m_ref=species.ref_species.mass)
        species.init_profiles_and_sources(geo, norms, imported)
        print("n_D = ", species['deuterium'].n())
        print("n_T = ", species['tritium'].n())
        print("n_e = ", species['electron'].n())
        print("T_D = ", species['deuterium'].T())
        print("T_T = ", species['tritium'].T())
        print("T_e = ", species['electron'].T())

        print("Sn_D = ", species['deuterium'].Sn_labeled['aux'])
        print("Sn_T = ", species['tritium'].Sn_labeled['aux'])
        print("Sn_e = ", species['electron'].Sn_labeled['aux'])

        print("Sp_D = ", species['deuterium'].Sp_labeled['aux'])
        print("Sp_T = ", species['tritium'].Sp_labeled['aux'])
        print("Sp_e = ", species['electron'].Sp_labeled['aux'])

        print("kappa = ", geo.kappa)

    def test_ascii(self):
        inputs = Input('tests/unit/test-import-ascii.in').input_dict
        grid = Grid(inputs)
        imported = Import(inputs, grid)
        geo = Geometry(inputs, grid, imported=imported)

        species = SpeciesDict(inputs, grid)
        norms = TrinNormalizations(Btor=geo.Btor,a_meters=geo.a_minor,
                                   vt_sqrt_2=False,
                                   m_ref=species.ref_species.mass)
        species.init_profiles_and_sources(geo, norms, imported)

        print("n_H = ", species['hydrogen'].n())
        print("n_e = ", species['electron'].n())
        print("T_H = ", species['hydrogen'].T())
        print("T_e = ", species['electron'].T())

    def test_ascii_power_source(self):
        inputs = Input('tests/unit/test-import-ascii-power.in').input_dict
        grid = Grid(inputs)
        imported = Import(inputs, grid)
        geo = Geometry(inputs, grid, imported=imported)

        species = SpeciesDict(inputs, grid)
        norms = TrinNormalizations(Btor=geo.Btor,a_meters=geo.a_minor,
                                   vt_sqrt_2=False,
                                   m_ref=species.ref_species.mass)
        species.init_profiles_and_sources(geo, norms, imported)

        print("n_D = ", species['deuterium'].n())
        print("n_C = ", species['carbon'].n())
        print("n_e = ", species['electron'].n())

        print("T_D = ", species['deuterium'].T())
        print("T_C = ", species['carbon'].T())
        print("T_e = ", species['electron'].T())

        print("Sp_D = ", species['deuterium'].Sp_labeled['aux'])
        print("Sp_C = ", species['carbon'].Sp_labeled['aux'])
        print("Sp_e = ", species['electron'].Sp_labeled['aux'])

    def test_npy(self):
        inputs = Input('tests/unit/test-import-npy.in').input_dict
        grid = Grid(inputs)
        imported = Import(inputs, grid)
        geo = Geometry(inputs, grid, imported=imported)

        species = SpeciesDict(inputs, grid)
        norms = TrinNormalizations(Btor=geo.Btor,a_meters=geo.a_minor,
                                   vt_sqrt_2=False,
                                   m_ref=species.ref_species.mass)
        species.init_profiles_and_sources(geo, norms, imported)

        data_check = Dataset('tests/regression/ref-w7x-relu.nc')
        n_H_check = data_check['species']['n_H'][-1,:]
        n_e_check = data_check['species']['n_e'][-1,:]
        T_H_check = data_check['species']['T_H'][-1,:]
        T_e_check = data_check['species']['T_e'][-1,:]
        # print("n_H = ", species['hydrogen'].n())
        # print("n_e = ", species['electron'].n())
        # print("T_H = ", species['hydrogen'].T())
        # print("T_e = ", species['electron'].T())

        self.assertTrue(np.allclose(species['hydrogen'].n(), n_H_check))
        self.assertTrue(np.allclose(species['electron'].n(), n_e_check))
        self.assertTrue(np.allclose(species['hydrogen'].T(), T_H_check))
        self.assertTrue(np.allclose(species['electron'].T(), T_e_check))

    def test_nc(self):
        inputs = Input('tests/unit/test-import-nc.in').input_dict
        grid = Grid(inputs)
        imported = Import(inputs, grid)
        geo = Geometry(inputs, grid, imported=imported)

        species = SpeciesDict(inputs, grid)
        norms = TrinNormalizations(Btor=geo.Btor,a_meters=geo.a_minor,
                                   vt_sqrt_2=False,
                                   m_ref=species.ref_species.mass)
        species.init_profiles_and_sources(geo, norms, imported)

        data_check = Dataset('tests/regression/ref-w7x-relu.nc')
        n_H_check = data_check['species']['n_H'][-1,:]
        n_e_check = data_check['species']['n_e'][-1,:]
        T_H_check = data_check['species']['T_H'][-1,:]
        T_e_check = data_check['species']['T_e'][-1,:]

        self.assertTrue(np.allclose(species['hydrogen'].n(), n_H_check))
        self.assertTrue(np.allclose(species['electron'].n(), n_e_check))
        self.assertTrue(np.allclose(species['hydrogen'].T(), T_H_check))
        self.assertTrue(np.allclose(species['electron'].T(), T_e_check))

        # print("n_H = ", species['hydrogen'].n())
        # print("n_e = ", species['electron'].n())
        # print("T_H = ", species['hydrogen'].T())
        # print("T_e = ", species['electron'].T())

    def test_bp(self):
        inputs = Input('tests/unit/test-import-bp.in').input_dict
        grid = Grid(inputs)
        imported = Import(inputs, grid)
        geo = Geometry(inputs, grid, imported=imported)

        species = SpeciesDict(inputs, grid)
        norms = TrinNormalizations(Btor=geo.Btor,a_meters=geo.a_minor,
                                   vt_sqrt_2=False,
                                   m_ref=species.ref_species.mass)
        species.init_profiles_and_sources(geo, norms, imported)

        data_check = Dataset('tests/regression/ref-jet-relu.nc')
        n_D_check = data_check['species']['n_D'][-1,:]
        n_e_check = data_check['species']['n_e'][-1,:]
        T_D_check = data_check['species']['T_D'][-1,:]
        T_e_check = data_check['species']['T_e'][-1,:]

        print("n_D = ", species['deuterium'].n())
        print("n_e = ", species['electron'].n())
        print("T_D = ", species['deuterium'].T())
        print("T_e = ", species['electron'].T())

        print("n_D_check = ", n_D_check)
        print("n_e_check = ", n_e_check)
        print("T_D_check = ", T_D_check)
        print("T_e_check = ", T_e_check)

        self.assertTrue(np.allclose(species['deuterium'].n(), n_D_check))
        self.assertTrue(np.allclose(species['electron'].n(), n_e_check))
        self.assertTrue(np.allclose(species['deuterium'].T(), T_D_check))
        self.assertTrue(np.allclose(species['electron'].T(), T_e_check))


if __name__ == '__main__':
    unittest.main()
