import unittest
from t3d.TrinityIO import Input
from t3d.Grid import Grid
from t3d.Time import Time
from t3d.Geometry import Geometry
from t3d.Species import SpeciesDict
from t3d.Import import Import
from t3d.FluxModels import FluxModelDict
from t3d.trinity_lib import TrinNormalizations


class TestGS2_FluxModel(unittest.TestCase):

    def test_1(self):
        inputs = Input('tests/unit/test-gs2-model.in').input_dict
        grid = Grid(inputs)
        time = Time(inputs)
        imported = Import(inputs, grid)
        try:
            flux_models = FluxModelDict(inputs, grid, time)
        except RuntimeError:
            raise unittest.SkipTest("GS2 not found. Skipping test.")
        geo = Geometry(inputs, grid, flux_models, imported)
        species = SpeciesDict(inputs, grid)
        norms = TrinNormalizations(a_meters=geo.a_minor,
                                   vt_sqrt_2=flux_models.vt_sqrt_2,
                                   m_ref=species.ref_species.mass)
        species.init_profiles_and_sources(geo, norms, imported)

        species.clear_fluxes()
        for model in flux_models.get_models_list():
            model.compute_fluxes(species, geo, norms)
        for model in flux_models.get_models_list():
            model.collect_results(species, geo, norms)
        for s in species.get_species_list():
            print(f"{s.type}: qflux = {s.qflux}")


if __name__ == '__main__':
    unittest.main()
