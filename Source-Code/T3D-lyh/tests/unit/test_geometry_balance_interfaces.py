import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from t3d.Geometry import Geometry
from t3d.Grid import Grid
from t3d.geometry_models.DESC import DescHandler
from t3d.Profiles import GridProfile
from t3d.trinity_lib import TrinityEngine, TrinNormalizations
from tests.unit.test_discrete_jacobian import fixture


class TestGeometryAndBalance(unittest.TestCase):
    def test_basic_geometry_dimensional_and_default_field(self):
        cfg = {"grid": {"N_radial": 5, "rho_edge": 1, "flux_label": "rminor"},
               "geometry":{"AspectRatio":4}, "rescale":{"L_def":"minor","a_minor":2,"B_mult":3}}
        grid = Grid(cfg)
        g=Geometry(cfg,grid)
        np.testing.assert_allclose(g.grho.profile,1)
        np.testing.assert_allclose(g.area.profile*g.a_minor**2,4*np.pi**2*g.R_major*g.a_minor*grid.midpoints)
        self.assertEqual(g.Btor,3.)
        self.assertAlmostEqual(g.volume_integrate(GridProfile(1,grid),active=True),
                               g.volume*grid.midpoints[-1]**2)
        cfg["rescale"]={"L_def":"minor"}
        self.assertEqual(Geometry(cfg,grid).a_minor,1.)
        cfg["rescale"]={}
        self.assertEqual(Geometry(cfg,grid).R_major,4.)
        cfg["geometry"].update(evolve_equilibrium=False,update_equilibrium=True)
        with self.assertRaises(ValueError):
            Geometry(cfg,grid)

    def test_desc_symmetric_threshold(self):
        np.testing.assert_allclose(DescHandler.pressure_ratio([10.,0],[5.,0]),2.)
        np.testing.assert_allclose(DescHandler.pressure_ratio([5.,0],[10.,0]),2.)
        with self.assertRaises(ValueError):
            DescHandler.pressure_ratio([5],[np.nan])

    def test_desc_commits_only_success_and_geometry_with_same_shells(self):
        d=DescHandler.__new__(DescHandler)
        d.desc_eq=Mock()
        d.get_pressure_SI=Mock(return_value=np.full(5,2000.))
        d.get_geometry=Mock()
        d.use_vmec_format=False
        template=d.desc_eq.copy.return_value
        result=SimpleNamespace(success=False)
        template.solve.return_value=(template,result)
        template.save.side_effect=lambda path: Path(path).touch()
        g=Grid({"grid":{"N_radial":5}})
        sp=SimpleNamespace(get_pressure_SI=lambda:np.full(5,1000.))
        time=SimpleNamespace(step_idx=2,iter_idx=0)
        ps=ModuleType("desc.profiles")
        ps.PowerSeriesProfile=MagicMock()
        ps.PowerSeriesProfile.from_values.return_value=lambda x: np.ones_like(x)
        with TemporaryDirectory() as root, patch.dict(sys.modules,{"desc.profiles":ps}):
            geo=SimpleNamespace(grid=g,desc_pressure_threshold=1.25,desc_pressure_perturb=1.9,
                                geo_outputs=root)
            with self.assertRaisesRegex(RuntimeError,"did not converge"):
                d.update_desc_eq(geo,sp,time)
            result.success=True
            candidate=d.update_desc_eq(geo,sp,time)
            self.assertIsNot(candidate,d)
            self.assertFalse(hasattr(d,"output_file"))
            self.assertTrue(Path(candidate.output_file).is_file())
            self.assertIsNone(candidate.update_desc_eq(geo,sp,time))

        geometry=Geometry({"geometry":{"AspectRatio":4}},g)
        geometry.DescHandler=Mock()
        geometry.DescHandler.update_desc_eq.return_value=SimpleNamespace(output_file="desc.h5")
        sp.norms=TrinNormalizations()
        with patch.object(Geometry,"init_geometry",autospec=True) as init:
            init.side_effect=lambda trial: setattr(trial,"a_minor",2)
            with self.assertRaisesRegex(NotImplementedError,"moving-volume"):
                geometry.update_equilibrium(sp,time)
            self.assertEqual(geometry.a_minor,1.)
            init.side_effect=lambda trial: setattr(trial,"Btor",1.3)
            self.assertTrue(geometry.update_equilibrium(sp,time))
            self.assertEqual(geometry.Btor,1.3)
            self.assertEqual(sp.norms.Btor,1.3)

    def test_power_balance_uses_current_sources_in_each_shell(self):
        from t3d.flux_models.PowerBalance import PowerBalance_FluxModel
        sp,_,physics,solver=fixture()
        geo=Geometry({"geometry":{"AspectRatio":4}},sp.grid)
        sp.norms.a_meters=geo.a_minor
        e=TrinityEngine.__new__(TrinityEngine)
        model=PowerBalance_FluxModel({},sp.grid,SimpleNamespace())
        e.flux_models=SimpleNamespace(get_models_list=lambda:[model])
        def inject(species, *_):
            S = np.asarray([1+sp.grid.rho**2 for s in sp.get_species_list()])
            species.add_source(S,3*S,label="test")
        source=SimpleNamespace(label="test",compute_sources=Mock(),collect_results=inject)
        e.source_models=SimpleNamespace(get_sources_list=lambda:[source])
        e.species,e.physics,e.geometry,e.norms,e.solver=sp,physics,geo,sp.norms,solver
        e.time=SimpleNamespace()
        solver.geo=geo
        for _ in range(2):
            e._evaluate_transport_terms()
            G=np.tile((geo.G_fac/sp.grid.drho).profile,sp.evolve_count)
            np.testing.assert_allclose(G*(solver.F_plus_vec(sp)-solver.F_minus_vec(sp)),solver.Svec(sp),atol=5e-14)
            self.assertIn(model.label,sp.electron.qflux_labeled)

    def test_engine_drains_backends_on_collection_error(self):
        e=TrinityEngine.__new__(TrinityEngine)
        e.time=SimpleNamespace(diagnostics_current=True)
        model, source=Mock(), Mock()
        e.flux_models=SimpleNamespace(get_models_list=lambda:[model])
        e.source_models=SimpleNamespace(get_sources_list=lambda:[source])
        e._compute_transport_terms=Mock(side_effect=RuntimeError("backend failed"))
        with self.assertRaisesRegex(RuntimeError,"backend failed"):
            e._evaluate_transport_terms()
        model.cancel_pending.assert_called_once()
        source.cancel_pending.assert_called_once()
        self.assertFalse(e.time.diagnostics_current)
