import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from netCDF4 import Dataset
from scipy.integrate import quad
from scipy.interpolate import interp1d

from t3d.Geometry import Geometry
from t3d.Grid import Grid
from t3d.Profiles import GridProfile
from t3d.Species import Species
from t3d.geometry_models.VMEC import VmecReader


class TestVmecTransportGeometry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.wout = Path(__file__).resolve().parents[1] / 'data' / 'wout_w7x.nc'
        cls.grid = Grid({'grid': {'N_radial': 9, 'rho_edge': 0.7}})
        cls.geometry = Geometry(
            {'geometry': {'geo_option': 'vmec', 'geo_file': str(cls.wout)},
             'rescale': {'L_mult': 1.7}},
            cls.grid)
        cls.vmec = cls.geometry.VmecReader
        with Dataset(cls.wout) as f:
            cls.vp = np.asarray(f.variables['vp'][1:])
        cls.half_s = (np.arange(cls.vmec.ns - 1) + 0.5) / (cls.vmec.ns - 1)

    def expected_derivative(self, rho):
        return 2 * rho * (2 * np.pi)**2 * interp1d(self.half_s, self.vp, fill_value='extrapolate')(rho**2)

    def test_volume_derivative_uses_the_vmec_half_grid(self):
        expected = self.expected_derivative(self.vmec.rho)
        np.testing.assert_allclose(self.vmec.dVdrho, expected, rtol=4e-15, atol=0)
        np.testing.assert_allclose((2 * np.pi)**2 * np.mean(self.vp), self.vmec.volume, rtol=1e-8, atol=0)
        self.assertEqual(self.vmec.dVdrho[0], 0.)

    def test_mapped_area_over_gradient_gives_scaled_volume_derivative(self):
        for rho, area, grho in (
            (self.grid.rho, self.geometry.area_grid, self.geometry.grho_grid),
            (self.grid.midpoints, self.geometry.area, self.geometry.grho),
        ):
            inferred = area.profile / grho.profile * self.geometry.a_minor**3
            expected = self.expected_derivative(rho) * self.geometry.L_scale**3
            np.testing.assert_allclose(inferred, expected, rtol=2e-15, atol=0)

    def test_periodic_area_quadrature_converges_without_double_seam(self):
        # Restrict the expensive high-resolution angular calculation to a
        # few surfaces without changing their native Fourier spectra.
        rows = [2, len(self.vmec.rho)//2, len(self.vmec.rho)-1]
        sample = copy.copy(self.vmec)
        sample.ns = len(rows)
        sample.rmnc = self.vmec.rmnc[rows]
        sample.zmns = self.vmec.zmns[rows]
        sample.dVdrho = self.vmec.dVdrho[rows]
        sample.compute_area_grad_rho(N_phi=96, N_theta=96)
        np.testing.assert_allclose(self.vmec.area[rows], sample.area, rtol=5e-7, atol=0)

    def test_asymmetric_wout_is_explicitly_rejected(self):
        with Dataset(self.wout) as symmetric:
            asym = Mock(variables=dict(symmetric.variables))
            asym.variables['lasym__logical__'] = np.array([1])
            with patch('netCDF4.Dataset', return_value=asym):
                with self.assertRaisesRegex(ValueError, 'stellarator-symmetric wout'):
                    VmecReader('unused.nc')
        asym.close.assert_called_once()


class TestAuxiliarySourceConservation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.grid = Grid({'grid': {'N_radial': 9, 'rho_edge': 0.7, 'flux_label': 'rminor'}})
        cls.geometry = Geometry({'geometry': {'AspectRatio': 4}, 'rescale': {'a_minor': 1.}}, cls.grid)
        cls.faces = np.r_[0., cls.grid.midpoints]
        cls.volumes = (
            cls.geometry.area_grid.profile[:-1] / cls.geometry.grho_grid.profile[:-1]
            * cls.grid.drho * cls.geometry.a_minor**3
        )

    def source(self, center, width, height=0., integrated=None, unit_factor=3.4):
        # Use the ordinary initialization path shared by old and new code.
        species = Species.__new__(Species)
        species.evolve_density = True
        species.evolve_temperature = True
        species.Sn_labeled, species.Sp_labeled = {}, {}
        for prefix in ('Sn', 'Sp'):
            for setting, value in (
                ('import', False), ('shape', 'gaussian'), ('center', center),
                ('width', width), ('height', height), ('int', integrated)
            ):
                setattr(species, f'{prefix}_aux_{setting}', value)
        norms = SimpleNamespace(Sn_ref_SI20=unit_factor, P_ref_MWm3=unit_factor)
        species.init_aux_sources(self.grid, self.geometry, norms)
        particle, power = species.Sn_labeled['aux'], species.Sp_labeled['aux']
        np.testing.assert_allclose(particle.profile, power.profile, rtol=0, atol=0)
        return power

    def test_height_preserves_continuous_peak_and_uses_shell_average(self):
        center, width, height, unit_factor = 0.33, 0.14, 1.7, 3.4
        source = self.source(center, width, height=height, unit_factor=unit_factor).profile
        expected = [
            quad(lambda r: height / unit_factor * np.exp(-0.5 * ((r-center)/width)**2), a, b)[0] / (b-a)
            for a, b in zip(self.faces[:-1], self.faces[1:])
        ]
        np.testing.assert_allclose(source[:-1], expected, rtol=3e-15, atol=0)
        self.assertEqual(source[-1], 0.)

    def test_integrated_narrow_source_between_nodes_is_not_lost(self):
        target, factor = 1.0, 3.4
        center = self.faces[-1] - 0.2 * self.grid.drho
        with patch('t3d.Species.warn', create=True) as log_warn:
            source = self.source(center, self.grid.drho * 1e-10, integrated=target, unit_factor=factor)
        np.testing.assert_allclose(factor * np.dot(source.profile[:-1], self.volumes), target, rtol=1e-15, atol=0)
        self.assertEqual(log_warn.call_count, 2)
        self.assertIn('not resolved', log_warn.call_args.args[0])
        self.assertTrue(np.all(source.profile[:-2] == 0.))
        self.assertGreater(source[-2], 0.)
        self.assertEqual(source[-1], 0.)

    def test_integrated_source_near_ended_shell_matches_independent_quadrature(self):
        center, width, target, factor = 0.65, 0.06, 1.0, 3.4
        source = self.source(center, width, integrated=target, unit_factor=factor)
        deposited = factor * source.profile[:-1] * self.volumes
        quadrature = np.array([
            quad(lambda r: np.exp(-0.5 * ((r-center)/width)**2), a, b)[0] / (b-a)
            for a, b in zip(self.faces[:-1], self.faces[1:])
        ]) * self.volumes
        np.testing.assert_allclose(deposited, target * quadrature / np.sum(quadrature), rtol=2e-14, atol=0)
        self.assertAlmostEqual(float(np.sum(deposited)), target, places=14)

    def test_outside_center_deposits_requested_integral_with_warning_if_tail_overlaps(self):
        with patch('t3d.Species.warn', create=True) as log_warn:
            source = self.source(self.grid.rho[-1], 0.02, integrated=1.)
        self.assertAlmostEqual(float(3.4 * np.dot(source.profile[:-1], self.volumes)), 1., places=14)
        self.assertEqual(source[-1], 0.)
        messages = ' '.join(call.args[0] for call in log_warn.call_args_list)
        self.assertIn('outside the evolved interval', messages)
        self.assertIn('renormalizing', messages)

    def test_invalid_or_zero_source_specification(self):
        with self.assertRaisesRegex(ValueError, 'outside the evolved interval.*fixed boundary node.*fraction'):
            self.source(self.grid.rho[-1], 0.001, integrated=1.)
        with self.assertRaisesRegex(ValueError, 'positive finite width'):
            self.source(0.5, 0., integrated=1.)
        self.assertTrue(np.all(self.source(0.3, 0.1, height=2., integrated=0.).profile == 0.))

    def test_integrated_diagnostics_exclude_fixed_node_even_for_imports(self):
        data = GridProfile(np.r_[np.ones_like(self.volumes), 1e9], self.grid)
        active = self.geometry.volume_integrate(data, active=True)
        cumulative = self.geometry.volume_integrate(data, active=True, profile=True).profile
        expected = np.cumsum(self.volumes)
        self.assertAlmostEqual(active, expected[-1], places=12)
        np.testing.assert_allclose(cumulative, np.r_[expected, expected[-1]], rtol=1e-15, atol=0)
