import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from netCDF4 import Dataset

from t3d.trinity_lib import TrinityEngine


class TestImportedFlux(unittest.TestCase):
    rho_edge = .9
    n_radial = 5
    time_end = .05
    step = .025

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.table = self.root / "external.dat"
        self.midpoints = 2 * self.rho_edge * np.arange(1, self.n_radial) / (2*self.n_radial - 1)
        # Both signs; one inlet/outlet pair must turn into redistribution,
        # not a pointwise negative source.
        self.particles = np.array([.20, -.35, .55, .48])
        self.heat = np.array([.19, -.11, .33, .28])

    def make_case(self, name, flux_file, import_fields, *, keys=True, size=.7, field=2.4):
        particle = 'particle_flux = { import = true, keys = "Gam_e" }' if keys else (
            'particle_flux = { import = true }')
        heat = 'heat_flux = { import = true, keys = "Q_H" }' if keys else (
            'heat_flux = { import = true }')
        case = self.root / (name + ".in")
        case.write_text(f"""
[grid]
N_radial = {self.n_radial}
rho_edge = {self.rho_edge}
flux_label = "rminor"

[time]
use_SI = true
alpha = 1
dtau = {self.step}
dtau_max = {self.step}
t_max = {self.time_end}
N_steps = 2
newton_threshold = 1e-11
newton_tolerance = 1e-10

[geometry]
geo_option = "basic"
AspectRatio = 4.0

[rescale]
L_def = "minor"
a_minor = {size}
Ba = {field}

[physics]
collisions = false
alpha_heating = false
radiation = false
bremsstrahlung = false
turbulent_exchange = false

[[model]]
model = "Import"

[[species]]
type = "hydrogen"
qneut_species = true
density = {{ core = 2.0, edge = 1.8, evolve = false }}
temperature = {{ core = 3.0, edge = 2.0, evolve = true }}
{heat}

[[species]]
type = "electron"
density = {{ core = 2.0, edge = 1.8, evolve = true }}
temperature = {{ core = 4.5, edge = 3.3, evolve = false }}
{particle}

[import]
file = {json.dumps(str(flux_file))}
{import_fields}

[log]
f_save = {json.dumps(str(self.root / name))}
output_netcdf = true
output_adios2 = false
""")
        return case

    def ascii_case(self, name="from_ascii"):
        # The file's raw particle rate is in particles/s; divide_by makes
        # it 1e20 particles/s. The power column already has units of MW.
        return self.make_case(
            name, self.table,
            'type = "columns"\ncolumns = ["rho", "Gam_e", "Q_H"]\n'
            'divide_by = [1.0, 1e20, 1.0]',
        )

    def save_table(self, rho=None, particles=None, heat=None):
        data = np.column_stack((
            self.midpoints if rho is None else rho,
            1e20 * (self.particles if particles is None else particles),
            self.heat if heat is None else heat,
        ))
        np.savetxt(self.table, data)

    def check_frame(self, path, particles, heat):
        with Dataset(path) as result:
            self.assertEqual(int(result["time"]["t_diagnostics_valid"][-1]), 1)
            diag = result["species"]
            for key in ("Gam_SI20_e", "Gam_Import_SI20_e"):
                np.testing.assert_allclose(diag[key][-1], particles, rtol=2e-14, atol=1e-15)
            for key in ("Q_MW_H", "Q_Import_MW_H"):
                np.testing.assert_allclose(diag[key][-1], heat, rtol=2e-14, atol=1e-15)
            for key in ("Sn_tot_e", "Sp_tot_H"):
                np.testing.assert_array_equal(diag[key][-1], 0.)

    def test_fixed_dimensional_face_flux_solves_shell_balance(self):
        self.save_table()
        e = TrinityEngine(str(self.ascii_case()))
        self.assertEqual(e.species.n_evolve_keys, ["electron"])
        self.assertEqual(e.species.T_evolve_keys, ["hydrogen"])
        np.testing.assert_allclose(e.grid.midpoints, self.midpoints, atol=1e-15, rtol=0)
        n0 = e.species["electron"].n().profile.copy()
        p0 = e.species["hydrogen"].p().profile.copy()
        Te0 = e.species["electron"].T().profile.copy()
        e.evolve_profiles()

        # Independent volume formula for basic geometry: V(<rho) is
        # 2*pi^2*(4a)*a^2*rho^2, with zero throughput at rho=0.
        shell = (2 * np.pi**2 * 4 * .7**3) * np.diff(np.r_[0., self.midpoints**2])
        dN = -np.diff(np.r_[0., self.particles]) * self.time_end
        dW = -np.diff(np.r_[0., self.heat]) * self.time_end
        delta_n = np.r_[dN / shell, 0.]
        delta_p = np.r_[dW * 1e6 / (1.5 * e.norms.p_ref * shell), 0.]
        n_final = e.species["electron"].n().profile
        p_final = e.species["hydrogen"].p().profile
        np.testing.assert_allclose(n_final, n0 + delta_n, rtol=0, atol=2e-13)
        np.testing.assert_allclose(p_final, p0 + delta_p, rtol=0, atol=2e-13)
        np.testing.assert_array_equal(e.species["hydrogen"].n().profile, n_final)
        np.testing.assert_allclose(e.species["electron"].T().profile, Te0, rtol=0, atol=2e-13)
        self.assertEqual(e.time.step_idx, 2)
        self.assertAlmostEqual(e.time.time * e.norms.t_ref, self.time_end)
        np.testing.assert_allclose(np.dot(shell, (n_final - n0)[:-1]),
                                   -self.time_end * self.particles[-1], rtol=0, atol=5e-14)
        np.testing.assert_allclose(np.dot(shell, (p_final - p0)[:-1]) * (1.5*e.norms.p_ref/1e6),
                                   -self.time_end * self.heat[-1], rtol=0, atol=5e-14)
        # Freeze a dimensional flux, not a normalized flux: no part of
        # the spatial residual may acquire a profile derivative.
        e._evaluate_transport_terms()
        np.testing.assert_allclose(e.solver.psimat(e.species), 0., rtol=0, atol=1e-12)
        np.testing.assert_array_equal(e.solver.Svec(e.species), 0.)
        self.check_frame(self.root / "from_ascii.nc", self.particles, self.heat)

    def test_face_throughput_and_its_discrete_compensating_source_are_equivalent(self):
        # A signed linear volume source is averaged exactly by the ASCII
        # source importer. Choose the flux's divergence to be its negative.
        shell_centers = (np.r_[0., self.midpoints[:-1]] + self.midpoints) / 2
        shell_volumes = (2 * np.pi**2 * 4 * .7**3) * np.diff(np.r_[0., self.midpoints**2])
        sn = lambda rho: .18 - .33*rho
        sp = lambda rho: .06 - .12*rho
        particle = -np.cumsum(shell_volumes*sn(shell_centers))
        power = -np.cumsum(shell_volumes*sp(shell_centers))
        self.save_table(particles=particle, heat=power)
        direct = TrinityEngine(str(self.ascii_case()))

        source_file = self.root / "source.dat"
        rho = np.array([0., 1.])
        np.savetxt(source_file, np.column_stack((rho, sn(rho), sp(rho))))
        source_case = self.make_case(
            "source", source_file, 'type = "columns"\ncolumns = ["rho", "Sn_e", "Sp_H"]',
        )
        config = 'model = []\n' + source_case.read_text().replace(
            '[[model]]\nmodel = "Import"\n', '',
        ).replace(
            'particle_flux = { import = true, keys = "Gam_e" }',
            'aux_particle_source = { import = true, keys = "Sn_e" }',
        ).replace(
            'heat_flux = { import = true, keys = "Q_H" }',
            'aux_power_source = { import = true, keys = "Sp_H" }',
        )
        source_case.write_text(config)
        equivalent = TrinityEngine(str(source_case))
        direct.evolve_profiles()
        equivalent.evolve_profiles()
        np.testing.assert_allclose(equivalent.solver.y_curr, direct.solver.y_curr,
                                   rtol=0, atol=4e-14)
        self.check_frame(self.root / "from_ascii.nc", particle, power)
        with Dataset(self.root / "source.nc") as result:
            d = result["species"]
            np.testing.assert_array_equal(d["Q_MW_H"][-1], 0.)
            np.testing.assert_allclose(d["Sp_tot_int_MW_H"][-1], -power[-1], atol=2e-14)
            np.testing.assert_allclose(d["Sn_tot_int_SI20_e"][-1], -particle[-1], atol=2e-14)

    def test_native_replay_keeps_dimensional_units_on_new_geometry(self):
        self.save_table()
        e = TrinityEngine(str(self.ascii_case()))
        e.evolve_profiles()
        for name, suffix, kind in (
            ("nc", ".nc", "t3d"),
            ("npy", ".log.npy", "trinity"),
        ):
            with self.subTest(format=name):
                replay = self.make_case(
                    "replayed_" + name, self.root / ("from_ascii" + suffix),
                    f'type = "{kind}"\nframe = -1', keys=False, size=.9, field=3.1,
                )
                e2 = TrinityEngine(str(replay))
                e2.evolve_profiles()
                self.assertNotEqual(e.norms.t_ref, e2.norms.t_ref)
                self.check_frame(replay.with_suffix(".nc"), self.particles, self.heat)

    def test_native_replay_rejects_finite_flux_from_invalid_frame(self):
        self.save_table()
        e = TrinityEngine(str(self.ascii_case()))
        e.evolve_profiles()
        with Dataset(self.root / "from_ascii.nc", "r+") as f:
            f["time"]["t_diagnostics_valid"][-1] = 0
            self.assertTrue(np.all(np.isfinite(f["species"]["Q_MW_H"][-1])))
        replay = self.make_case(
            "stale", self.root / "from_ascii.nc",
            'type = "trinity"\nframe = -1', keys=False,
        )
        with self.assertRaisesRegex(ValueError, "unavailable at frame"):
            e2 = TrinityEngine(str(replay))
            e2._evaluate_transport_terms()

    def test_native_replay_rejects_gyrobohm_column(self):
        self.save_table()
        e = TrinityEngine(str(self.ascii_case()))
        e.evolve_profiles()
        replay = self.make_case(
            "not_dimensional", self.root / "from_ascii.nc",
            'type = "trinity"\nframe = -1', keys=False,
        )
        replay.write_text(replay.read_text().replace(
            'heat_flux = { import = true }', 'heat_flux = { import = true, keys = "qflux_H" }',
        ))
        with self.assertRaisesRegex(ValueError, "physical diagnostic"):
            TrinityEngine(str(replay))

    def test_ascii_linear_face_interpolation_preserves_axis_and_sign(self):
        rho = np.array([0., .49, .9])
        self.save_table(rho, rho * -.4, rho * .3)
        e = TrinityEngine(str(self.ascii_case()))
        e.evolve_profiles()
        self.check_frame(self.root / "from_ascii.nc", self.midpoints * -.4, self.midpoints * .3)

    def test_ascii_can_sum_distinct_flux_columns_in_species_binding(self):
        values = np.column_stack((self.midpoints, 1e20*self.particles, .25*self.heat, .75*self.heat))
        np.savetxt(self.table, values)
        case = self.make_case(
            "sum", self.table, 'type = "columns"\ncolumns = ["rho", "Gam_e", "conductive", "enthalpy"]\n'
            'divide_by = [1.0, 1e20, 1.0, 1.0]',
        )
        case.write_text(case.read_text().replace('keys = "Q_H"', 'keys = ["conductive", "enthalpy"]'))
        e = TrinityEngine(str(case))
        e.evolve_profiles()
        self.check_frame(self.root / "sum.nc", self.particles, self.heat)

    def test_table_source_and_flux_can_share_import_and_add_a_model(self):
        rho = np.r_[0., self.midpoints, self.rho_edge]
        particles = np.r_[0., self.particles, self.particles[-1]]
        heat = np.r_[0., self.heat, self.heat[-1]]
        power_density = .015 - .01*rho
        np.savetxt(self.table, np.column_stack((rho, 1e20*particles, heat, power_density)))
        case = self.ascii_case()
        config = case.read_text().replace(
            'model = "Import"\n', 'model = "Import"\n\n[[model]]\nmodel = "ReLU"\n'
        ).replace(
            'heat_flux = { import = true, keys = "Q_H" }',
            'heat_flux = { import = true, keys = "Q_H" }\n'
            'aux_power_source = { import = true, keys = "Sp_H" }\n'
            'pressure_relu_flux = { critical_gradient = 0.0, slope = 0.0001 }',
        ).replace(
            'type = "electron"\n', 'type = "electron"\ndensity_relu_flux = { slope = 0.0 }\n',
        ).replace(
            '"Gam_e", "Q_H"]', '"Gam_e", "Q_H", "Sp_H"]',
        ).replace(
            '[1.0, 1e20, 1.0]', '[1.0, 1e20, 1.0, 1.0]',
        )
        case.write_text(config)
        e = TrinityEngine(str(case))
        e.evolve_profiles()
        with Dataset(self.root / "from_ascii.nc") as result:
            diag = result["species"]
            np.testing.assert_allclose(diag["Q_Import_MW_H"][-1], self.heat, atol=1e-15)
            relu = diag["Q_ReLU_MW_H"][-1]
            self.assertGreater(np.max(abs(relu)), 1e-10)
            np.testing.assert_allclose(diag["Q_MW_H"][-1], self.heat + relu, atol=1e-14)
            np.testing.assert_allclose(diag["Sp_aux_H"][-1, :-1] * e.norms.P_ref_MWm3,
                                       .015 - .01*e.grid.rho[:-1], atol=1e-15)

            # Diagnostic gyro-Bohm totals should include the dimensional input
            # once, without altering the online component in either output.
            q_gb = diag["qflux_Import_H"][-1]
            gb_to_MW = 1.5 * e.norms.P_ref_MWm3 * e.norms.a_meters**3
            ref = e.species.ref_species
            face_scale = e.species["hydrogen"]._heat_face_factor(ref, e.geometry).profile
            np.testing.assert_allclose(q_gb * face_scale * gb_to_MW, self.heat, atol=1e-15)
            np.testing.assert_allclose(diag["qflux_H"][-1],
                                       q_gb + diag["qflux_ReLU_H"][-1], atol=1e-14)

    def test_ascii_rejects_bad_grid_axis_or_values(self):
        duplicate = np.array([.2, .4, .4, .8])
        uncovered = np.linspace(self.midpoints[0] + .01, self.midpoints[-1], self.n_radial-1)
        with_axis = np.r_[0., self.midpoints]
        for name, rho, particles, heat, message in (
            ("duplicate radius", duplicate, self.particles, self.heat, "strictly increasing"),
            ("out of coverage", uncovered, self.particles, self.heat, "cover|outside|extrapolat"),
            ("axis flow", with_axis, np.r_[.01, self.particles], np.r_[0., self.heat], "axis"),
            ("nonfinite power", self.midpoints, self.particles, [0., 0., np.nan, 0.],
             "finite"),
        ):
            with self.subTest(name=name):
                self.save_table(rho, particles, heat)
                with self.assertRaisesRegex(ValueError, message):
                    e = TrinityEngine(str(self.ascii_case()))
                    e._evaluate_transport_terms()

    def test_ascii_requires_explicit_radial_column(self):
        np.savetxt(self.table, np.column_stack((1e20*self.particles, self.heat)))
        case = self.make_case(
            "implicit_rho", self.table, 'type = "columns"\ncolumns = ["Gam_e", "Q_H"]\n'
            'divide_by = [1e20, 1.0]',
        )
        with self.assertRaisesRegex(ValueError, "rho|radial"):
            e = TrinityEngine(str(case))
            e._evaluate_transport_terms()

    def test_model_is_required_exactly_once_for_the_shared_import(self):
        self.save_table()
        case = self.ascii_case()
        initial = case.read_text()
        for kind, replacement, match in (
            ("missing", 'model = "ReLU"', "requires.*model"),
            ("duplicate", 'model = "Import"\n\n[[model]]\nmodel = "import"\n'
             'label = "SecondImport"', "only once"),
        ):
            with self.subTest(kind=kind):
                case.write_text(initial.replace('model = "Import"', replacement))
                with self.assertRaisesRegex(ValueError, match):
                    TrinityEngine(str(case))

    def test_nonzero_flux_for_dependent_density_is_not_silently_applied(self):
        self.save_table()
        case = self.ascii_case()
        case.write_text(case.read_text()
                        .replace('particle_flux = { import = true, keys = "Gam_e" }\n', '')
                        .replace('qneut_species = true',
                                 'qneut_species = true\nparticle_flux = '
                                 '{ import = true, keys = "Gam_e" }'))
        with self.assertRaisesRegex(ValueError, "evolv|active|quasineutral|quasi-neutral"):
            e = TrinityEngine(str(case))
            e._evaluate_transport_terms()
