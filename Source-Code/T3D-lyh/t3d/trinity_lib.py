import numpy as np
import sys
import os

from t3d.Grid import Grid
from t3d.Time import Time
from t3d.Species import SpeciesDict
from t3d.Solver import TransportSolver
from t3d.Geometry import Geometry
from t3d.TrinityIO import Input, Logger
from t3d.Physics import Physics
from t3d.FluxModels import FluxModelDict
from t3d.SourceModels import SourceModelDict
from t3d.Import import Import
from t3d.Logbook import info, bold, warn
from t3d.constants import ELEMENTARY_CHARGE
from t3d.constants import SPEED_OF_LIGHT
from t3d.constants import REFERENCE_THERMAL_SPEED
from t3d.constants import REFERENCE_GYRORADIUS

from termcolor import colored


_version = "1.0.0"


def check_for_stop_file(stopfile="stop") -> bool:
    """Check for a stop file."""
    if os.path.isfile(stopfile):
        bold(' Stop file detected, exiting.', color='blue')
        os.remove(stopfile)
        return True
    return False


class TrinityEngine():

    def __init__(self, trinity_input):

        # Parse the input file data, and store in dictionary self.input_dict
        self.inputs = Input(trinity_input)
        self.input_dict = self.inputs.input_dict
        self.trinity_infile = trinity_input  # save input file name
        self.version = _version

        # initialize grid
        self.grid = Grid(self.input_dict)

        # initialize time
        self.time = Time(self.input_dict)

        # initialize flux models
        self.flux_models = FluxModelDict(self.input_dict, self.grid, self.time)

        # initialize source models
        self.source_models = SourceModelDict(self.input_dict, self.grid, self.time)

        # initialize imported profiles, sources, etc
        self.imported = Import(self.input_dict, self.grid)

        # initialize geometry
        self.geometry = Geometry(self.input_dict, self.grid, self.flux_models, self.imported)

        # initialize all species
        self.species = SpeciesDict(self.input_dict, self.grid)

        # initialize Trinity normalizations (not to be confused with FluxModel normalizations)
        self.norms = TrinNormalizations(Btor=self.geometry.Btor, a_meters=self.geometry.a_minor, vt_sqrt_2=self.flux_models.vt_sqrt_2, m_ref=self.species.ref_species.mass)
        self.time.normalize_inputs(self.norms.t_ref)

        # initialize physics
        self.physics = Physics(self.input_dict, self.grid, self.geometry, self.norms, self.imported)

        # set initial profiles
        self.species.init_profiles_and_sources(self.geometry, self.norms, self.imported)
        if self.physics.scale_density_to is not None:
            if self.source_models.get_sources_list():
                raise ValueError("Sudo density control with external heating requires coupled iteration")
            self.physics.scale_to_sudo_density(self.species)
        self.physics.compute_sources(self.species, self.geometry, self.norms)
        self.species.print_info()

        # initialize solver
        self.solver = TransportSolver(self.grid, self.species, self.geometry, self.physics)

        self.solver.balance_closure = any(getattr(model, "after_sources", False)
                                          for model in self.flux_models.get_models_list())
        if self.solver.balance_closure and (len(self.flux_models.get_models_list()) != 1
                                           or self.physics.turbulent_exchange):
            raise ValueError("PowerBalance is a standalone source-balance closure, not an additive flux model")

        # initialize logger (it is important that this comes last, after all the other initializations)
        self.logger = Logger(self.trinity_infile, self.input_dict, self.grid, self.time, self.species, self.geometry, self.physics, self.norms, self.flux_models, source_models=self.source_models)

        # flush output
        sys.stdout.flush()

    # End of __init__ function

    def _evaluate_transport_terms(self):
        models = tuple(self.flux_models.get_models_list())
        sources = tuple(self.source_models.get_sources_list())
        try:
            self._compute_transport_terms(models, sources)
        except BaseException:
            for backend in (*models, *sources):
                cancel = getattr(backend, "cancel_pending", None)
                if cancel is not None:
                    try:
                        cancel()
                    except Exception as error:
                        warn(f"Could not clean pending {backend.label} jobs: {error}")
            raise

    def _compute_transport_terms(self, models, sources):
        self.species.clear_fluxes()
        self.species.clear_sources()
        self.solver.external_source_jacobian = None
        for model in models:
            if not getattr(model, "after_sources", False):
                info(f" computing {model.label} fluxes", color="green")
                model.compute_fluxes(self.species, self.geometry, self.norms)
        for source in sources:
            info(f" computing {source.label} sources", color="magenta")
            source.compute_sources(self.species, self.geometry, self.norms)
        for model in models:
            if not getattr(model, "after_sources", False):
                model.collect_results(self.species, self.geometry, self.norms)

        self.physics.compute_sources(self.species, self.geometry, self.norms)
        for source in sources:
            source.collect_results(self.species, self.geometry, self.norms)
            jac = getattr(source, "source_jacobian", None)
            if jac is not None:
                if self.solver.external_source_jacobian is None:
                    self.solver.external_source_jacobian = jac.copy()
                else:
                    self.solver.external_source_jacobian += jac
        for model in models:
            if getattr(model, "after_sources", False):
                model.compute_fluxes(self.species, self.geometry, self.norms)
                model.collect_results(self.species, self.geometry, self.norms)
        self.time.diagnostics_current = True

    def _refresh_final_diagnostics(self):
        if self.time.diagnostics_current:
            return
        # A diagnostic row is not a new transport take. Do not submit an
        # external job or alter a lagged/cached closure solely for logging.
        immediate_models = {"t3d.flux_models.ReLU", "t3d.flux_models.ChangHintonNeoclassical",
                            "t3d.flux_models.RR", "t3d.flux_models.PowerBalance"}
        if (self.source_models.get_sources_list()
                or any(type(m).__module__ not in immediate_models
                       for m in self.flux_models.get_models_list())):
            info("Final flux/source diagnostics not evaluated at the final profiles; see t_diagnostics_valid.")
            return
        self._evaluate_transport_terms()

    def evolve_profiles(self):
        evaluated = False
        retries = 0
        # A run with prescribed n and T is meaningful as a diagnostic, but
        # there is no unknown to time advance or send to the transport solver.
        stopped = check_for_stop_file("stop")
        if not self.solver.N_profiles:
            info("No evolved profiles; writing a diagnostic snapshot only.")

        while (not stopped and self.solver.N_profiles
               and self.time.step_idx < self.time.N_steps and self.time.time < self.time.t_max):

            info("")

            if self.time.iter_idx == 0:
                end = self.time.time + self.time.dtau
                end_tol = 32*np.finfo(float).eps*max(abs(end), abs(self.time.t_max))
                if end >= self.time.t_max - end_tol:
                    self.time.dtau = self.time.t_max - self.time.time
                    info(f"\n{'':>13}*** Clipping timestep to end at max time: dtau = {colored(f'{self.time.dtau:.3e}', 'blue')}. ***")

                if self.time.time + self.time.dtau <= self.time.time:
                    raise RuntimeError("Transport timestep is too small to advance time")

            if not self.species.profiles_are_physical():
                raise RuntimeError("Cannot evaluate transport at nonpositive or nonfinite density/pressure")
            if self.geometry.evolve_equilibrium and self.geometry.update_equilibrium(self.species, self.time):
                self.solver.reset = False
                self.time.diagnostics_current = False

            self._evaluate_transport_terms()
            evaluated = True

            # solve transport equations
            self.solver.solve(self.species, self.time)
            rms = self.time.rms

            if self.time.iter_idx == 0:
                info(f" iterate = {self.time.iter_idx:>2} " +
                     f"time index = {self.time.step_idx:>2} " +
                     f"normalized time = {self.time.time:.3e}, " +
                     f"normalized dt = {self.time.dtau:.3f}, " +
                     f"rms = {rms:.3e}, " +
                     f"time = {self.time.time*self.norms.t_ref:.3e} (s)")
            else:
                info(f" iterate = {self.time.iter_idx:>2}{'':>69}" +
                     f"rms = {rms:.3e}", 'blue')

            # info(f"time = {self.time.time*self.norms.t_ref:.3e} (s),  time_index = {self.time.step_idx}, iteration index = {self.time.iter_idx}, dtau = {self.time.dtau}, rms = {rms:.3e}")
            # info(f"time = {self.time.time:.3e}*t_ref, time index = {self.time.step_idx}, iteration index = {self.time.iter_idx}, dtau = {self.time.dtau}, rms = {rms:.3e}")

            self.logger.save(self.species, self.solver, self.time)

            finite = np.isfinite(rms)
            if (finite and self.time.iter_idx == 0
                    and self.time.steady_rms < self.time.rms_converged):
                info(f"Transport residual is below rms_converged: {self.time.steady_rms:.3e} "
                     "per unit of normalized time.")
                break

            if finite and (rms < self.time.newton_threshold
                           or (self.time.iter_idx == self.time.max_newton_iter-1 and rms < self.time.newton_tolerance)
                           or self.time.alpha == 0):
                self.time.prev_step_success = True
                self.time.arrival_order = 2 if self.time.dm1(self.time.step_idx) != 0 else 1
                self.time.step_idx += 1
                self.time.time += self.time.dtau
                self.time.dtau_previous_accepted = self.time.dtau
                retries = 0
                self.time.retry_generation = 0
                if self.time.iter_idx > 0:
                    # if prev time step halved, do NOT update dtau_success because the halving skipped GX
                    # if prev time was not halved, there is no need to update dtau_success
                    self.time.dtau_success = self.time.dtau
                self.time.iter_idx = 0
                self.solver.advance(self.species)
                self.time.diagnostics_current = False

                changed_profiles = False
                if self.physics.update_edge:
                    changed_profiles = bool(self.species.update_edge(self.physics))
                if self.physics.update_source:
                    self.species.update_source(self.physics, self.geometry)
                if self.physics.scale_density_to is not None:
                    changed_profiles |= self.physics.scale_to_sudo_density(self.species)
                if changed_profiles:
                    # An imposed profile jump is not a BDF time derivative.
                    # Synchronize the actual state and restart at first order.
                    self.solver.y_curr = self.species.get_vec_from_profs()
                    self.time.dtau_previous_accepted = None
                    self.time.arrival_order = 0

                if rms < self.time.dtau_increase_threshold and self.time.step_idx < self.time.N_steps and self.time.alpha > 0:
                    self.time.dtau_old = self.time.dtau
                    self.time.dtau = min(2.0*self.time.dtau, self.time.dtau_max)
                    info(f"\n{'':>13}*** Increasing timestep at time index {self.time.step_idx} ({colored(f'{rms:.3e}', 'blue')} < {self.time.dtau_increase_threshold:.3e}). ***")

            elif not finite or self.time.iter_idx == self.time.max_newton_iter-1:
                info(f"Transport solve rejected at iteration {self.time.iter_idx} (rms = {rms:.3e}).")
                self.solver.reset_timestep(self.species, self.time)
                self.time.diagnostics_current = False
                retries += 1
                self.time.retry_generation += 1
                if retries > self.time.max_timestep_retries:
                    raise RuntimeError(
                        f"Transport solve failed at t={self.time.time:.3e}; "
                        f"exhausted {self.time.max_timestep_retries} timestep retries")

            else:
                # iteration failed, trinity tries another iteration
                self.time.prev_step_success = False
                self.time.iter_idx += 1
                self.solver.advance(self.species)
                self.time.diagnostics_current = False

            if (self.time.iter_idx == 0 and self.time.prev_step_success
                    and self.time.retry_generation == 0
                    and self.time.step_idx % self.logger.n_write == 0):
                # save every n_write trinity steps
                self.logger.export()
                self.logger.write_netCDF4(self.trinity_infile)
                self.logger.write_adios2(self.trinity_infile)
            sys.stdout.flush()

            if check_for_stop_file("stop"):
                if self.time.iter_idx:
                    self.solver.y_curr = self.solver.y_m
                    self.species.get_profs_from_vec(self.solver.y_curr)
                    self.time.iter_idx = 0
                    self.time.diagnostics_current = False
                break

        if evaluated:
            info(f"\n{'':>14}time index = {self.time.step_idx:>2} " +
                 f"normalized time = {self.time.time:.3e}, " +
                 f"normalized dt = {self.time.dtau:.3f}, rms = {self.time.rms:.3e}, " +
                 f"time = {self.time.time*self.norms.t_ref:.3e} (s)")

            info("\n Time evolution finished!")
            # info(f"\n Time evolution finished! time = {self.time.time:.3e}*t_ref, time index = {self.time.step_idx}")
            for model in self.flux_models.get_models_list():
                model.final_info()

        self._refresh_final_diagnostics()
        self.logger.save(self.species, self.solver, self.time, final=True)
        self.logger.export()
        self.logger.write_netCDF4(self.trinity_infile)
        self.logger.write_adios2(self.trinity_infile)


# a class for handling normalizations in Trinity
class TrinNormalizations():
    def __init__(self,
                 Btor=1,      # T
                 m_ref=2.0,   # units of proton mass
                 a_meters=1,  # minor radius, in m
                 vt_sqrt_2=False):

        info("  Initializing normalizations")
        info(f"    Ba      = {Btor:.2f} T")
        info(f"    a_minor = {a_meters:.2f} m")
        info(f"    m_ref   = {m_ref:.2f} m_p")

        self.e = ELEMENTARY_CHARGE  # in Coulombs
        self.c = SPEED_OF_LIGHT     # in m/s

        if vt_sqrt_2:
            gam = 2
        else:
            gam = 1

        # reference gyroradius in [m], assuming
        # B_ref = 1 and T_ref = 1 keV
        self.rho_ref = REFERENCE_GYRORADIUS * np.sqrt(gam * m_ref)  # m

        # reference thermal speed in [m/s], assuming
        # T_ref = 1 keV
        vT_ref = REFERENCE_THERMAL_SPEED * np.sqrt(gam / m_ref)  # m/s

        t_ref = a_meters / vT_ref
        p_ref = 1e20 * 1e3 * self.e  # J/m^3
        rho_star = self.rho_ref / a_meters

        # multiply trinity time (tau) by this to get SI (seconds)
        t_ref = t_ref * rho_star**-2

        # multiply SI (W/m^3) by pressure_source_scale to get Trinity units
        pressure_source_scale = t_ref / p_ref
        P_ref_MWm3 = 1 / pressure_source_scale / 1e6  # MW/m^3

        # multiply SI (10^20 m^-3 s^-1) by particle_source_scale to get Trinity units
        particle_source_scale = t_ref
        Sn_ref_SI20 = 1 / particle_source_scale  # 10^20 m^-3 s^-1

        # info(f"    a_minor = {a_meters:.2f} m")
        info(f"    t_ref = {t_ref:.3f} s")    # deprecated report, to be fixed in the next round.
        # info(f"    P_ref = {P_ref_MWm3:.3f} MW/m^3")
        # info(f"    S_ref = {Sn_ref_SI20:.3f} 10^20/(m^3 s)")
        info(f"    rho_star(T=1keV) = {rho_star / Btor:.3e}\n")

        # save
        self.Btor = Btor
        self.a_meters = a_meters

        self.vT_ref = vT_ref
        self.vtfac = gam
        self.t_ref = t_ref
        self.p_ref = p_ref
        self.rho_star = rho_star
        self.P_ref_MWm3 = P_ref_MWm3
        self.Sn_ref_SI20 = Sn_ref_SI20
