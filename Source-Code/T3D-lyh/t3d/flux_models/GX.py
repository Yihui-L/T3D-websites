import numpy as np
from scipy.integrate import trapezoid
import subprocess
from datetime import datetime
import time as _time
from t3d import Profiles as pf
import os
import re
import copy
import sys
from t3d.FluxModels import FluxModel
from netCDF4 import Dataset
import math
from t3d.Logbook import logbook
from t3d.flux_models._external import (
    cancel_jobs, command_prefix, run_workspace, start_job, trial_tag, valid_output,
)

def map_to_periodic_range(arr, low_bound, high_bound):
    range_size = high_bound - low_bound
    mapped_arr = ((arr - low_bound) % range_size) + low_bound
    return mapped_arr


class GX_FluxModel(FluxModel):

    def __init__(self, pars, grid, time):

        # call base class constructor
        super().__init__(pars, grid, time)

        # environment variable containing path to gx executable
        GX_PATH = pars.get("gx_path", os.environ.get("GX_PATH") or "")
        self.gx_app = os.path.join(GX_PATH,"gx")

        gx_template = pars.get('gx_template', 'tests/regression/gx_template.in')
        out_dir = run_workspace(pars.get('gx_outputs', 'gx/'))
        self.out_dir = out_dir
        self.logfile = pars.get('gx_logfile', 'gx.log')
        self.overwrite = pars.get('overwrite', False)
        self.gpus_per_gx = pars.get('gpus_per_gx', 1)
        self.gx_max_concurrent = pars.get('gx_max_concurrent')
        if type(self.gpus_per_gx) is not int or self.gpus_per_gx <= 0:
            raise ValueError("gpus_per_gx must be a positive integer")
        if self.gx_max_concurrent is not None and (
                type(self.gx_max_concurrent) is not int or self.gx_max_concurrent <= 0):
            raise ValueError("gx_max_concurrent must be a positive integer")
        # force GX to use the electrostatic limit (regardless of beta in T3D) in GX
        self.electromagnetic = pars.get('electromagnetic', True)
        # list of zeta angles about which flux tubes are centered (for stellarator calculations only)
        self.zeta_centers = pars.get('zeta_center', [0.0])
        if not hasattr(self.zeta_centers, '__len__'):  # allow a scalar input instead of a list
            self.zeta_centers = [self.zeta_centers]
        if np.asarray(self.zeta_centers).ndim != 1 or len(self.zeta_centers) == 0 or not np.all(np.isfinite(self.zeta_centers)):
            raise ValueError("GX requires at least one finite zeta_center")
        # interpolate fluxes on theta-zeta grid (stellarator calculations only)
        self.theta_zeta_fluxes = pars.get('theta_zeta_fluxes', False)
        # specify a species should be treated adiabatically by GX
        self.adiabatic_species = pars.get('adiabatic_species', None)
        # scale the GX collisionality by a factor
        self.collisionality_scaling_factor = pars.get('collisionality_scaling_factor', 1.0)
        # scale qfluxes by an artificial factor (useful for scoping)
        self.qflux_scaling_factor = pars.get('qflux_scaling_factor', 1.0)
        # shift tprim used in GX away from nominal tprim by a constant (useful for scoping)
        self.tprim_shift = pars.get('tprim_shift', 0.0)
        # specify some species temperatures that do not need to be perturbed (e.g. don't perturb ion temperatures for ETG calculations)
        self.no_perturb_temperature = pars.get('no_perturb_temperature', [])
        if not hasattr(self.no_perturb_temperature, '__len__'):  # allow a scalar input instead of a list
            self.no_perturb_temperature = [self.no_perturb_temperature]
        # parameter to set perturbation size for gradients
        self.dkap_T = pars.get('dkap_T', 0.5)
        self.dkap_n = pars.get('dkap_n', 0.5)
        self.label = pars.get('label', 'GX')  # label for diagnostics
        # check for stalled GX runs. should be set False when there are not enough GPUs
        # to cover all the flux tubes in the T3D calculation
        self.check_stalls = pars.get('check_stalls', False)
        self.monitor_time = pars.get('monitor_time', 10)  # check on GX output every [monitor_time] seconds
        self.stall_abort_count = pars.get('stall_abort_count', 6)  # abort after this many stall monitors
        self.build_ReLU = pars.get('build_ReLU', False)  # use GX runs to build a ReLU model at each radius
        self.nsteps_ReLU = pars.get('nsteps_ReLU', 1000)  # take [nsteps_ReLU] ReLU steps before re-running GX
        if self.build_ReLU and (type(self.nsteps_ReLU) is not int or self.nsteps_ReLU < 1):
            raise ValueError("nsteps_ReLU must be a positive integer when build_ReLU is enabled")
        if self.check_stalls and (self.monitor_time <= 0 or self.stall_abort_count <= 0):
            raise ValueError("GX stall polling interval and count must be positive")

        # use a non-default run command (e.g. "srun ...")
        # full command line will be cmd = f"{self.run_command} {gx_app} {f_in}"
        self.run_command = pars.get('run_command', None)

        self.abort_on_nans = pars.get('abort_on_nans', True)
        self.no_restarts = pars.get('no_restarts', False)

        self.dry_run = pars.get('dry_run', False)

        self.vt_sqrt_2 = False  # GX normalizations use vt = sqrt(T/m), without the sqrt(2)
        self.target_chie_over_chii = pars.get('chie_over_chii', 0)  # if non-zero, compute Qe from Qi by using chi ratio

        # Create output directory if not found
        found_path = os.path.exists(out_dir)
        if not found_path:
            os.mkdir(out_dir)

        # Create logger
        self.log = logbook()
        self.log.set_handlers(term_stream=False,
                              file_stream=True,
                              file_handler=os.path.join(self.out_dir, self.logfile))
        self.info = self.log.info  # Short-cut command

        # Check file path
        self.info("  Looking for GX files")
        self.info(f"    Expecting GX template: {gx_template}")
        self.info(f"    Expecting GX executable: {self.gx_app}")
        self.info(f"    GX-Trinity output path: {out_dir}")

        if not self.dry_run:
            found_gx = os.path.isfile(self.gx_app) and os.access(self.gx_app, os.X_OK)
            if not found_gx:
                self.info("ERROR: gx executable not found! Make sure the GX_PATH environment variable is set.")
                raise RuntimeError('GX not found')

        self.info("")

        # load an input template
        self.read_input(gx_template)
        try:
            self.nstep_gx = self.inputs['Time']['nstep']
        except:
            self.nstep_gx = None

        self.processes = []
        self.call_counter = 0
        self.logs = []
        self.gx_index = 0
        self._progress = {}
        self._restart_outputs = {}
        self._restart_geometry = None
        self._gx_pending_geometry = None

        self.use_ReLU = False
        if self.build_ReLU:
            self.dpflux_dkn_sj = {}
            self.dqflux_dkn_sj = {}
            self.dheat_dkn_sj = {}
            self.dpflux_dkT_sj = {}
            self.dqflux_dkT_sj = {}
            self.dheat_dkT_sj = {}

        self._configure_devices()

    def _configure_devices(self):
        self.gk_system = os.environ.get('GK_SYSTEM', 'local')
        clusters = ('stellar', 'traverse', 'satori', 'perlmutter', 'polaris', 'summit')
        self.gpu_groups = []
        if self.dry_run:
            self.workers = 1
            return
        if self.gk_system == 'polaris' and self.run_command is None:
            if self.gpus_per_gx > 4:
                raise ValueError("Polaris built-in GX launcher supports at most 4 GPUs per job; use run_command for larger jobs")
            nodefile = os.environ.get("PBS_NODEFILE")
            if nodefile is None:
                raise RuntimeError("PBS_NODEFILE is required for the Polaris GX launcher")
            with open(nodefile) as nodes:
                hostnames = list(dict.fromkeys(line.strip() for line in nodes if line.strip()))
            for i, host in enumerate(hostnames):
                hostfile = os.path.join(self.out_dir, f'local_hostfile.{i+1:02}')
                with open(hostfile, 'w') as f:
                    print(host, file=f)
                for start in range(0, 5-self.gpus_per_gx, self.gpus_per_gx):
                    devices = ','.join(str(j) for j in range(start, start+self.gpus_per_gx))
                    self.gpu_groups.append((hostfile, devices))
        elif self.gk_system not in clusters and self.run_command is None:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible is None:
                try:
                    result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                            capture_output=True, text=True, check=True)
                    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
                except (OSError, subprocess.CalledProcessError):
                    devices = []
            else:
                devices = [d.strip() for d in visible.split(',') if d.strip()]
                if "-1" in devices:
                    devices = devices[:devices.index("-1")]
            for i in range(0, len(devices)-self.gpus_per_gx+1, self.gpus_per_gx):
                self.gpu_groups.append((None, ','.join(devices[i:i+self.gpus_per_gx])))
        if self.run_command is not None or (self.gk_system in clusters and self.gk_system != 'polaris'):
            self.workers = self.gx_max_concurrent or 1
        elif self.gpu_groups:
            self.workers = min(len(self.gpu_groups), self.gx_max_concurrent or len(self.gpu_groups))
        else:
            raise RuntimeError("Allocated/visible GPUs cannot cover one GX case; set CUDA_VISIBLE_DEVICES or configure run_command")

    def __del__(self):
        ''' Use destructor to close the log file '''
        self.log.finalize()

    def logfile_info(self) -> str:
        ''' Return string of log file location '''
        return self.log.file_handler.name

    def compute_fluxes(self, species, geo, norms):
        geometry = self._geometry_token(geo)
        if (self.build_ReLU and self.time.iter_idx == 0
                and (self.time.step_idx % self.nsteps_ReLU == 0
                     or getattr(self.time, "retry_generation", 0) != 0)):
            self.use_ReLU = False
        if geometry != self._restart_geometry:
            self.use_ReLU = False
        if self.use_ReLU:
            self.compute_fluxes_ReLU(species, geo, norms)
        else:
            self.compute_fluxes_gx(species, geo, norms)

    def _perturbation_keys(self, species):
        # Adiabatic is a choice of GX kinetic representation, not the
        # physical closure used for charge/equal-to constraints in T3D.
        return (list(species.n_evolve_keys),
                [key for key in species.T_evolve_keys if key not in self.no_perturb_temperature])

    def compute_fluxes_gx(self, species, geo, norms):
        self.gx_index = 0
        self.outs_jk = []
        self.dkap_jk = []
        self._gx_pending_geometry = self._geometry_token(geo)
        if species.has_adiabatic_species:
            self.adiabatic_species = species.adiabatic_species.type

        n_keys, T_keys = GX_FluxModel._perturbation_keys(self, species)
        probes = [(None, None, None)] + [(key, None, self.dkap_n) for key in n_keys]
        probes += [(None, key, self.dkap_T) for key in T_keys]
        for j, (pert_n, pert_T, step) in enumerate(probes):
            kn, kT, kp, dkap = species.get_grads_on_flux_grid(pert_n=pert_n, pert_T=pert_T, abs_step=step)
            if j == 0 and self.build_ReLU:
                self.kn0_sj, self.kT0_sj, self.kp0_sj = kn.copy(), kT.copy(), kp.copy()
            self.dkap_jk.append(dkap)
            self.outs_jk.append(self.run_gx_fluxtubes(self.rho, kn, kT, species, species, geo, j))

    @staticmethod
    def _geometry_token(geo):
        path = getattr(geo, "geo_file", None)
        try:
            stat = os.stat(path)
            return (geo.geo_option, os.path.abspath(path), stat.st_ino, stat.st_size, stat.st_mtime_ns)
        except (OSError, TypeError):
            # Without a stable geometry file, do not offer a warm restart.
            return None

    def compute_fluxes_ReLU(self, species, geo, norms):
        n_keys, T_keys = GX_FluxModel._perturbation_keys(self, species)
        kn_sj, _, kp_sj, _ = species.get_grads_on_flux_grid(pert_n=None, pert_T=None)
        gradient_row = {s.type: i for i, s in enumerate(species.get_species_list())}
        pflux_sj = self.pflux0_sj.copy()
        qflux_sj = self.qflux0_sj.copy()
        heat_sj = self.heat0_sj.copy()
        dpflux_dkn_sj = copy.deepcopy(self.dpflux_dkn_sj)
        dpflux_dkT_sj = copy.deepcopy(self.dpflux_dkT_sj)
        dqflux_dkn_sj = copy.deepcopy(self.dqflux_dkn_sj)
        dqflux_dkT_sj = copy.deepcopy(self.dqflux_dkT_sj)
        dheat_dkn_sj = copy.deepcopy(self.dheat_dkn_sj)
        dheat_dkT_sj = copy.deepcopy(self.dheat_dkT_sj)
        for stype in n_keys:
            row = gradient_row[stype]
            delta = kn_sj[row, :] - self.kn0_sj[row, :]
            pflux_sj += dpflux_dkn_sj[stype]*delta
            heat_sj += dheat_dkn_sj[stype]*delta
            delta_q = dqflux_dkn_sj[stype]*delta
            qflux_sj += np.maximum(delta_q, 0)
            dqflux_dkn_sj[stype][delta_q < 0] = 0.0

        # The density finite difference holds kappa_p fixed. The temperature
        # finite difference holds kappa_n fixed, so its step is also delta_kappa_p.
        for stype in T_keys:
            row = gradient_row[stype]
            delta = kp_sj[row, :] - self.kp0_sj[row, :]
            pflux_sj += dpflux_dkT_sj[stype]*delta
            heat_sj += dheat_dkT_sj[stype]*delta
            delta_q = dqflux_dkT_sj[stype]*delta
            qflux_sj += np.maximum(delta_q, 0)
            dqflux_dkT_sj[stype][delta_q < 0] = 0.0

        floor_mask = qflux_sj < 0
        qflux_sj[floor_mask] = 1e-16
        for derivatives in (dqflux_dkn_sj, dqflux_dkT_sj):
            for slope in derivatives.values():
                slope[floor_mask] = 0.0
        for i, s in enumerate(species.get_species_list()):
            if s.type == self.adiabatic_species:
                pflux_sj[i, :] = heat_sj[i, :] = 0.0
                for derivatives in (dpflux_dkn_sj, dpflux_dkT_sj, dheat_dkn_sj, dheat_dkT_sj):
                    for slope in derivatives.values():
                        slope[i, :] = 0.0
                # A nonzero chi_e/chi_i closure can explicitly supply Q
                # for the adiabatic species in the full GX branch.
                if not self.target_chie_over_chii:
                    qflux_sj[i, :] = 0.0
                    for derivatives in (dqflux_dkn_sj, dqflux_dkT_sj):
                        for slope in derivatives.values():
                            slope[i, :] = 0.0

        # Turbulent exchange can heat or cool a species; do not clamp H.
        species.add_flux(pflux_sj, qflux_sj, heat_sj, label=self.label)
        self.info(f'\n  qflux_GX_ReLU = {qflux_sj}\n')
        self.info(f'\n  pflux_GX_ReLU = {pflux_sj}\n')
        for stype in n_keys:
            species.add_dflux_dkn(stype, dpflux_dkn_sj[stype], dqflux_dkn_sj[stype], dheat_dkn_sj[stype])
        for stype in T_keys:
            species.add_dflux_dkT(stype, dpflux_dkT_sj[stype], dqflux_dkT_sj[stype], dheat_dkT_sj[stype])

    def collect_results(self, species, geo, norms):

        if self.use_ReLU:
            return

        self.wait()
        for stems in self.outs_jk:
            self._require_gx_outputs(stems)

        n_keys, T_keys = GX_FluxModel._perturbation_keys(self, species)

        # read GX output data and pass results to species
        # base
        pert_id = 0
        pflux0_sj, qflux0_sj, heat0_sj = self.read_gx_fluxes(self.outs_jk[pert_id], species.get_species_list())
        species.add_flux(pflux0_sj, qflux0_sj, heat0_sj, label=self.label)
        if self.build_ReLU:
            self.pflux0_sj = pflux0_sj
            self.qflux0_sj = qflux0_sj
            self.heat0_sj = heat0_sj

        # save qflux value for use when deciding whether to use a restart for GX
        self.qflux_prev = qflux0_sj.copy()
        pert_id = pert_id + 1

        # perturbed density cases
        for stype in n_keys:
            dkn_j = self.dkap_jk[pert_id]
            pflux_sj, qflux_sj, heat_sj = self.read_gx_fluxes(self.outs_jk[pert_id], species.get_species_list())
            dpflux_dkn_sj = (pflux_sj - pflux0_sj) / dkn_j
            dqflux_dkn_sj = (qflux_sj - qflux0_sj) / dkn_j
            dheat_dkn_sj = (heat_sj - heat0_sj) / dkn_j
            species.add_dflux_dkn(stype, dpflux_dkn_sj, dqflux_dkn_sj, dheat_dkn_sj)
            if self.build_ReLU:
                self.dpflux_dkn_sj[stype] = dpflux_dkn_sj
                self.dqflux_dkn_sj[stype] = dqflux_dkn_sj
                self.dheat_dkn_sj[stype] = dheat_dkn_sj
            pert_id = pert_id + 1

        # perturbed temperature cases
        for stype in T_keys:
            dkT_j = self.dkap_jk[pert_id]
            pflux_sj, qflux_sj, heat_sj = self.read_gx_fluxes(self.outs_jk[pert_id], species.get_species_list())
            dpflux_dkT_sj = (pflux_sj - pflux0_sj) / dkT_j
            dqflux_dkT_sj = (qflux_sj - qflux0_sj) / dkT_j
            dheat_dkT_sj = (heat_sj - heat0_sj) / dkT_j
            species.add_dflux_dkT(stype, dpflux_dkT_sj, dqflux_dkT_sj, dheat_dkT_sj)
            if self.build_ReLU:
                self.dpflux_dkT_sj[stype] = dpflux_dkT_sj
                self.dqflux_dkT_sj[stype] = dqflux_dkT_sj
                self.dheat_dkT_sj[stype] = dheat_dkT_sj
            pert_id = pert_id + 1

        # Results have been checked and parsed. A restart is only a seed
        # for a fresh run, never a substitute for new transport flux.
        self._restart_outputs = {
            (j, slot): stem for j, stems in enumerate(self.outs_jk)
            for slot, stem in enumerate(stems)
        }
        self._restart_geometry = self._gx_pending_geometry

        # ReLU data is now built, so turn it on
        if self.build_ReLU:
            self.use_ReLU = True

    def _wait_one_gx(self, process, log):
        if not self.check_stalls:
            code = process.wait()
        else:
            # Trust process status, not a text sentinel. Watch progress only
            # when explicitly enabled, since scheduler queue waits are silent.
            while (code := process.poll()) is None:
                now = _time.monotonic()
                size = os.path.getsize(log)
                last_size, last_update = self._progress.setdefault(log, (size, now))
                if size != last_size:
                    self._progress[log] = (size, now)
                elif now-last_update >= self.monitor_time * self.stall_abort_count:
                    raise RuntimeError(f"{self.label} stopped writing progress: {log}")
                _time.sleep(self.monitor_time)
        if code != 0:
            raise RuntimeError(f"{self.label} exited with status {code}; see {log}")
        self._progress.pop(log, None)

    def _finish_first_gx(self):
        process, log = self.processes[0], self.logs[0]
        self._wait_one_gx(process, log)
        self.processes.pop(0)
        self.logs.pop(0)

    def wait(self):
        try:
            while self.processes:
                self._finish_first_gx()
        except BaseException:
            self.cancel_pending()
            raise

    def cancel_pending(self):
        jobs, self.processes = self.processes, []
        self.logs.clear()
        self._progress.clear()
        cancel_jobs(jobs)

    def _require_gx_outputs(self, stems):
        unready = [stem for stem in stems
                   if not any(valid_output(stem + ext) for ext in (".nc", ".out.nc"))]
        if unready:
            raise RuntimeError(f"{self.label} did not produce complete output: {', '.join(unready)}")

    def read_input(self, fin):

        with open(fin) as f:
            data = f.readlines()

        obj = {}
        header = ''
        for line in data:

            # strip comments
            if line.find('#') > -1:
                end = line.find('#')
                line = line[:end]

            # parse headers
            if line.find('[') == 0:
                header = line.split('[')[1].split(']')[0]
                obj[header] = {}
                continue

            # skip blanks
            if line.find('=') < 0:
                continue

            # store data
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()

            if header == '':
                obj[key] = value
            else:
                obj[header][key] = value

        # check for all possible header types in GX input file,
        # and initialize empty if not found
        try:
            obj['Dimensions']
        except:
            obj['Dimensions'] = {}
        try:
            obj['Domain']
        except:
            obj['Domain'] = {}
        try:
            obj['Time']
        except:
            obj['Time'] = {}
        try:
            obj['Initialization']
        except:
            obj['Initialization'] = {}
        try:
            obj['Restart']
        except:
            obj['Restart'] = {}
        try:
            obj['Dissipation']
        except:
            obj['Dissipation'] = {}
        try:
            obj['Diagnostics']
        except:
            obj['Diagnostics'] = {}
        try:
            obj['Geometry']
        except:
            obj['Geometry'] = {}
        try:
            obj['Physics']
        except:
            obj['Physics'] = {}
        try:
            obj['species']
        except:
            obj['species'] = {}
        try:
            obj['Boltzmann']
        except:
            obj['Boltzmann'] = {}

        self.inputs = obj
        self.filename = fin

    def write_input(self, fout='temp.in', inputs=None):

        if inputs is None:
            inputs = self.inputs
        # Render numpy scalars as ordinary numbers without changing the
        # interpreter's global print options.
        def plain(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, (list, tuple)):
                return [plain(x) for x in value]
            return value

        with open(fout, 'x') as f:
            for item in inputs.items():

                if (type(item[1]) is not dict):
                    print('  %s = %s ' % item, file=f)
                    continue

                header, nest = item
                print('\n[%s]' % header, file=f)

                longest_key = max(nest.keys(), key=len, default='')
                N_space = len(longest_key)
                for pair in nest.items():
                    s = '  {:%i}  =  {}' % N_space
                    print(s.format(pair[0], plain(pair[1])), file=f)

        self.info(f'  wrote input: {fout}')

    # run a GX flux tube calculation at each radius, given gradient values kns and kts
    def run_gx_fluxtubes(self, rho, kns, kts, species, species_all, geo, pert_id, dummy=False):

        trial = trial_tag(self.time)
        gx_inputs = copy.deepcopy(self.inputs)

        if not dummy:
            # handle adiabatic species
            if self.adiabatic_species is not None:
                T_adiab = species_all[self.adiabatic_species].T().toFluxProfile()
                T_ref = species_all.ref_species.T().toFluxProfile()
                tau_fac = T_ref/T_adiab
                if self.adiabatic_species == "electron":
                    adiab_type = "electron"
                else:
                    adiab_type = "ion"
            if self.adiabatic_species is not None and self.adiabatic_species == species.ref_species.type:
                raise ValueError("GX requires a kinetic T3D reference species")
            kinetic = np.array([s.type != self.adiabatic_species for s in species.get_species_list()])
            N_species = int(np.sum(kinetic))
            if N_species == 0:
                raise ValueError("GX requires at least one kinetic species")

            # get profile values on flux grid
            # these are 2d arrays, e.g. ns = ns[species_idx, rho_idx]
            ns, Ts, nu_ss = species.get_profiles_on_flux_grid(normalize=True, a_minor=geo.a_minor)
            beta_ref = species.ref_species.beta(geo.Btor).toFluxProfile()

        # loop over flux tubes, one at each rho
        outs = []
        for r_id in np.arange(self.N_fluxtubes):
            for a_id, zeta in enumerate(self.zeta_centers):

                path = self.out_dir
                tag = f"{trial}-r{r_id}-a{a_id}-j{pert_id}"
                fout = tag + '.in'
                if os.path.exists(os.path.join(path, fout)):
                    raise RuntimeError(f"Duplicate GX trial input: {os.path.join(path, fout)}")
                previous_stem = self._restart_outputs.get((pert_id, r_id*len(self.zeta_centers)+a_id))
                f_load = previous_stem + ".restart.nc" if previous_stem is not None else None

                if geo.geo_option == 'vmec':
                    gx_inputs['Geometry']['geo_option'] = "'vmec'"
                    gx_inputs['Geometry']['desired_normalized_toroidal_flux'] = rho[r_id]**2
                    gx_inputs['Geometry']['zeta_center'] = zeta
                    gx_inputs['Geometry']['vmec_file'] = f"'{geo.geo_file}'"
                elif geo.geo_option == 'desc' and str(geo.geo_file).endswith('.nc'):
                    # A DESC-exported wout is a VMEC-format file, not native DESC HDF5.
                    gx_inputs['Geometry']['geo_option'] = "'vmec'"
                    gx_inputs['Geometry']['desired_normalized_toroidal_flux'] = rho[r_id]**2
                    gx_inputs['Geometry']['vmec_file'] = f"'{geo.geo_file}'"
                    gx_inputs['Geometry']['zeta_center'] = zeta
                elif geo.geo_option == 'desc':
                    gx_inputs['Geometry']['geo_option'] = "'desc'"
                    gx_inputs['Geometry']['geo_file'] = f"'{geo.geo_file}'"
                    gx_inputs['Geometry']['rhotor'] = rho[r_id]
                    gx_inputs['Geometry']['zeta_center'] = zeta
                elif geo.geo_option == 'geqdsk':
                    gx_inputs['Geometry']['geo_option'] = "'gs2_geo'"
                    gx_inputs['Geometry']['rhoc'] = rho[r_id]
                    gx_inputs['Geometry']['iflux'] = 1
                    gx_inputs['Geometry']['bishop'] = 1
                    gx_inputs['Geometry']['equal_arc'] = "true"
                    gx_inputs['Geometry']['efit_eq'] = "true"
                    gx_inputs['Geometry']['eqfile'] = f"'{geo.geo_file}'"
                    if self.grid.flux_label == 'torflux':
                        gx_inputs['Geometry']['irho'] = 1
                    elif self.grid.flux_label == 'rminor':
                        gx_inputs['Geometry']['irho'] = 2
                    gx_inputs['Geometry']['eqfile'] = f"'{geo.geo_file}'"
                elif geo.geo_option == 'miller':
                    gx_inputs['Geometry']['geo_option'] = "'miller'"
                    gx_inputs['Geometry']['rhoc'] = rho[r_id]
                    gx_inputs['Geometry']['Rmaj'] = geo.AspectRatio
                    gx_inputs['Geometry']['R_geo'] = geo.rmaj[r_id]
                    gx_inputs['Geometry']['qinp'] = geo.qsf[r_id]
                    gx_inputs['Geometry']['shat'] = geo.shat[r_id]
                    gx_inputs['Geometry']['shift'] = geo.rmajor_prime[r_id]
                    gx_inputs['Geometry']['akappa'] = geo.kappa[r_id]
                    gx_inputs['Geometry']['akappri'] = geo.kappa_prime[r_id]
                    gx_inputs['Geometry']['tri'] = geo.delta[r_id]
                    gx_inputs['Geometry']['tripri'] = geo.delta_prime[r_id]
                    gx_inputs['Geometry']['betaprim'] = -beta_ref[r_id]*np.sum(ns[:, r_id]*Ts[:, r_id]*(kns[:, r_id]+kts[:, r_id]))

                if dummy:
                    gx_inputs['Time']['nstep'] = 1
                    gx_inputs['Dimensions']['nspecies'] = 1
                    gx_inputs['Restart']['restart'] = 'false'
                    gx_inputs['species']['mass'] = [1.0]
                    gx_inputs['species']['z'] = [1.0]
                    gx_inputs['species']['dens'] = [1.0]
                    gx_inputs['species']['temp'] = [1.0]
                    gx_inputs['species']['fprim'] = [1.0]
                    gx_inputs['species']['tprim'] = [1.0]
                    gx_inputs['species']['vnewk'] = [1.0]
                    gx_inputs['species']['type'] = ['ion']
                    gx_inputs['Boltzmann']['add_Boltzmann_species'] = 'false'
                else:
                    if self.nstep_gx:
                        gx_inputs['Time']['nstep'] = self.nstep_gx
                    else:
                        gx_inputs['Time']['nstep'] = 10000000

                    gx_inputs['Dimensions']['nspecies'] = N_species
                    gx_inputs['Dimensions']['nperiod'] = 1

                    # set reference beta
                    if self.electromagnetic:
                        gx_inputs['Physics']['beta'] = beta_ref[r_id]
                    else:
                        # force beta = 5e-4 to get electrostatic limit in GX to avoid omega_H mode when beta=0 (or fapar=0)
                        # keep actual beta value as comment for reference
                        gx_inputs['Physics']['beta'] = f"5e-4  # {beta_ref[r_id]}"

                    # set species parameters (these are lists)
                    gx_inputs['species']['mass'] = list(species.get_masses(normalize=True)[kinetic])
                    gx_inputs['species']['z'] = list(species.get_charges(normalize=True)[kinetic])
                    gx_inputs['species']['dens'] = list(ns[kinetic, r_id])
                    gx_inputs['species']['temp'] = list(Ts[kinetic, r_id])
                    gx_inputs['species']['fprim'] = list(kns[kinetic, r_id])
                    gx_inputs['species']['tprim'] = list(kts[kinetic, r_id] + self.tprim_shift)
                    gx_inputs['species']['vnewk'] = list(nu_ss[kinetic, r_id]*self.collisionality_scaling_factor)
                    gx_inputs['species']['type'] = list(np.asarray(species.get_types_ion_electron())[kinetic])

                    if self.adiabatic_species is not None:
                        gx_inputs['Boltzmann']['add_Boltzmann_species'] = 'true'
                        gx_inputs['Boltzmann']['tau_fac'] = tau_fac[r_id]
                        gx_inputs['Boltzmann']['Boltzmann_type'] = f"'{adiab_type}'"
                    else:
                        gx_inputs['Boltzmann']['add_Boltzmann_species'] = 'false'

                    restart = (not self.no_restarts
                               and getattr(self.time, 'retry_generation', 0) == 0
                               and self._restart_geometry is not None
                               and self._restart_geometry == self._gx_pending_geometry
                               and f_load is not None and valid_output(f_load)
                               and np.all(np.isfinite(self.qflux_prev[:, r_id]))
                               and np.any(np.abs(self.qflux_prev[:, r_id]) >= 1e-10))
                    gx_inputs['Restart']['restart'] = 'true' if restart else 'false'
                    if restart:
                        gx_inputs['Restart']['restart_from_file'] = f'"{f_load}"'

                    # save restart file (always)
                    gx_inputs['Restart']['save_for_restart'] = 'true'

                # execute
                self.write_input(os.path.join(path, fout), gx_inputs)
                out = self.submit_gx_job(tag, path)  # this returns a file name
                outs.append(out)
            # end zeta_center loop
        # end radius loop
        return outs

    def submit_gx_job(self, tag, path):
        f_tag = os.path.join(path, tag)
        f_in = f_tag + '.in'
        if self.dry_run:
            return f_tag
        if len(self.processes) >= self.workers:
            self._finish_first_gx()
        env = os.environ.copy()
        devices = str(self.gpus_per_gx)
        system = self.gk_system
        if self.run_command is not None:
            prefix = command_prefix(self.run_command)
        elif system == 'polaris':
            hostfile, mask = self.gpu_groups[self.gx_index % self.workers]
            env["CUDA_VISIBLE_DEVICES"] = mask
            prefix = ["mpiexec", "-n", devices, "--ppn", devices, "--depth=8",
                      "--cpu-bind", "core", "--hostfile", hostfile]
        elif system in ('stellar', 'traverse', 'satori', 'perlmutter'):
            # Slurm owns device placement. One rank per allocated GPU.
            per_node = 1 if system in ('stellar', 'satori') else min(4, self.gpus_per_gx)
            nodes = math.ceil(self.gpus_per_gx/per_node)
            prefix = ["srun", "-N", str(nodes), "-n", devices, "-c", "1",
                      "--exact", "--gpus-per-task", "1"]
        elif system == 'summit':
            prefix = ["jsrun", "-n", devices, "-a", "1", "-c", "1", "-g", "1"]
        else:
            _, mask = self.gpu_groups[self.gx_index % self.workers]
            env["CUDA_VISIBLE_DEVICES"] = mask
            prefix = ["mpiexec", "-n", devices] if self.gpus_per_gx > 1 else []
        cmd = [*prefix, self.gx_app, f_in]
        self.info(f"> {cmd}")
        self.print_time()
        log = os.path.join(path, 'log.' + tag)
        self.processes.append(start_job(cmd, log, env=env))
        self.logs.append(log)
        self._progress[log] = (0, _time.monotonic())
        self.call_counter += 1
        self.gx_index += 1
        return f_tag

    def read_gx_fluxes(self, out_j, species_list):

        species_list = list(species_list)
        if len(out_j) != self.N_fluxtubes*len(self.zeta_centers):
            raise ValueError("GX requires one output for every radius and angle")
        pflux = np.zeros((len(species_list), self.N_fluxtubes))
        qflux = np.zeros((len(species_list), self.N_fluxtubes))
        heat = np.zeros((len(species_list), self.N_fluxtubes))

        out_id = 0
        for r_id in np.arange(self.N_fluxtubes):
            # fluxes will be summed over all field lines
            if self.theta_zeta_fluxes:
                pflux_sz = None
                qflux_sz = None
                heat_sz = None
                theta_z = None
                zeta_z = None
                grho_z = None
            for a_id in np.arange(len(self.zeta_centers)):
                out = out_j[out_id]
                self.info(f'  read_gx_fluxes: reading output for {out}')
                with self._open_gx_output(out) as f:
                    new_gx_diag = 'Diagnostics' in f.groups
                    if self.theta_zeta_fluxes:
                        if not new_gx_diag:
                            raise ValueError("GX theta_zeta_fluxes requires current diagnostics")
                        scale = f.groups['Geometry'].variables['theta_scale'][:]
                        theta = f.groups['Grids'].variables['theta'][:]*scale
                        nz = len(theta)

                        iota = 1/f.groups['Geometry'].variables['q'][:]
                        nfp = f.groups['Geometry'].variables['nfp'][:]
                        zeta_center = f.groups['Geometry'].variables['zeta_center'][:]
                        alpha = -iota*zeta_center
                        zeta = (theta - alpha)/iota

                        theta_p = map_to_periodic_range(theta, -np.pi, np.pi)
                        zeta_p = map_to_periodic_range(zeta, -np.pi/nfp, np.pi/nfp)
                        if theta_z is None:
                            theta_z = np.zeros(nz*len(self.zeta_centers))
                            zeta_z = np.zeros(nz*len(self.zeta_centers))
                            grho_z = np.zeros(nz*len(self.zeta_centers))
                        theta_z[a_id*nz:(a_id+1)*nz] = theta_p
                        zeta_z[a_id*nz:(a_id+1)*nz] = zeta_p

                        jacobian = f.groups['Geometry'].variables['jacobian'][:]
                        grho = f.groups['Geometry'].variables['grho'][:]
                        fluxDenom = np.sum(jacobian*grho)
                        grho_z[a_id*nz:(a_id+1)*nz] = jacobian*grho

                    # read qflux[t,s] = time trace of heat flux for each species
                    is_gx = 0
                    for i, s in enumerate(species_list):
                        if s.type == self.adiabatic_species:
                            pflux[i, r_id] = 0.0
                            qflux[i, r_id] = 0.0
                        else:
                            if self.theta_zeta_fluxes:
                                assert new_gx_diag, 'theta_zeta_fluxes require new gx diagnostics'

                                pflux_zt = f.groups['Diagnostics'].variables['ParticleFlux_zst'][:,is_gx,:]
                                pflux_z = np.mean(pflux_zt[int(len(pflux_zt[:,0])/2):,:], axis=0)*fluxDenom
                                if pflux_sz is None:
                                    pflux_sz = np.zeros((len(species_list), nz*len(self.zeta_centers)))
                                pflux_sz[i,a_id*nz:(a_id+1)*nz] = pflux_z
                            else:
                                if new_gx_diag:
                                    pflux_t = f.groups['Diagnostics'].variables['ParticleFlux_st'][:,is_gx]
                                else:
                                    pflux_t = f.groups['Fluxes'].variables['pflux'][:,is_gx]
                                pflux[i, r_id] += self.median_estimator(pflux_t)

                            if self.theta_zeta_fluxes:
                                qflux_zt = f.groups['Diagnostics'].variables['HeatFlux_zst'][:,is_gx,:]
                                qflux_z = np.mean(qflux_zt[int(len(qflux_zt[:,0])/2):,:], axis=0)*fluxDenom
                                if qflux_sz is None:
                                    qflux_sz = np.zeros((len(species_list), nz*len(self.zeta_centers)))
                                qflux_sz[i,a_id*nz:(a_id+1)*nz] = qflux_z
                            else:
                                if new_gx_diag:
                                    qflux_t = f.groups['Diagnostics'].variables['HeatFlux_st'][:,is_gx]
                                else:
                                    qflux_t = f.groups['Fluxes'].variables['qflux'][:,is_gx]
                                qflux[i, r_id] += self.median_estimator(qflux_t)

                            if self.theta_zeta_fluxes:
                                heat_zt = f.groups['Diagnostics'].variables['TurbulentHeating_zst'][:,is_gx,:]
                                heat_z = np.mean(heat_zt[int(len(heat_zt[:,0])/2):,:], axis=0)*fluxDenom
                                if heat_sz is None:
                                    heat_sz = np.zeros((len(species_list), nz*len(self.zeta_centers)))
                                heat_sz[i,a_id*nz:(a_id+1)*nz] = heat_z
                            else:
                                if new_gx_diag:
                                    heat_t = f.groups['Diagnostics'].variables['TurbulentHeating_st'][:,is_gx]
                                    heat[i, r_id] += self.median_estimator(heat_t)
                            is_gx += 1
                out_id += 1
            # end zeta_center loop
            if self.theta_zeta_fluxes:
                from scipy.interpolate import griddata
                grid_zeta, grid_theta = np.mgrid[-np.pi/nfp:np.pi/nfp:100j, -np.pi:np.pi:100j]
                interp_grho = griddata((zeta_z, theta_z), grho_z, (grid_zeta, grid_theta), method='cubic', fill_value=1e-10)
                integrated_grho = trapezoid(trapezoid(interp_grho, grid_theta[0,:], axis=1), grid_zeta[:,0])
                for i, s in enumerate(species_list):
                    if s.type == self.adiabatic_species:
                        continue
                    interp_pflux = griddata((zeta_z, theta_z), pflux_sz[i,:], (grid_zeta, grid_theta), method='cubic', fill_value=1e-10)
                    interp_qflux = griddata((zeta_z, theta_z), qflux_sz[i,:], (grid_zeta, grid_theta), method='cubic', fill_value=1e-10)
                    interp_heat = griddata((zeta_z, theta_z), heat_sz[i,:], (grid_zeta, grid_theta), method='cubic', fill_value=1e-10)

                    pflux[i, r_id] = trapezoid(trapezoid(interp_pflux, grid_theta[0,:], axis=1), grid_zeta[:,0])/integrated_grho
                    qflux[i, r_id] = trapezoid(trapezoid(interp_qflux, grid_theta[0,:], axis=1), grid_zeta[:,0])/integrated_grho
                    heat[i, r_id] = trapezoid(trapezoid(interp_heat, grid_theta[0,:], axis=1), grid_zeta[:,0])/integrated_grho
        # end radius loop

        if not self.theta_zeta_fluxes:
            # average fluxes over all field lines (already summed via += above)
            qflux = qflux/len(self.zeta_centers)
            pflux = pflux/len(self.zeta_centers)
            heat = heat/len(self.zeta_centers)

        qflux = qflux*self.qflux_scaling_factor

        if self.target_chie_over_chii:
            if self.adiabatic_species != "electron" or len(species_list) != 2:
                raise ValueError("GX chie_over_chii requires an adiabatic electron and one kinetic ion")
            e = next(i for i, item in enumerate(species_list) if item.type == "electron")
            ion, elec = species_list[1-e], species_list[e]
            ni, ne = (item.n().toFluxProfile().profile for item in (ion, elec))
            Ti, Te = (item.T().toFluxProfile().profile for item in (ion, elec))
            dTi, dTe = (-item.T().gradient_as_FluxProfile().profile for item in (ion, elec))
            Qi = qflux[1-e]*ni*Ti**2.5
            if np.any((dTi == 0) & (Qi != 0)):
                raise ValueError("GX chi_i is undefined for nonzero ion heat flux at zero ion temperature gradient")
            chi_i = np.divide(Qi, ni*dTi, out=np.zeros_like(Qi), where=dTi != 0)
            qflux[e] = self.target_chie_over_chii*chi_i*dTe/Te**2.5

        self.info(f'\n  {self.label}: qflux = {qflux}\n')
        self.info(f'\n  {self.label}: pflux = {pflux}\n')

        if self.abort_on_nans and not np.all(np.isfinite([pflux, qflux, heat])):
            raise ValueError(f"{self.label} contains nonfinite particle, heat, or exchange flux")

        return pflux, qflux, heat

    def median_estimator(self, flux):

        data = np.ma.asarray(flux)
        if data.ndim != 1 or data.size == 0 or np.any(np.ma.getmaskarray(data)):
            raise ValueError("GX diagnostics require a nonempty, unmasked time trace")
        if data.size == 1:
            return float(data[0])
        return float(np.median([np.median(data[::-1][:k]) for k in range(1, data.size)]))

    @staticmethod
    def _open_gx_output(stem):
        # GX legacy and current schemas are identified by group names,
        # rather than relying only on the output suffix.
        for suffix in (".out.nc", ".nc"):
            path = stem + suffix
            if valid_output(path):
                return Dataset(path, mode="r")
        raise RuntimeError(f"Missing GX NetCDF output for {stem}")

    @staticmethod
    def _scalar_output(dataset, groups_and_names):
        for group, key in groups_and_names:
            if group in dataset.groups and key in dataset[group].variables:
                v = np.ma.asarray(dataset[group][key][...], dtype=float)
                if v.size == 1 and not np.any(np.ma.getmaskarray(v)) and np.all(np.isfinite(v)):
                    return float(v.item())
        raise ValueError(f"Missing GX geometry variable(s) {groups_and_names}: {dataset.filepath()}")

    def get_gx_eqdsk_geometry(self, geo):
        # There is no field-line average in axisymmetry; exactly one
        # reference tube per surface is needed regardless of configured angles.
        original_angles = self.zeta_centers
        try:
            self.zeta_centers = [original_angles[0]]
            stems = self.run_gx_fluxtubes(self.rho, None, None, None, None, geo, 'i', dummy=True)
            self.wait()
        except BaseException:
            self.cancel_pending()
            raise
        finally:
            self.zeta_centers = original_angles
        self._require_gx_outputs(stems)
        if len(stems) != self.N_fluxtubes:
            raise ValueError("Expected one GX geometry output per radial surface")
        B_ref, a_ref, grho, area, Rnorm = [], [], [], [], []
        get = self._scalar_output
        for stem in stems:
            with self._open_gx_output(stem) as f:
                B_ref.append(get(f, (('Inputs','B_ref'), ('Geometry','B_ref'))))
                a_ref.append(get(f, (('Inputs','a_ref'), ('Geometry','a_ref'))))
                grho.append(get(f, (('Inputs','grhoavg'), ('Geometry','grhoavg'))))
                area.append(get(f, (('Inputs','surfarea'), ('Geometry','surfarea'))))
                Rnorm.append(get(f, (('Geometry','Rmaj'), ('Geometry','rmaj'))))
        if (min(B_ref) <= 0 or min(a_ref) <= 0 or min(Rnorm) <= 0
                or min(area) <= 0 or min(grho) <= 0
                or not np.allclose(B_ref,B_ref[0]) or not np.allclose(a_ref,a_ref[0])):
            raise ValueError("Invalid or inconsistent GX equilibrium geometry")
        geo.Btor_eq, geo.a_minor_eq, geo.R_major_eq = B_ref[0], a_ref[0], Rnorm[0]
        geo.B_ref = pf.FluxProfile(np.asarray(B_ref), self.grid)
        geo.a_ref = pf.FluxProfile(np.asarray(a_ref), self.grid)
        geo.grho = pf.FluxProfile(np.asarray(grho), self.grid)
        geo.area = pf.FluxProfile(np.asarray(area), self.grid)

    def final_info(self):

        self.info(f"Total number of GX calls = {self.call_counter}")

    def print_time(self):

        dt = datetime.now()
        self.info(f'  time: {dt}')
        # ts = datetime.timestamp(dt)
        # self.info(f'  time: {ts}')
