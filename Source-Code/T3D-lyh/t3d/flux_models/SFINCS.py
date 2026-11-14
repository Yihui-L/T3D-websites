import numpy as np
import subprocess
from datetime import datetime
import os
import shutil
from t3d.FluxModels import FluxModel
from t3d.Logbook import logbook
from t3d.flux_models._external import (
    available_cpus, cancel_jobs, command_prefix, require_outputs, run_workspace,
    start_job, trial_tag, wait_checked, worker_count,
)

class SFINCS_FluxModel(FluxModel):

    def __init__(self, pars, grid, time):

        # call base class constructor
        super().__init__(pars, grid, time)

        # environment variable containing path to sfincs.x executable
        SFINCS_PATH = self.SFINCS_PATH = os.path.abspath(
            os.environ.get("SFINCS_PATH") or pars.get('sfincs_path', ""))

        out_dir = self.out_dir = run_workspace(pars.get('sfincs_outputs', 'sfincs/'))
        self.logfile = pars.get('sfincs_logfile', 'sfincs.log')
        sfincs_template = pars.get('sfincs_template', 'tests/sfincs/sfincs_template.in')
        self.overwrite = pars.get('overwrite', False)
        self.label = pars.get('label', 'SFINCS')  # label for diagnostics

        # SFINCS uses sqrt(2T/m) as v normalization but we modify its behavior
        # to allow simulations with GX
        self.vt_sqrt_2 = False

        # use a non-default run command (e.g. "srun ...")
        # full command line will be cmd = f"{self.run_command} {gx_app} {f_in}"
        self.run_command = pars.get('run_command', None)
        self.cpus_per_sfincs = pars.get('cpus_per_sfincs', 1)
        self.mpis_per_sfincs = pars.get('mpis_per_sfincs', 1)
        if (type(self.cpus_per_sfincs) is not int or self.cpus_per_sfincs <= 0
                or type(self.mpis_per_sfincs) is not int or self.mpis_per_sfincs <= 0):
            raise ValueError("SFINCS ranks and threads per rank must be positive integers")
        # total number of allocated cpus
        self.cpus_total = pars.get('cpus_total', available_cpus())
        self.workers = worker_count(self.cpus_total, self.cpus_per_sfincs * self.mpis_per_sfincs)
        self.sfincs_index = 0

        found_path = os.path.exists(out_dir)
        if not found_path:
            os.mkdir(out_dir)

        # Create logger
        self.log = logbook()
        self.log.set_handlers(term_stream=False,
                              file_stream=True,
                              file_handler=os.path.join(self.out_dir, self.logfile))
        self.info = self.log.info  # Short-cut command
        self.errr = self.log.errr  # Short-cut command

        # Check file path
        self.info("  Looking for SFINCS files")
        self.info(f"    Expecting SFINCS template: {sfincs_template}")
        self.info(f"    Expecting SFINCS executable: {SFINCS_PATH+'/sfincs'}")
        self.info(f"    SFINCS-Trinity output path: {out_dir}")

        found_sfincs = os.path.exists(SFINCS_PATH+"/sfincs")
        if not found_sfincs:
            self.errr("sfincs executable not found! Make sure the SFINCS_PATH environment variable is set.")
            raise RuntimeError('SFINCS not found')

        self.info("")

        self.read_sfincs_input(sfincs_template)

        system = os.environ.get('GK_SYSTEM', 'local')

        # set up node files for PBS systems like Polaris
        if system == "polaris":  # or other PBS systems eventually
            nodefile = os.environ.get("PBS_NODEFILE")
            if nodefile is None:
                raise RuntimeError("PBS_NODEFILE is required for the Polaris SFINCS launcher")
            # One host file per node; each case occupies a disjoint CPU slot.
            self.cores_per_case = self.mpis_per_sfincs * self.cpus_per_sfincs
            self.cases_per_node = worker_count(32, self.cores_per_case)
            with open(nodefile) as hosts:
                hostnames = list(dict.fromkeys(line.strip() for line in hosts if line.strip()))
            if not hostnames:
                raise RuntimeError("PBS_NODEFILE contains no allocated nodes")
            self.workers = min(self.workers, len(hostnames)*self.cases_per_node)
            for i, host in enumerate(hostnames):
                with open(os.path.join(self.out_dir, f'local_hostfile.{i+1:02}'), 'w') as file:
                    print(host, file=file)

        self.first = True
        self.processes = []

    def __del__(self):
        ''' Use destructor to close the log file '''
        self.log.finalize()

    def logfile_info(self) -> str:
        ''' Return string of log file location '''
        return self.log.file_handler.name

    # run SFINCS calculations and pass fluxes and flux jacobian to SpeciesDict species
    def compute_fluxes(self, species, geo, norms):
        # in the following, A_sjk indicates A[s, j, k] where
        # s is species index
        # j is rho index
        # k is perturb index

        self.outs_jk = []
        self.dkap_jk = []
        self.sfincs_index = 0
        rho_j = self.rho

        if geo.geo_option == 'desc' and not str(geo.geo_file).lower().endswith('.nc'):
            raise ValueError("SFINCS DESC coupling requires VMEC-format wout; set desc_use_vmec_format")
        if geo.geo_option in ('vmec', 'desc'):
            if self.grid.flux_label != 'torflux':
                raise ValueError("SFINCS with VMEC wout requires the toroidal-flux radius")
            self.namelist["geometryparameters"]["geometryscheme"] = "5"
            self.namelist["geometryparameters"]["equilibriumfile"] = '"../../../{}"'.format(geo.geo_file.strip())
            self.namelist["geometryparameters"]["inputradialcoordinate"] = "3"
            self.namelist["geometryparameters"]["inputradialcoordinateforgradients"] = "3"
            self.namelist["geometryparameters"]["vmecradialoption"] = "0"

        self.namelist["speciesparameters"]["zs"] = ' '.join(map(str, species.get_charges(normalize=True)))
        self.namelist["speciesparameters"]["mhats"] = ' '.join(map(str, species.get_masses(normalize=True)))
        self.namelist['general']['ambipolarsolve'] = '.true.'
        self.namelist['general']['ambipolarsolveoption'] = '2'

        # set normalization parameters so that SFINCS reference parameters match T3D ones... but not R_bar!
        # n_bar = 10^20 m^-3
        # T_bar = 1 keV
        # B_bar = 1 T
        # m_bar = m_ref

        # for R_bar, geometryOption = 5 requires R_bar = 1 m, which is different from the T3D reference length, a_meters
        # this means we need to multiply T3D reference rho_star by a_meters
        self.namelist['physicsparameters']['Delta'] = norms.rho_star*np.sqrt(2/norms.vtfac)*norms.a_meters
        self.namelist['physicsparameters']['alpha'] = 1.0
        self.namelist['physicsparameters']['nu_n'] = 8.33e-3

        # base profiles case
        pert_id = 0
        # get gradient values on flux grid
        # these are 2d arrays, e.g. kn_sj = kap_n[s, j]
        kn0_sj, kT0_sj, kp0_sj, dkap = species.get_grads_on_flux_grid(pert_n=None, pert_T=None)
        self.dkap_jk.append(dkap)  # this is a dummy entry, unused
        outs_j = self.run_sfincs(rho_j, kn0_sj, kT0_sj, species, geo, pert_id)
        self.outs_jk.append(outs_j)

        # perturbed density cases
        # for each evolved density species, need to perturb density
        for stype in species.n_evolve_keys:
            pert_id = pert_id + 1
            kn_sj, kT_sj, kp_sj, dkap = species.get_grads_on_flux_grid(pert_n=stype, pert_T=None)
            self.dkap_jk.append(dkap)
            outs_j = self.run_sfincs(rho_j, kn_sj, kT_sj, species, geo, pert_id)
            self.outs_jk.append(outs_j)

        # perturbed temperature cases
        # for each evolved temperature species, need to perturb temperature
        for stype in species.T_evolve_keys:
            pert_id = pert_id + 1
            kn_sj, kT_sj, kp_sj, dkap = species.get_grads_on_flux_grid(pert_n=None, pert_T=stype)
            self.dkap_jk.append(dkap)
            outs_j = self.run_sfincs(rho_j, kn_sj, kT_sj, species, geo, pert_id)
            self.outs_jk.append(outs_j)

    def collect_results(self, species, geo, norms):
        # collect parallel runs
        self.wait()

        for outputs in self.outs_jk:
            require_outputs(outputs, self.label)

        # read SFINCS output data and pass results to species
        # base
        pert_id = 0
        pflux0_sj, qflux0_sj = self.read_sfincs_fluxes(self.outs_jk[pert_id], species, geo, norms)
        species.add_flux(pflux0_sj, qflux0_sj, None, label=self.label)

        pert_id = pert_id + 1

        # perturbed density cases
        for stype in species.n_evolve_keys:
            dkn_j = self.dkap_jk[pert_id]
            pflux_sj, qflux_sj = self.read_sfincs_fluxes(self.outs_jk[pert_id], species, geo, norms)
            dpflux_dkn_sj = (pflux_sj - pflux0_sj) / dkn_j
            dqflux_dkn_sj = (qflux_sj - qflux0_sj) / dkn_j
            species.add_dflux_dkn(stype, dpflux_dkn_sj, dqflux_dkn_sj, None)
            pert_id = pert_id + 1

        # perturbed temperature cases
        for stype in species.T_evolve_keys:
            dkT_j = self.dkap_jk[pert_id]
            pflux_sj, qflux_sj = self.read_sfincs_fluxes(self.outs_jk[pert_id], species, geo, norms)
            dpflux_dkT_sj = (pflux_sj - pflux0_sj) / dkT_j
            dqflux_dkT_sj = (qflux_sj - qflux0_sj) / dkT_j
            species.add_dflux_dkT(stype, dpflux_dkT_sj, dqflux_dkT_sj, None)
            pert_id = pert_id + 1

    def wait(self):
        try:
            while self.processes:
                self._finish_first()
        except BaseException:
            self.cancel_pending()
            raise

    def _finish_first(self):
        wait_checked(self.processes[0], self.label)
        self.processes.pop(0)

    def cancel_pending(self):
        jobs, self.processes = self.processes, []
        cancel_jobs(jobs)

    def run_sfincs(self, rho, kns, kts, species, geo, pert_id):
        trial = trial_tag(self.time)

        # get profile values on flux grid
        # these are 2d arrays, e.g. ns = ns[species_idx, rho_idx]
        ns, Ts, nu_ss = species.get_profiles_on_flux_grid(normalize=False, a_minor=geo.a_minor)

        run_dir = os.path.join(self.out_dir, f"{trial}-j{pert_id}")
        os.mkdir(run_dir)

        flux_files = []

        flux_dirs = [run_dir + f"/rho{n:02}" for n in np.arange(self.N_fluxtubes)]
        for r_id, flux_dir in enumerate(flux_dirs):
            os.mkdir(flux_dir)
            input_file = flux_dir + "/input.namelist"

            sfincs_namelist = self.namelist
            sfincs_namelist["speciesparameters"]["nhats"] = ' '.join(map(str, ns[:, r_id]))
            sfincs_namelist["speciesparameters"]["thats"] = ' '.join(map(str, Ts[:, r_id]))
            sfincs_namelist["speciesparameters"]["dnhatdrns"] = ' '.join(map(str, -ns[:, r_id]*kns[:, r_id]))
            sfincs_namelist["speciesparameters"]["dthatdrns"] = ' '.join(map(str, -Ts[:, r_id]*kts[:, r_id]))
            sfincs_namelist["geometryparameters"]["rn_wish"] = rho[r_id]

            geo_file_path = os.path.abspath(geo.geo_file.strip())
            geo_filename = os.path.basename(geo_file_path)
            shutil.copyfile(geo_file_path, os.path.join(flux_dir, geo_filename))
            self.namelist["geometryparameters"]["equilibriumfile"] = f'"{geo_filename}"'

            self.write_sfincs_input(input_file,sfincs_namelist)
            flux_files.append(self.submit_sfincs_job(flux_dir))
        return flux_files

    def submit_sfincs_job(self, flux_dir):
        # Wait before reusing a resource slot. With FIFO submission each slot
        # has completed before the next case assigned to it can be started.
        if len(self.processes) >= self.workers:
            self._finish_first()
        flux_file = os.path.join(flux_dir, "sfincsOutput.h5")
        sfincs_app = os.path.join(self.SFINCS_PATH, "sfincs")
        system = os.environ.get('GK_SYSTEM', 'local')
        ranks = str(self.mpis_per_sfincs)
        threads = str(self.cpus_per_sfincs)
        env = dict(os.environ, OMP_NUM_THREADS=threads)

        if self.run_command is not None:
            cmd = [*command_prefix(self.run_command), sfincs_app]
        elif system == 'perlmutter':
            cmd = ["srun", "-u", "-t", "0:30:00", "-N", "1", "--exact",
                   "--ntasks", ranks, "--cpus-per-task", threads, sfincs_app]
        elif system == 'stellar':
            cmd = ["srun", "-t", "4:00:00", "--exact", "--ntasks", ranks,
                   "-c", threads, sfincs_app]
        elif system == 'polaris':
            node, slot = divmod(self.sfincs_index % self.workers, self.cases_per_node)
            first = 32 + slot*self.cores_per_case
            cpu_lists = [','.join(str(cpu) for cpu in range(first + i*self.cpus_per_sfincs,
                                                          first + (i+1)*self.cpus_per_sfincs))
                         for i in range(self.mpis_per_sfincs)]
            hostfile = os.path.join(self.out_dir, f"local_hostfile.{node+1:02}")
            cmd = ["mpiexec", "-n", ranks, "--ppn", ranks, "--cpu-bind",
                   "list:" + ':'.join(cpu_lists), "--env", f"OMP_NUM_THREADS={threads}",
                   "--hostfile", hostfile, sfincs_app, "-ksp_view", "-mat_mumps_cntl_1", "1e-3"]
        elif system == 'summit':
            cmd = ["jsrun", "-n", "1", "-a", ranks, "-c",
                   str(self.mpis_per_sfincs*self.cpus_per_sfincs), "-g", "0",
                   f"-EOMP_NUM_THREADS={threads}", sfincs_app, "-ksp_view", "-mat_mumps_cntl_1", "1e-3"]
        else:
            cmd = ["mpirun", "-np", ranks, sfincs_app, "-ksp_max_it", "500"]

        self.info(f"> {cmd}")
        self.print_time()
        log_path = os.path.join(flux_dir, "output.out")
        self.processes.append(start_job(cmd, log_path, cwd=flux_dir, env=env))
        self.sfincs_index += 1
        return flux_file

    def read_sfincs_fluxes(self, outs_j, species, geo, norms):
        import h5py

        if len(outs_j) != self.N_fluxtubes:
            raise ValueError("SFINCS requires one output per transport face")
        pflux = np.zeros((species.N_species, self.N_fluxtubes))
        qflux = np.zeros_like(pflux)

        for r_id, out in enumerate(outs_j):
            with h5py.File(out, 'r') as f:
                def scalar(key):
                    value = np.asarray(f[key][...])
                    if value.size != 1:
                        raise ValueError(f"Malformed SFINCS status {key}: {out}")
                    return value.item()

                true = scalar("integerToRepresentTrue") if "integerToRepresentTrue" in f else 1
                if "finished" not in f or scalar("finished") != true:
                    raise RuntimeError(f"SFINCS output is not marked finished: {out}")
                if "didNonlinearCalculationConverge" in f and scalar("didNonlinearCalculationConverge") != true:
                    raise RuntimeError(f"SFINCS nonlinear solve did not converge: {out}")
                if "includePhi1" in f:
                    include_phi1 = scalar("includePhi1") == true
                else:
                    # Older files may omit the configuration metadata.
                    include_phi1 = self.namelist["physicsparameters"].get("includephi1", ".false.").lower() == ".true."
                suffix = "vd" if include_phi1 else "vm"

                def final_flux(kind):
                    data = f[f"{kind}Flux_{suffix}_rN"]
                    if data.ndim != 2 or data.shape[0] != species.N_species or data.shape[1] == 0:
                        raise ValueError(f"Malformed SFINCS {kind} flux array: {out}")
                    if "NIterations" in f and scalar("NIterations") != data.shape[1]:
                        raise ValueError(f"Incomplete SFINCS {kind} flux history: {out}")
                    y = data[:, -1]
                    if not np.all(np.isfinite(y)):
                        raise ValueError(f"Nonfinite SFINCS {kind} flux: {out}")
                    return y

                pflux[:, r_id] = final_flux("particle")
                qflux[:, r_id] = final_flux("heat")

        # SFINCS fluxes are in normalized sfincs units. need to normalize to gyroBohm
        # note SFINCS uses vt = sqrt(2T/m), but to make this model compatible with other
        # models with vt_sqrt_2 = False, multiply by sqrt(2) factors
        for i, s in enumerate(species.get_species_list()):
            qflux[i, :] = qflux[i, :]*np.sqrt(8)*geo.Btor**2/(species.ref_species.n()*species.ref_species.T()**2.5).toFluxProfile()/geo.grho*norms.rho_star**-2*norms.a_meters
            pflux[i, :] = pflux[i, :]*np.sqrt(2)*geo.Btor**2/(species.ref_species.n()*species.ref_species.T()**1.5).toFluxProfile()/geo.grho*norms.rho_star**-2*norms.a_meters

        self.info(f'\n  {self.label}: qflux = {qflux}\n')
        self.info(f'\n  {self.label}: pflux = {pflux}\n')
        return pflux, qflux

    def write_sfincs_input(self, fout, namelist):
        with open(fout, 'x') as f:

            for item in namelist.items():

                if (type(item[1]) is not dict):
                    print('  %s = %s ' % item, file=f)
                    continue

                header, nest = item
                print('\n&%s' % header, file=f)
                if len(nest.keys()) <= 0:
                    print('/\n', file=f)
                    continue
                longest_key = max(nest.keys(), key=len)
                N_space = len(longest_key)
                for pair in nest.items():
                    s = '  {:%i}  =  {}' % N_space
                    print(s.format(*pair), file=f)
                print('/\n', file=f)

        self.info(f'  wrote input: {fout}')

    def read_sfincs_input(self,fin):
        with open(fin) as f:
            data = f.readlines()

        obj = {}
        header = ''
        for line in data:

            # strip comments
            if line.find('!') > -1:
                end = line.find('!')
                line = line[:end]

            # parse headers
            if line.find('&') == 0:
                header = line.split('&')[1].strip().lower()
                obj[header] = {}
                continue

            if len(line.strip()) > 0 and line.strip()[0] == '/':
                header = ''
            # skip blanks
            if line.find('=') < 0:
                continue

            # store data
            key, value = line.split('=')
            key = key.strip().lower()
            value = value.strip()

            if header == '':
                obj[key] = value
            else:
                obj[header][key] = value

        # check for all possible header types in GX input file,
        # and initialize empty if not found
        try:
            obj['general']
        except:
            obj['general'] = {}
        try:
            obj['geometryparameters']
        except:
            obj['geometryparameters'] = {}
        try:
            obj['speciesparameters']
        except:
            obj['speciesparameters'] = {}
        try:
            obj['physicsparameters']
        except:
            obj['physicsparameters'] = {}
        try:
            obj['resolutionparameters']
        except:
            obj['resolutionparameters'] = {}
        try:
            obj['othernumericalparameters']
        except:
            obj['othernumericalparameters'] = {}
        try:
            obj['preconditioneroptions']
        except:
            obj['preconditioneroptions'] = {}
        try:
            obj['export_f']
        except:
            obj['export_f'] = {}
        try:
            obj['adjointoptions']
        except:
            obj['adjointoptions'] = {}

        self.namelist = obj
        self.filename = fin

    def print_time(self):

        dt = datetime.now()
        self.info(f'  time: {dt}')
        # ts = datetime.timestamp(dt)
        # print('  time', ts)
