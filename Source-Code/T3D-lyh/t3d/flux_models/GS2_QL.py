from copy import deepcopy

import numpy as np
from scipy.integrate import trapezoid
import subprocess
from t3d import Profiles as pf
import os
from t3d.FluxModels import FluxModel
from netCDF4 import Dataset
import f90nml
from t3d.Logbook import logbook
from t3d.flux_models._external import (
    require_outputs, run_in_parallel, run_workspace, trial_tag, worker_count,
)


# Cefficients for the quasi linear metric
a = 2.5
b = 25


def _finite_nc(dataset, key):
    data = np.ma.asarray(dataset[key][...], dtype=float)
    if np.any(np.ma.getmaskarray(data)) or not np.all(np.isfinite(data.data)):
        raise ValueError(f"Invalid GS2 NetCDF variable {key}: {dataset.filepath()}")
    return np.asarray(data)


class GS2_QL_FluxModel(FluxModel):

    def __init__(self, pars, grid, time):

        # call base class constructor
        super().__init__(pars, grid, time)

        # environment variable containing path to gs2 executable
        GS2_PATH = pars.get('gs2_path', os.environ.get("GS2_PATH") or "")
        self.mpiexec = pars.get('mpiexec', 'mpirun')
        if self.mpiexec == 'srun':
            print('WARNING: srun should be used only in short runs, typically less than 10-20 time steps. '
                  'Please consider using multigs2 instead.')
        if 'multigs2' in self.mpiexec:
            self.gs2_app = os.path.join(GS2_PATH, "multigs2")
        else:
            self.gs2_app = os.path.join(GS2_PATH, "gs2")

        gs2_template = pars.get('gs2_template', 'tests/unit/gs2_template.in')
        out_dir = run_workspace(pars.get('gs2_outputs', 'gs2/'))
        self.logfile = pars.get('gs2_logfile', 'gs2.log')
        self.out_dir = out_dir
        self.overwrite = pars.get('overwrite', False)
        self.cpus_per_gs2 = pars.get('cpus_per_gs2', 1)
        self.nproc = pars.get('nproc', 1)
        self.workers = worker_count(self.nproc, self.cpus_per_gs2)
        self.electromagnetic = pars.get('electromagnetic', True)
        self.gammaE = pars.get('gammaE', 0)
        self.bakdif_on_template = pars.get('bakdif_on_template', False)
        self.label = pars.get('label', 'GS2')  # label for diagnostics

        # Set grid for ballooning simulations
        self.nky = pars.get('nky', 2)
        if type(self.nky) is not int or self.nky < 1:
            raise ValueError("GS2 nky must be a positive integer")
        kymin = pars.get('kymin', 0.1)
        kymax = pars.get('kymax', 1.0)
        self.ky = np.linspace(kymin, kymax, self.nky)
        theta0 = pars.get('theta0', [])
        if len(theta0) > 0:
            self.th0 = np.array(theta0)
            self.nth0 = len(theta0)
        else:
            self.nth0 = pars.get('ntheta0', 1)
            if type(self.nth0) is not int or self.nth0 < 1:
                raise ValueError("GS2 ntheta0 must be a positive integer")
            th0min = pars.get('th0min', 0)
            th0max = pars.get('th0max', 0)
            self.th0 = np.linspace(th0min, th0max, self.nth0)
        self.use_toroidal_n = pars.get('use_toroidal_n', False)
        if self.use_toroidal_n:
            try:
                nn = pars['nn']
            except KeyError:
                raise ValueError("Please provide a list of toroidal mode numbers")
            if not isinstance(nn, list):
                raise TypeError("nn must be a list")
            else:
                self.nn = np.array(nn)
                self.nky = len(nn)
                # Set ky to None because not used
                self.ky = None

        if self.nky < 2:
            raise ValueError("GS2 quasi-linear ky integration requires at least two modes")
        if not np.all(np.isfinite(self.th0)) or len(self.th0) < 1:
            raise ValueError("GS2 theta0 values must be finite and nonempty")
        if len(self.th0) > 1 and (self.th0[0] != 0 or not np.all(np.diff(self.th0) > 0)):
            raise ValueError("GS2 theta0 scan must start at zero and increase")
        if not np.all(np.isfinite(self.gammaE)):
            raise ValueError("GS2 gammaE must be finite")
        if np.ndim(self.gammaE) != 0 and np.shape(self.gammaE) != (self.N_fluxtubes,):
            raise ValueError("GS2 gammaE profile must match all transport faces")

        # GS2 uses vt = sqrt(2*T/m)
        self.vt_sqrt_2 = True
        # Factor to convert from quasi-linear metric normalization
        # to GS2 normalization. Alternatevely this factor can be
        # included in the b coefficient. Be careful that this factor
        # includes the conversion from Q_gB=(rho_s/a)**2*cs*T*n, used to
        # evaluate the fitting coefficients, to Q_gB_GS2 = (rho_th/a)**2*v_th*T*n,
        # used by T3D (vt_sqrt_2=True)
        self.facnorm = 2**(a-1.5)

        found_path = os.path.exists(out_dir)
        if not found_path:
            os.mkdir(out_dir)

        # Create logger
        self.log = logbook()
        self.log.set_handlers(term_stream=False,
                              file_stream=True,
                              file_handler=os.path.join(
                                  self.out_dir, self.logfile))
        self.info = self.log.info  # Short-cut command
        # Check file path
        self.info("  Looking for GS2 files")
        self.info("    Expecting GS2 template:", gs2_template)
        self.info("    Expecting GS2 executable:", self.gs2_app)
        self.info("    GS2-Trinity output path:", out_dir)

        found_gs2 = os.path.isfile(self.gs2_app) and os.access(self.gs2_app, os.X_OK)
        if not found_gs2:
            self.info("  Error: gs2 executable not found! Make sure the GS2_PATH environment variable is set.")
            raise RuntimeError('GS2 not found')

        self.info("")

        # load an input template
        self.read_input(gs2_template)
        try:
            self.nstep_gs2 = self.inputs['knobs']['nstep']
        except:
            self.nstep_gs2 = None

        self.old_t_id = 0
        self.old_p_id = 0
        self.processes = []
        self.logs = []

    def compute_fluxes(self, species, geo, norms):
        # in the following, A_sjk indicates A[s, j, k] where
        # s is species index
        # j is rho index
        # k is perturb index

        self.gs2_index = 0

        self.out_ncs_jk = []
        self.dkap_jk = []

        rho_j = self.rho
        finputs = []

        # base profiles case
        pert_id = 0
        # get gradient values on flux grid
        # these are 2d arrays, e.g. kn_sj = kap_n[s, j]
        kn0_sj, kT0_sj, kp0_sj, dkap = species.get_grads_on_flux_grid(pert_n=None, pert_T=None)
        self.dkap_jk.append(dkap)  # this is a dummy entry, unused
        fins, out_ncs_j = self.run_gs2_ballooning(rho_j, kn0_sj, kT0_sj, species, geo, pert_id, norms=norms)
        for f in fins:
            finputs.append(f)
        self.out_ncs_jk.append(out_ncs_j)

        # perturbed density cases
        # for each evolved density species, need to perturb density
        for stype in species.n_evolve_keys:
            pert_id = pert_id + 1
            kn_sj, kT_sj, kp_sj, dkap = species.get_grads_on_flux_grid(pert_n=stype, pert_T=None)
            self.dkap_jk.append(dkap)
            fins, out_ncs_for_sim = self.run_gs2_ballooning(rho_j, kn_sj, kT_sj, species, geo, pert_id, norms=norms)
            for f in fins:
                finputs.append(f)
            self.out_ncs_jk.append(out_ncs_for_sim)

        # perturbed temperature cases
        # for each evolved temperature species, need to perturb temperature
        for stype in species.T_evolve_keys:
            pert_id = pert_id + 1
            kn_sj, kT_sj, kp_sj, dkap = species.get_grads_on_flux_grid(pert_n=None, pert_T=stype)
            self.dkap_jk.append(dkap)
            fins, out_ncs_for_sim = self.run_gs2_ballooning(rho_j, kn_sj, kT_sj, species, geo, pert_id, norms=norms)
            for f in fins:
                finputs.append(f)
            self.out_ncs_jk.append(out_ncs_for_sim)

        # submit all the gs2 jobs at once
        self.submit_gs2_job(finputs)
        for outputs in self.out_ncs_jk:
            require_outputs(outputs, self.label)

        self.old_t_id = self.time.step_idx
        self.old_p_id = self.time.iter_idx

    def collect_results(self, species, geo, norms):

        # read GS2 output data, compute QL fluxes and pass
        # results to species base
        pert_id = 0
        pflux0_sj, qflux0_sj, heat0_sj = self.read_gs2_fluxes(self.out_ncs_jk[pert_id], species, geo)
        species.add_flux(pflux0_sj, qflux0_sj, heat0_sj, label=self.label)

        # save qflux value
        self.qflux_prev = qflux0_sj
        for i, s in enumerate(species.get_species_list()):
            if s.is_adiabatic:
                self.qflux_prev[i, :] = 1  # for adiabatic species, use a dummy value that won't turn off restarts

        pert_id = pert_id + 1

        # perturbed density cases
        for stype in species.n_evolve_keys:
            dkn_j = self.dkap_jk[pert_id]
            pflux_sj, qflux_sj, heat_sj = self.read_gs2_fluxes(self.out_ncs_jk[pert_id], species, geo)
            dpflux_dkn_sj = (pflux_sj - pflux0_sj) / dkn_j
            dqflux_dkn_sj = (qflux_sj - qflux0_sj) / dkn_j
            dheat_dkn_sj = (heat_sj - heat0_sj) / dkn_j
            species.add_dflux_dkn(stype, dpflux_dkn_sj, dqflux_dkn_sj, dheat_dkn_sj)
            pert_id = pert_id + 1

        # perturbed temperature cases
        for stype in species.T_evolve_keys:
            dkT_j = self.dkap_jk[pert_id]
            pflux_sj, qflux_sj, heat_sj = self.read_gs2_fluxes(self.out_ncs_jk[pert_id], species, geo)
            dpflux_dkT_sj = (pflux_sj - pflux0_sj) / dkT_j
            dqflux_dkT_sj = (qflux_sj - qflux0_sj) / dkT_j
            dheat_dkT_sj = (heat_sj - heat0_sj) / dkT_j
            species.add_dflux_dkT(stype, dpflux_dkT_sj, dqflux_dkT_sj, dheat_dkT_sj)
            pert_id = pert_id + 1

    def read_input(self, fin):

        with open(fin) as f:
            data = f90nml.read(f)

        self.inputs = data
        self.filename = fin

    # run a GS2 ballooning calculation at each radius, given gradient values kns and kts
    def run_gs2_ballooning(self, rho, kns, kts, species, geo, pert_id, norms=None, dummy=False):

        trial = trial_tag(self.time)
        gs2_inputs = deepcopy(self.inputs)

        if dummy:
            kymode, theta0 = [1.], [0.]
            gs2_inputs['theta_grid_parameters']['nperiod'] = 1
        else:
            theta0 = self.th0
            # Set ky modes
            if self.use_toroidal_n:
                kymode = self.nn
                Tr = species.ref_species.T().toFluxProfile().profile
                Br = geo.Btor
                rhostar = norms.rho_star * np.sqrt(Tr) / Br
            else:
                kymode = self.ky

            # get profile values on flux grid
            # these are 2d arrays, e.g. ns = ns[species_idx, rho_idx]
            ns, Ts, nu_ss = species.get_profiles_on_flux_grid(normalize=True, a_minor=geo.a_minor)
            beta_ref = species.ref_species.beta(geo.Btor).toFluxProfile()
            if geo.use_beta_prime_alpha:
                beta_prime_alpha = geo.beta_prime_alpha
            else:
                beta_prime_alpha = np.zeros(self.N_fluxtubes)
            if geo.beta_prime_const:
                beta_prime = species.beta_prime
                bcf = 0
            else:
                beta_prime = np.zeros(self.N_fluxtubes)
                bcf = 1
            N_species = species.N_species

        # loop over radii and (ky, theta0) modes
        finput = []
        out_ncs = []
        path = self.out_dir
        for r_id in np.arange(self.N_fluxtubes):
            for ky_id, ky in enumerate(kymode):
                for th0_id, th0 in enumerate(theta0):

                    tag = f"{trial}-r{r_id}-ky{ky_id}-th0{th0_id}-{pert_id}"
                    fin = os.path.join(path, tag + '.in')
                    out_ncs.append(os.path.join(path, tag + '.out.nc'))
                    if os.path.exists(fin):
                        raise RuntimeError(f"Duplicate GS2 trial input: {fin}")
                    finput.append(fin)

                    if self.use_toroidal_n and not dummy:
                        gs2_inputs['kt_grids_single_parameters']['n0'] = ky
                        gs2_inputs['knobs']['rhostar'] = rhostar[r_id]
                    else:
                        gs2_inputs['kt_grids_single_parameters']['aky'] = ky
                        gs2_inputs['kt_grids_single_parameters'].pop('n0', None)
                    gs2_inputs['kt_grids_single_parameters']['theta0'] = th0

                    if geo.geo_option == 'geqdsk':
                        gs2_inputs['theta_grid_parameters']['rhoc'] = rho[r_id]
                        gs2_inputs['theta_grid_eik_knobs']['irho'] = 1 if self.grid.flux_label == 'torflux' else 2
                        gs2_inputs['theta_grid_eik_knobs']['eqfile'] = os.path.abspath(geo.geo_file)
                        gs2_inputs['theta_grid_eik_knobs']['efit_eq'] = True
                        gs2_inputs['theta_grid_eik_knobs']['bishop'] = 1
                        gs2_inputs['theta_grid_eik_knobs']['local_eq'] = False
                    elif geo.geo_option == 'miller':
                        gs2_inputs['theta_grid_eik_knobs']['bishop'] = 4
                        gs2_inputs['theta_grid_eik_knobs']['local_eq'] = True
                        gs2_inputs['theta_grid_parameters']['rhoc'] = rho[r_id]
                        gs2_inputs['theta_grid_parameters']['r_geo'] = geo.AspectRatio
                        gs2_inputs['theta_grid_parameters']['rmaj'] = geo.rmaj[r_id]
                        gs2_inputs['theta_grid_parameters']['qinp'] = geo.qsf[r_id]
                        gs2_inputs['theta_grid_eik_knobs']['s_hat_input'] = geo.shat[r_id]
                        gs2_inputs['theta_grid_parameters']['shift'] = geo.rmajor_prime[r_id]
                        gs2_inputs['theta_grid_parameters']['akappa'] = geo.kappa[r_id]
                        gs2_inputs['theta_grid_parameters']['akappri'] = geo.kappa_prime[r_id]
                        gs2_inputs['theta_grid_parameters']['tri'] = geo.delta[r_id]
                        gs2_inputs['theta_grid_parameters']['tripri'] = geo.delta_prime[r_id]
                        gs2_inputs['theta_grid_eik_knobs']['beta_prime_input'] = (
                            -beta_ref[r_id]*bcf*np.sum(
                                ns[:, r_id]*Ts[:, r_id]*(kns[:, r_id]+kts[:, r_id])) +
                            beta_prime_alpha[r_id] + (1-bcf)*beta_prime[r_id])
                    else:
                        raise ValueError(
                            "Invalid value of geo_option. Please use 'geqdsk' or 'miller'")

                    if dummy:
                        gs2_inputs['knobs']['nstep'] = 1
                    else:
                        if self.nstep_gs2:
                            gs2_inputs['knobs']['nstep'] = self.nstep_gs2
                        else:
                            gs2_inputs['knobs']['nstep'] = 100000

                    if dummy:
                        # A neutral H/e pair with zero gradients, independent of evolving profiles.
                        gs2_inputs['species_knobs']['nspec'] = 2
                        gs2_inputs['knobs']['beta'] = 0.0
                        gs2_inputs['knobs']['fapar'] = 0.0
                        gs2_inputs['knobs']['fbpar'] = 0.0
                        for isp, (mass, charge, kind) in enumerate(((1., 1., 'ion'), (1./1836, -1., 'electron')), 1):
                            gs2_inputs[f'species_parameters_{isp}'] = dict(
                                mass=mass, z=charge, dens=1., temp=1., fprim=0., tprim=0.,
                                vnewk=0., type=kind)
                    else:
                        gs2_inputs['species_knobs']['nspec'] = species.N_species

                        # set reference beta
                        if self.electromagnetic:
                            gs2_inputs['knobs']['beta'] = beta_ref[r_id]
                        else:
                            gs2_inputs['knobs']['beta'] = 0
                            gs2_inputs['knobs']['fapar'] = 0.0
                            gs2_inputs['knobs']['fbpar'] = 0.0

                        # set species parameters
                        for isp in range(species.N_species):
                            gs2_inputs[f'species_parameters_{isp+1}'] = {}
                            gs2_inputs[f'species_parameters_{isp+1}']['mass'] = species.get_masses(normalize=True)[isp]
                            gs2_inputs[f'species_parameters_{isp+1}']['z'] = species.get_charges(normalize=True)[isp]
                            gs2_inputs[f'species_parameters_{isp+1}']['dens'] = ns[isp, r_id]
                            gs2_inputs[f'species_parameters_{isp+1}']['temp'] = Ts[isp, r_id]
                            gs2_inputs[f'species_parameters_{isp+1}']['fprim'] = kns[isp, r_id]
                            gs2_inputs[f'species_parameters_{isp+1}']['tprim'] = kts[isp, r_id]
                            gs2_inputs[f'species_parameters_{isp+1}']['vnewk'] = nu_ss[isp, r_id]
                            gs2_inputs[f'species_parameters_{isp+1}']['type'] = species.get_types_ion_electron()[isp]
                            if not self.bakdif_on_template:
                                gs2_inputs[f'dist_fn_species_knobs_{isp+1}'] = {}
                                gs2_inputs[f'dist_fn_species_knobs_{isp+1}']['fexpr'] = 0.48
                                gs2_inputs[f'dist_fn_species_knobs_{isp+1}']['bakdif'] = 0.05

                    # restart file not needed
                    gs2_inputs['gs2_diagnostics_knobs']['save_for_restart'] = False

                    # set some diagnostics
                    gs2_inputs['gs2_diagnostics_knobs']['write_fluxes'] = True
                    gs2_inputs['gs2_diagnostics_knobs']['write_fields'] = False
                    gs2_inputs['gs2_diagnostics_knobs']['write_moments'] = False
                    gs2_inputs['gs2_diagnostics_knobs']['write_final_fields'] = True
                    gs2_inputs['gs2_diagnostics_knobs']['write_ascii'] = False

                    # write input file
                    gs2_inputs.write(fin, force=self.overwrite)

        # gs2 jobs will be submitted allogether in compute_fluxes
        # here we return the input file list

        return finput, out_ncs

    def run_gs2(self, cmd, log_path):
        with open(log_path, 'w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)

    def submit_gs2_job(self, finput):
        if not finput:
            return
        nprocsim = self.cpus_per_gs2
        if self.mpiexec in ("mpirun", "srun"):
            prefix = [self.mpiexec, "-n", str(nprocsim)]
            if self.mpiexec == "srun":
                prefix.append("--exact")

            def run_one(fin):
                self.run_gs2([*prefix, self.gs2_app, fin], fin[:-3] + ".out")

            run_in_parallel(run_one, finput, self.workers)
        elif self.mpiexec in ("multigs2", "mpirun-multigs2"):
            flist = os.path.join(self.out_dir, trial_tag(self.time) + "-input.list.txt")
            with open(flist, "w") as f:
                for fin in finput:
                    print(fin, file=f)
            if self.mpiexec == "multigs2":
                prefix = ["srun", "-n", str(self.nproc), "--cpus-per-task=1", "--hint=nomultithread"]
                options = ["--debug"]
            else:
                prefix = ["mpirun", "-n", str(self.nproc)]
                options = []
            self.run_gs2([*prefix, self.gs2_app, "--nbatch", str(self.workers), *options, flist],
                         os.path.join(self.out_dir, trial_tag(self.time) + "-multigs2.out"))
        else:
            raise ValueError("GS2 mpiexec must be mpirun, srun, multigs2, or mpirun-multigs2")

    @staticmethod
    def _ballooning_average(values, angles, shear, gamma_E, gamma_zero):
        # Giacomin et al. JPP 90 (2024), equations (3.4)-(3.5):
        # no-flow-shear limit is the theta0=0 value, not an integral
        # over the first nonzero mesh interval. Apply the advection range
        # to the piecewise-linear interpolant, including its truncated tail.
        if gamma_zero <= 0:
            return np.zeros(values.shape[:-1])
        if gamma_E == 0:
            return values[..., 0]
        thmax = min(np.pi, abs(gamma_E/(shear*gamma_zero))) if shear != 0 else np.pi
        if thmax <= 0:
            return values[..., 0]
        if thmax > angles[-1] + 1e-12:
            raise ValueError("GS2 theta0 scan does not span the required flow-shear integration interval")
        thmax = min(thmax, angles[-1])
        k = np.searchsorted(angles, thmax, side="left")
        endpoint = values[..., k-1] + (values[..., k]-values[..., k-1])*(thmax-angles[k-1])/(angles[k]-angles[k-1])
        vals = np.concatenate((values[..., :k], endpoint[..., None]), axis=-1)
        return trapezoid(vals, np.r_[angles[:k], thmax], axis=-1)/thmax

    def read_gs2_fluxes(self, fnc_j, species, geo):
        if len(fnc_j) != self.N_fluxtubes*self.nky*self.nth0:
            raise ValueError("GS2 requires a result for each radius, ky, and theta0")
        pflux = np.zeros((species.N_species, self.N_fluxtubes))
        qflux = np.zeros_like(pflux)
        knobs = self.inputs["knobs"]
        fapar = self.electromagnetic and knobs.get("fapar", 1.)
        fbpar = self.electromagnetic and knobs.get("fbpar", 0.)
        for r_id in range(self.N_fluxtubes):
            gamma_E = np.asarray(self.gammaE, dtype=float)
            if gamma_E.ndim:
                gamma_E = gamma_E[r_id]
            ky_values = np.empty(self.nky)
            integrands = []
            for ky_id in range(self.nky):
                modes = [self.compute_ql_metric(fnc_j[(r_id*self.nky+ky_id)*self.nth0+j], fapar, fbpar)
                         for j in range(self.nth0)]
                gamma, ky, lam, heat, particle = zip(*modes)
                if not np.allclose(ky, ky[0], rtol=1e-10, atol=0):
                    raise ValueError("GS2 returned inconsistent ky for one theta0 scan")
                ky_values[ky_id] = ky[0]
                values = np.vstack((lam, np.asarray(heat).T, np.asarray(particle).T))
                if self.nth0 == 1:
                    if gamma_E != 0:
                        raise ValueError("GS2 flow-shear suppression requires a theta0 scan including zero")
                    integrands.append(values[:, 0])
                else:
                    integrands.append(self._ballooning_average(values, self.th0, geo.shat[r_id], gamma_E, gamma[0]))
            order = np.argsort(ky_values)
            ky_values = ky_values[order]
            if self.nky < 2 or not np.all(np.diff(ky_values) > 0):
                raise ValueError("GS2 quasi-linear ky integration requires at least two distinct ky values")
            lam, *weighted = trapezoid(np.asarray(integrands)[order], ky_values, axis=0)
            factor = self.facnorm*b*lam**(a-1)
            qflux[:, r_id] = factor*np.asarray(weighted[:species.N_species])
            pflux[:, r_id] = factor*np.asarray(weighted[species.N_species:])
        return pflux, qflux, np.zeros_like(qflux)

    def compute_ql_metric(self, fin, fapar, fbpar):
        with Dataset(fin, mode="r") as f:
            def get(key):
                return _finite_nc(f, key)
            phi2_fit, t, ky = get('phi2'), get('t'), float(get('ky').item())
            qes = np.asarray(f['es_heat_flux'][-1]).reshape(-1)
            zeros = np.zeros_like(qes, dtype=float)
            if (t.ndim != 1 or phi2_fit.shape != t.shape or len(t) < 2
                    or not np.all(np.diff(t) > 0)):
                raise ValueError(f"GS2 requires a valid linear amplitude time trace in {fin}")
            if phi2_fit[-1] < 1e-16:
                return 0., ky, 0., zeros.copy(), zeros.copy()
            n_tail = min(len(t), max(2, int(np.ceil(.3*len(t)))))
            tail = phi2_fit[-n_tail:]
            if np.any(tail <= 0):
                raise ValueError(f"GS2 linear fit needs positive phi2 in {fin}")
            gamma = float(np.polyfit(t[-n_tail:], np.log(tail), 1)[0]/2)
            if gamma <= 0:
                return gamma, ky, 0., zeros.copy(), zeros.copy()

            def last(key):
                row = np.ma.asarray(f[key][-1])
                if np.any(np.ma.getmaskarray(row)) or not np.all(np.isfinite(row.data)):
                    raise ValueError(f"Missing final GS2 flux diagnostic {key} in {fin}")
                return np.asarray(row).reshape(-1)

            theta, jacob = get('theta'), np.abs(get('jacob'))
            kperp = np.squeeze(get('kperp2'))
            if (theta.ndim != 1 or jacob.shape != theta.shape or kperp.shape != theta.shape
                    or not np.all(np.diff(theta) > 0) or np.any(jacob <= 0) or np.any(kperp < 0)):
                raise ValueError(f"Invalid GS2 theta/J/kperp2 grid in {fin}")

            def field_amplitude(key):
                field = np.squeeze(get(key))
                if field.shape != (theta.size, 2):
                    raise ValueError(f"Invalid GS2 final field shape for {key} in {fin}")
                return np.hypot(field[:, 0], field[:, 1])

            phi = field_amplitude('phi')
            phi_max = np.max(phi)
            if phi_max <= 0:
                raise ValueError(f"GS2 unstable mode has zero potential eigenfunction in {fin}")
            qtot, gtot = last('es_heat_flux'), last('es_part_flux')
            lam = 0.
            for enabled, key in ((True, 'phi'), (fapar, 'apar'), (fbpar, 'bpar')):
                if not enabled:
                    continue
                if key == 'phi':
                    amplitude = phi
                else:
                    amplitude = field_amplitude(key)
                    qtot = qtot + last(key+'_heat_flux')
                    gtot = gtot + last(key+'_part_flux')
                peak = np.max(amplitude)
                if peak == 0:
                    continue
                power = (amplitude/peak)**2*jacob
                denom = trapezoid(power*kperp, theta)
                if denom <= 0:
                    raise ValueError(f"GS2 unstable mode has undefined kperp average for {key} in {fin}")
                lam += gamma*(peak/phi_max)*trapezoid(power, theta)/denom
            qsum = np.sum(qtot)
            if abs(qsum) <= np.finfo(float).eps*np.sum(np.abs(qtot)):
                raise ValueError(f"GS2 unstable mode has zero total heat, leaving quasi-linear weights undefined in {fin}")
            # Keep signed weights as in Eq. (3.7)-(3.8), so sum_s Q_s = Q_ql.
            return gamma, ky, lam, qtot/qsum*lam, gtot/qsum*lam

    def get_gs2_eqdsk_geometry(self, geo):
        # Read one nperiod=1, init-only result per transport face.
        fins, fnc_j = self.run_gs2_ballooning(self.rho, None, None, None, geo, 'i', dummy=True)
        self.submit_gs2_job(fins)
        require_outputs(fnc_j, self.label)
        if len(fnc_j) != self.N_fluxtubes:
            raise RuntimeError("GS2 geometry needs exactly one output per transport face")

        B_ref, a_ref, grho, area, shear, Rnorm = [], [], [], [], [], []
        for path in fnc_j:
            with Dataset(path, mode='r') as f:
                get = lambda key: _finite_nc(f, key)
                theta, g = get('theta'), get('grho')
                if 'jacob' in f.variables:
                    weight = np.abs(get('jacob'))
                else:
                    denom = get('drhodpsi')*get('gradpar')*get('bmag')
                    if np.any(denom == 0):
                        raise ValueError(f"Invalid GS2 geometry Jacobian in {path}")
                    weight = 1/np.abs(denom)
                if (theta.ndim != 1 or theta.size < 2 or not np.all(np.diff(theta) > 0)
                        or g.shape != theta.shape or np.any(g <= 0)
                        or weight.shape != theta.shape or np.any(weight <= 0)):
                    raise ValueError(f"Invalid GS2 geometry grid in {path}")
                surface = 2*np.pi*trapezoid(g*weight, theta)
                area.append(surface)
                grho.append(surface/(2*np.pi*trapezoid(weight, theta)))
                B_ref.append(float(get('bref').item()))
                a_ref.append(float(get('aref').item()))
                shear.append(float(get('shat').item()))
                # GS2 modern main output does not emit rmaj; use RMAXIS / aref
                # from the GEQDSK file then, not the local Rplot.
                if 'rmaj' in f.variables:
                    Rnorm.append(float(get('rmaj').item()))
        if (min(B_ref) <= 0 or min(a_ref) <= 0 or not np.allclose(B_ref, B_ref[0])
                or not np.allclose(a_ref, a_ref[0])):
            raise ValueError("GS2 geometry requires consistent positive aref and bref (see normalisations_knobs)")
        if Rnorm and (len(Rnorm) != self.N_fluxtubes or min(Rnorm) <= 0 or not np.allclose(Rnorm,Rnorm[0])):
            raise ValueError("GS2 geometry has inconsistent rmaj reference values")
        if not Rnorm:
            try:
                with open(geo.geo_file) as f:
                    next(f)  # GEQDSK header
                    next(f)  # rdim,zdim,rcentr,rleft,zmid
                    rm = float(next(f)[:16].replace('D','E').replace('d','e'))
            except (OSError, ValueError, StopIteration) as exc:
                raise ValueError("Cannot read GS2 rmaj or EFIT GEQDSK RMAXIS") from exc
            if not np.isfinite(rm) or rm <= 0:
                raise ValueError("Invalid GEQDSK RMAXIS")
            Rnorm = [rm/a_ref[0]]

        geo.Btor_eq = B_ref[0]
        geo.a_minor_eq = a_ref[0]
        geo.R_major_eq = Rnorm[0]
        geo.B_ref = pf.FluxProfile(np.asarray(B_ref), self.grid)
        geo.a_ref = pf.FluxProfile(np.asarray(a_ref), self.grid)
        geo.grho = pf.FluxProfile(np.asarray(grho), self.grid)
        geo.area = pf.FluxProfile(np.asarray(area), self.grid)
        geo.shat = pf.FluxProfile(np.asarray(shear), self.grid)
