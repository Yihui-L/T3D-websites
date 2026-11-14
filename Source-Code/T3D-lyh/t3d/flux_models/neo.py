import numpy as np
import subprocess
import os
import sys
from pathlib import Path
from t3d.FluxModels import FluxModel
from t3d.flux_models._external import (
    available_cpus, require_outputs, run_in_parallel, run_workspace, trial_tag, worker_count,
)


class neo_FluxModel(FluxModel):

    def __init__(self, pars, grid, time):

        super().__init__(pars, grid, time)
        NEO_PATH = os.environ.get("GACODE_ROOT") or ""
        self.neo_app = str(Path(NEO_PATH, "neo").resolve())

        neo_template = pars.get("neo_template")
        out_dir = run_workspace(pars.get("neo_outputs", "neo"))
        self.qmax = pars.get("qmax", 0)
        self.out_dir = out_dir
        self.mpiexec = pars.get("mpiexec", "mpirun")
        self.cpu = pars.get("cpus_per_task", 1)
        self.workers = worker_count(pars.get("nproc", available_cpus()), self.cpu)
        self.mem = pars.get("mem_per_cpu", 5)
        self.label = pars.get('label', 'NEO')  # label for diagnostics
        # This needs to be true to be able
        # to be coupled with GS2_QL.
        # This factor is then corrected
        # when the fluxes are converted
        # to trinity units.
        self.vt_sqrt_2 = True

        # Check file path
        print("  Looking for NEO files")
        print("    Expecting NEO template:", neo_template)
        print("    Expecting NEO executable:", self.neo_app)
        print("    NEO-Trinity output path:", out_dir)

        found_path = os.path.exists(out_dir)
        if not found_path:
            print(f"      creating new output dir {out_dir}")
            os.mkdir(out_dir)

        found_neo = os.path.exists(self.neo_app)
        if not found_neo:
            print("  Error: neo executable not found!"
                  "Make sure the GACODE_ROOT environment variable is set.")
            raise RuntimeError('NEO not found')

        print("")

        self.read_input(neo_template)

    def compute_fluxes(self, species, geo, norms):
        self.neo_index = 0
        self.out_jk = []
        self.dkap_jk = []
        rho_j = self.rho
        self.compute_bunit(geo)

        pert_id = 0
        kn0_sj, kT0_sj, kp0_sj, dkap = species.get_grads_on_flux_grid(
            pert_n=None, pert_T=None)
        self.dkap_jk.append(dkap)
        pths = self.run_neo(rho_j, kn0_sj, kT0_sj, species, geo, pert_id, norms=norms)
        finputs = []
        fouts = []
        for f in pths:
            finputs.append(f)
            fouts.append(os.path.join(f, 'out.neo.transport'))
        self.out_jk.append(fouts)

        # perturbed density cases
        # for each evolved density species, need to perturb density
        for stype in species.n_evolve_keys:
            pert_id = pert_id + 1
            kn_sj, kT_sj, kp_sj, dkap = species.get_grads_on_flux_grid(
                pert_n=stype, pert_T=None)
            self.dkap_jk.append(dkap)
            pths = self.run_neo(rho_j, kn_sj, kT_sj, species, geo, pert_id, norms=norms)
            fouts = []
            for f in pths:
                finputs.append(f)
                fouts.append(os.path.join(f, 'out.neo.transport'))
            self.out_jk.append(fouts)

        # perturbed temperature cases
        # for each evolved temperature species, need to perturb temperature
        for stype in species.T_evolve_keys:
            pert_id = pert_id + 1
            kn_sj, kT_sj, kp_sj, dkap = species.get_grads_on_flux_grid(
                pert_n=None, pert_T=stype)
            self.dkap_jk.append(dkap)
            pths = self.run_neo(rho_j, kn_sj, kT_sj, species, geo, pert_id, norms=norms)
            fouts = []
            for f in pths:
                finputs.append(f)
                fouts.append(os.path.join(f, 'out.neo.transport'))
            self.out_jk.append(fouts)

        # submit all the neo jobs at once
        self.submit_neo(finputs)

        self.old_t_id = self.time.step_idx
        self.old_p_id = self.time.iter_idx

    def collect_results(self, species, geo, norms):

        pert_id = 0
        pflux0_sj, qflux0_sj, heat0_sj = self.read_neo_fluxes(
            self.out_jk[pert_id], species, geo, norms)
        species.add_flux(pflux0_sj, qflux0_sj, heat0_sj, label=self.label)

        pert_id = pert_id + 1

        # perturbed density cases
        for stype in species.n_evolve_keys:
            dkn_j = self.dkap_jk[pert_id]
            pflux_sj, qflux_sj, heat_sj = self.read_neo_fluxes(
                self.out_jk[pert_id], species, geo, norms)
            dpflux_dkn_sj = (pflux_sj - pflux0_sj) / dkn_j
            dqflux_dkn_sj = (qflux_sj - qflux0_sj) / dkn_j
            dheat_dkn_sj = (heat_sj - heat0_sj) / dkn_j
            species.add_dflux_dkn(stype, dpflux_dkn_sj, dqflux_dkn_sj, dheat_dkn_sj)
            pert_id = pert_id + 1

        # perturbed temperature cases
        for stype in species.T_evolve_keys:
            dkT_j = self.dkap_jk[pert_id]
            pflux_sj, qflux_sj, heat_sj = self.read_neo_fluxes(
                self.out_jk[pert_id], species, geo, norms)
            dpflux_dkT_sj = (pflux_sj - pflux0_sj) / dkT_j
            dqflux_dkT_sj = (qflux_sj - qflux0_sj) / dkT_j
            dheat_dkT_sj = (heat_sj - heat0_sj) / dkT_j
            species.add_dflux_dkT(stype, dpflux_dkT_sj, dqflux_dkT_sj, dheat_dkT_sj)
            pert_id = pert_id + 1

    def compute_bunit(self, geo):
        # Compute Bunit/B0 to convert rho_unit to rho_s
        theta = np.linspace(0, 2 * np.pi, 256)
        bunit = np.zeros(self.N_fluxtubes)
        for r_id in np.arange(self.N_fluxtubes):
            kappa = geo.kappa[r_id]
            s_kappa = geo.s_kappa[r_id]
            delta = geo.tri[r_id]
            s_delta = geo.s_delta[r_id]
            shift = geo.rmajor_prime[r_id]
            rmaj = geo.rmaj[r_id]
            rmin = self.rho[r_id]

            R = rmaj + rmin * np.cos(theta + np.arcsin(delta) * np.sin(theta))
            Z = kappa * rmin * np.sin(theta)
            dR = (np.roll(R, 1) - np.roll(R, -1)) / 2.0
            dZ = (np.roll(Z, 1) - np.roll(Z, -1)) / 2.0
            dL = np.sqrt(dR**2 + dZ**2)

            x = np.arcsin(delta)
            term1 = 1 / kappa
            term2 = np.sqrt(
                np.sin(theta + x * np.sin(theta))**2 * (1 + x * np.cos(theta))**2 +
                (kappa * np.cos(theta))**2
            )
            term3 = np.cos(x * np.sin(theta)) + shift * np.cos(theta)
            term4 = (
                (s_kappa - s_delta * np.cos(theta) +
                 (1 + s_kappa) * x * np.cos(theta)) *
                np.sin(theta) *
                np.sin(theta + x * np.sin(theta))
            )
            grad_r = term1 * term2 / (term3 + term4)

            integral = np.sum(dL / (R * grad_r))

            bunit[r_id] = integral * rmaj / (2 * np.pi * rmin)

        self.bunit_over_b0 = bunit

    def read_input(self, fin):

        data = {}
        with open(fin) as f:
            for line in f.readlines():
                data[line.split('=')[0]] = line.split('=')[-1].strip()

        self.inputs = data

    def write_input(self, fin, data):

        with open(fin, 'w') as f:
            for key in data.keys():
                line = f"{key}={data[key]}\n"
                f.write(line)

    def run_neo(self, rho, kns, kts, species, geo, pert_id, norms):

        trial = trial_tag(self.time)
        neo_inputs = self.inputs

        Tr = species.electron.T().toFluxProfile().profile
        Br = geo.Btor * self.bunit_over_b0
        rhostar = norms.rho_star * np.sqrt(Tr) / Br

        ns, Ts, nu_ss = species.get_profiles_on_flux_grid(normalize=True, a_minor=geo.a_minor)
        N_species = species.N_species

        # loop over radii
        pths = []
        path = self.out_dir
        for r_id in np.arange(self.N_fluxtubes):
            pth = os.path.join(path, f"{trial}-r{r_id}-j{pert_id}")
            os.mkdir(pth)
            pths.append(pth)
            fin = os.path.join(pth, 'input.neo')

            neo_inputs['Q'] = geo.qsf[r_id]
            neo_inputs['SHEAR'] = geo.shat[r_id]
            neo_inputs['SHIFT'] = geo.rmajor_prime[r_id]
            neo_inputs['KAPPA'] = geo.kappa[r_id]
            neo_inputs['S_KAPPA'] = geo.s_kappa[r_id]
            neo_inputs['DELTA'] = geo.tri[r_id]
            neo_inputs['S_DELTA'] = geo.s_delta[r_id]
            neo_inputs['RMIN_OVER_A'] = rho[r_id]
            neo_inputs['RMAJ_OVER_A'] = geo.rmaj[r_id]
            neo_inputs['RHO_STAR'] = rhostar[r_id]

            # set species parameters
            neo_inputs['N_SPECIES'] = species.N_species
            for isp in range(species.N_species):
                neo_inputs[f'MASS_{isp+1}'] = species.get_masses(normalize=True)[isp]
                neo_inputs[f'Z_{isp+1}'] = species.get_charges(normalize=True)[isp]
                neo_inputs[f'DENS_{isp+1}'] = ns[isp, r_id]
                neo_inputs[f'TEMP_{isp+1}'] = Ts[isp, r_id]
                neo_inputs[f'DLNNDR_{isp+1}'] = kns[isp, r_id]
                neo_inputs[f'DLNTDR_{isp+1}'] = kts[isp, r_id]
            # Only NU_1 is required in NEO
            # Account for vt_sqrt_2 = True
            neo_inputs['NU_1'] = nu_ss[0, r_id] * np.sqrt(2)

            # write input file
            self.write_input(fin, neo_inputs)

        # neo jobs will be submitted allogether in compute_fluxes
        # here we return the input file list

        return pths

    def run_neo_single(self, path):
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.cpu)
        with open(os.path.join(path, "out.neo.run"), "w") as log:
            kwargs = dict(cwd=path, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, os.path.join(self.neo_app, "bin/neo_parse.py")], **kwargs)
            if self.mpiexec == "srun":
                prefix = ["srun", "-N", "1", "--exclusive", "-n", "1", "-c", str(self.cpu),
                          f"--mem-per-cpu={self.mem}G"]
            elif self.mpiexec == "mpirun":
                prefix = ["mpirun", "-n", "1"]
            elif self.mpiexec in ("local", "none", ""):
                prefix = []
            else:
                raise ValueError("NEO mpiexec must be mpirun, srun, or local")
            subprocess.run([*prefix, os.path.join(self.neo_app, "src/neo")], **kwargs)
        require_outputs([os.path.join(path, "out.neo.transport")], self.label)

    def submit_neo(self, finput):
        run_in_parallel(self.run_neo_single, finput, self.workers)

    def read_neo_fluxes(self, f_j, species, geo, norms):
        pflux = np.zeros((species.N_species, self.N_fluxtubes))
        qflux = np.zeros((species.N_species, self.N_fluxtubes))
        heat = np.zeros((species.N_species, self.N_fluxtubes))
        for r_id, fout in enumerate(f_j):
            with open(fout) as f:
                data = f.readlines()[0].split()
                for isp in range(species.N_species):
                    pflux[isp, r_id] = float(data[5+isp*8])
                    qflux[isp, r_id] = float(data[6+isp*8])

        # Convert to trinity units
        pflux = pflux / (norms.rho_star**2 * np.sqrt(2))
        qflux = qflux / (norms.rho_star**2 * np.sqrt(2))
        # Clamp the value to qmax near the magnetic axis
        # if qmax>0 (default qmax=0).
        # Convergence near the magnetic axis may be tricky
        # and small differences are enlarged by the choice
        # of the normalisation.
        if self.qmax > 0:
            for isp in range(species.N_species):
                if abs(qflux[isp, 0]) > self.qmax:
                    qflux[isp, 0] = np.sign(qflux[isp, 0]) * self.qmax

        return pflux, qflux, heat
