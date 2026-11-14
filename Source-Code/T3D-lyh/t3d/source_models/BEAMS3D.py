import os
import shlex
import subprocess
from copy import deepcopy
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from scipy.constants import proton_mass

from t3d.SourceModels import SourceModel


class BEAMS3D_SourceModel(SourceModel):
    """Neutral-beam thermalization and slowing-down sources from STELLOPT."""

    def __init__(self, pars, grid, time):
        super().__init__(pars, grid, time)
        self.label = pars.get("label", "BEAMS3D")
        self.particle_source = pars.get("particle_source", False)
        self.power_source = pars.get("power_source", True)
        if not self.particle_source and not self.power_source:
            raise ValueError("BEAMS3D must enable particle_source and/or power_source")
        if grid.flux_label != "torflux":
            raise ValueError("BEAMS3D requires rho = sqrt(normalized toroidal flux)")
        beam_path = Path(pars.get("beams3d_path", os.environ.get("BEAMS3D_PATH") or "."))
        self.beams3d_app = (beam_path / "xbeams3d").resolve()
        if not self.beams3d_app.is_file() or not os.access(self.beams3d_app, os.X_OK):
            raise RuntimeError(f"BEAMS3D executable not found or not executable: {self.beams3d_app}")
        template = pars.get("beams3d_template")
        if template is None:
            raise ValueError("Set beams3d_template to a BEAMS3D neutral-beam namelist")
        import f90nml
        self.input_data = f90nml.read(template)
        if "beams3d_input" not in self.input_data:
            raise ValueError("Missing &BEAMS3D_INPUT namelist")
        self.run_command = shlex.split(pars.get("run_command") or "")
        self.run_options = shlex.split(pars.get("run_options", ""))
        if any(x.lower() in {"-depo", "-hitonly", "-hit_only", "-plasma"} for x in self.run_options):
            raise ValueError("BEAMS3D thermal sources require full collisional tracking, not deposition-only mode")
        out = Path(pars.get("beams3d_outputs", "beams3d")).resolve()
        out.mkdir(parents=True, exist_ok=True)
        self.out_dir = Path(mkdtemp(prefix="t3d-run-", dir=out))
        self.logfile = Path(pars.get("beams3d_logfile", "beams3d.log")).name
        self.implicit = pars.get("implicit", False)
        self.relative_step = float(pars.get("relative_step", 1e-3))
        if not 0 < self.relative_step < 1:
            raise ValueError("BEAMS3D relative_step must be between 0 and 1")
        self.source_jacobian = None

    def logfile_info(self):
        return str(self.out_dir / "*" / self.logfile)

    def compute_sources(self, species, geo, norms):
        self.source_jacobian = None
        n, T, _ = species.get_profiles_on_grid()
        base = self.run_beams3d(n, T, species, geo)
        self.base = self.read_beams3d_deposition(base, species, norms)
        if not self.implicit or self.time.alpha == 0:
            return

        y = species.get_vec_from_profs()
        jac = np.zeros((y.size, y.size))
        scratch = species.profile_scratch()
        baseline = self.source_vector(self.base, species)
        for k, value in enumerate(y):
            if k % self.N_radial == self.N_radial - 1:
                continue  # prescribed edge row
            for sign in (1, -1):
                delta = sign*self.relative_step*value
                trial = y.copy()
                trial[k] += delta
                scratch.get_profs_from_vec(trial)
                if scratch.profiles_are_physical():
                    break
            else:
                raise ValueError("BEAMS3D: no physical perturbation of independent n,p")
            n, T, _ = scratch.get_profiles_on_grid()
            output = self.run_beams3d(n, T, scratch, geo)
            perturbed = self.read_beams3d_deposition(output, species, norms)
            jac[:, k] = (self.source_vector(perturbed, species) - baseline)/delta
        self.source_jacobian = jac

    def source_vector(self, sources, species):
        # Exactly the independent-variable order and units of TransportSolver.Svec.
        Sn, Sp = sources
        index = {s.type: i for i, s in enumerate(species.get_species_list())}
        blocks = [Sn[index[k]].copy() for k in species.n_evolve_keys]
        blocks.extend((2/3)*Sp[index[k]] for k in species.T_evolve_keys)
        for block in blocks:
            block[-1] = 0
        return np.concatenate(blocks)

    def collect_results(self, species, geo, norms):
        species.add_source(*self.base, label=self.label)

    def run_beams3d(self, ns, Ts, species, geo):
        import f90nml

        geo_file = Path(geo.geo_file).resolve()
        if geo_file.suffix != ".nc" or not geo_file.is_file():
            raise ValueError("BEAMS3D requires an existing VMEC-format wout .nc file")
        prefix = f"t{self.time.step_idx:03d}-p{self.time.iter_idx}-r{getattr(self.time, 'retry_generation', 0)}-"
        case = Path(mkdtemp(prefix=prefix, dir=self.out_dir))
        tag = "t3d"
        (case / f"wout_{tag}.nc").symlink_to(geo_file)
        inputs = deepcopy(self.input_data)
        p = inputs["beams3d_input"]
        e = species.sidx_electron
        # BEAMS3D profile splines use VMEC s, eV, SI densities, ion masses in kg.
        s = self.rho**2
        if s[0] > 0 or s[-1] < 1:
            # Keep the prescribed first/last profile value outside the T3D domain.
            s_full = np.unique(np.r_[0., s, 1.])
            ns = np.stack([np.interp(s_full, s, profile) for profile in ns])
            Ts = np.stack([np.interp(s_full, s, profile) for profile in Ts])
            s = s_full
        names = list(species.get_species_list())
        bulk = next(i for i, item in enumerate(names) if item is species.bulk_ion)
        if self.particle_source:
            # Validate matching Fortran beam-line indices, not offsets in each
            # parsed list (indexed namelist slices can start at different lines).
            def indexed(key):
                first = p.start_index.get(key, [1])[0] or 1
                return {i: value for i, value in enumerate(np.atleast_1d(p.get(key, [])), first)
                        if value is not None}

            powers, masses = indexed("power_beams"), indexed("mass_beams")
            for index, power in powers.items():
                if power > 0 and not np.isclose(masses.get(index, np.nan),
                                                  names[bulk].mass*proton_mass, atol=0, rtol=.02):
                    raise ValueError(f"BEAMS3D beam {index} particle_source requires the bulk ion isotope")
        charges = species.get_charges(normalize=False, drop="electron")
        if not np.all(np.isfinite(charges) & (charges > 0) & (charges == np.rint(charges))):
            raise ValueError("BEAMS3D background ion charges must be positive integers")
        p["te_aux_s"], p["te_aux_f"] = s.tolist(), (Ts[e] * 1e3).tolist()
        p["ti_aux_s"], p["ti_aux_f"] = s.tolist(), (Ts[bulk] * 1e3).tolist()
        p["ne_aux_s"], p["ne_aux_f"] = s.tolist(), (ns[e] * 1e20).tolist()
        p["ni_aux_s"] = s.tolist()
        p["ni_aux_z"] = charges.astype(int).tolist()
        # Species.mass uses m_p units (as in T3D collision frequencies).
        p["ni_aux_m"] = (species.get_masses(normalize=False, drop="electron") * proton_mass).tolist()
        # Fortran NI_AUX_F(ion, radial); f90nml expects its first index innermost.
        p["ni_aux_f"] = (np.delete(ns, e, axis=0).T * 1e20).tolist()
        # Replacing the arrays does not remove start_index from a parsed input.
        # Force replacement values to begin at ion 1/radius 1.
        for key in ("te_aux_s", "te_aux_f", "ti_aux_s", "ti_aux_f", "ne_aux_s",
                    "ne_aux_f", "ni_aux_s", "ni_aux_z", "ni_aux_m", "ni_aux_f"):
            p.start_index[key] = [1, 1] if key == "ni_aux_f" else [1]
        f90nml.write(inputs, case / f"input.{tag}")

        argv = [*self.run_command, str(self.beams3d_app), "-vmec", tag, "-collisions", *self.run_options]
        with (case / self.logfile).open("w") as log:
            subprocess.run(argv, cwd=case, stdout=log, stderr=subprocess.STDOUT, check=True)
        output = case / f"beams3d_{tag}.h5"
        if not output.is_file():
            raise RuntimeError(f"BEAMS3D did not produce {output}; see {case / self.logfile}")
        return output

    def read_beams3d_deposition(self, out_file, species, norms):
        import h5py

        Sn = np.zeros((species.N_species, self.N_radial))
        Sp = np.zeros_like(Sn)
        names = list(species.get_species_list())
        bulk = next(i for i, item in enumerate(names) if item is species.bulk_ion)
        e = species.sidx_electron
        with h5py.File(out_file, "r") as data:
            rho = np.asarray(data["dist_rhoaxis"]).reshape(-1)
            if rho.size < 2 or not np.all(np.isfinite(rho)) or not np.all(np.diff(rho) > 0):
                raise ValueError("Invalid BEAMS3D dist_rhoaxis")

            def profile(key):
                # STELLOPT writes Fortran (nbeams,ns_prof1), visible in h5py
                # with radial bins as the first (slow) axis.
                a = np.asarray(data[key])
                if a.ndim != 2 or a.shape[0] != len(rho) or not np.all(np.isfinite(a)):
                    raise ValueError(f"Invalid BEAMS3D {key} shape or values: {a.shape}")
                return np.interp(self.rho, rho, a.sum(axis=1))

            if self.particle_source:
                ndot = profile("ndot_prof") * 1e-20 / norms.Sn_ref_SI20
                Sn[bulk] = ndot
                Sn[e] = species.bulk_ion.Z * ndot
            if self.power_source:
                Sp[e] = profile("epower_prof") * 1e-6 / norms.P_ref_MWm3
                Sp[bulk] = profile("ipower_prof") * 1e-6 / norms.P_ref_MWm3
        return Sn, Sp
