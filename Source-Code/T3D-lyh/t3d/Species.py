from collections import OrderedDict
import copy
import re

import numpy as np
from scipy.special import erf, erfc

from t3d.Profiles import FluxProfile, GridProfile
from t3d.Logbook import info, warn
import t3d.constants

m_proton_cgs = t3d.constants.M_PROTON_CGS  # mass of proton in grams
e_cgs = t3d.constants.ELEMENTARY_CHARGE  # in Coulombs


class SpeciesDict():

    def __init__(self, inputs, grid):
        species_list_params = inputs.get('species', {})
        self.N_species = len(species_list_params)
        self.N_radial = grid.N_radial
        self.grid = grid

        # create dictionary of species objects, keyed by species type (e.g. 'deuterium', 'electron', etc)
        self.species_dict = OrderedDict()
        self.has_adiabatic_species = False
        reference_species_count = 0
        adiabatic_species_count = 0
        bulk_ion_count = 0
        impurity_count = 0
        impurity_type = None
        first_ion_type = None
        last_ion_type = None
        explicit_qn = []
        last_evolving_ion_type = None

        self.N_dens_profiles = 0
        self.N_temp_profiles = 0
        self.N_profiles = 0
        self.n_evolve_keys = []
        self.T_evolve_keys = []
        self.n_equalto_keys = []
        self.T_equalto_keys = []
        force_qn = False

        for idx, sp in enumerate(species_list_params):
            # initialize a species object
            s = Species(sp, grid)
            # store species object in dictionary keyed by species type (e.g. 'deuterium', 'electron', etc)
            if s.type in self.species_dict:
                raise ValueError(f"Duplicate species type {s.type!r}")
            self.species_dict[s.type] = s
            if s.qneut:
                explicit_qn.append(s.type)

            # check for adiabatic species
            if s.is_adiabatic:
                adiabatic_species_count = adiabatic_species_count + 1
                adiabatic_type = s.type
                s.evolve_density = False
                s.evolve_temperature = False

            # check for reference species
            if s.use_as_reference:
                reference_species_count = reference_species_count + 1
                reference_type = s.type

            # check for bulk ion species
            if s.bulk_ion:
                bulk_ion_count = bulk_ion_count + 1
                bulk_ion_type = s.type
                assert s.type != 'electron', "Error: electrons cannot be used as bulk ion species"

            # check for impurity species
            if s.impurity:
                impurity_count = impurity_count + 1
                impurity_type = s.type

            # save the type of the first listed ion species
            if first_ion_type is None and s.type != 'electron':
                first_ion_type = s.type

            if s.density_equal_to is not None:
                s.evolve_density = False
                self.n_equalto_keys.append(s.type)
            if s.evolve_density:
                self.N_dens_profiles = self.N_dens_profiles + 1
                # make a list of species types that will evolve density
                self.n_evolve_keys.append(s.type)
            if s.temperature_equal_to is not None:
                s.evolve_temperature = False
                self.T_equalto_keys.append(s.type)
            if s.evolve_temperature:
                self.N_temp_profiles = self.N_temp_profiles + 1
                # make a list of species types that will evolve temperature
                self.T_evolve_keys.append(s.type)

            # save the type of the last ion species
            if s.type != 'electron':
                last_ion_type = s.type
                if s.evolve_density:
                    last_evolving_ion_type = s.type
            else:
                self.sidx_electron = idx

        if "electron" not in self.species_dict or last_ion_type is None:
            raise ValueError("Transport requires an electron species and at least one ion species")
        if len(explicit_qn) > 1:
            raise ValueError("Only one species may set qneut_species=true")
        if explicit_qn:
            qneut_type = explicit_qn[0]
        elif impurity_type is not None:
            qneut_type = impurity_type
        elif last_evolving_ion_type is not None:
            qneut_type = last_evolving_ion_type
        else:
            # An equal_to density is already constrained. Let a free ion (or
            # the electron if all ions are linked) supply charge neutrality.
            qneut_type = next((s.type for s in reversed(list(self.species_dict.values()))
                               if s.type != "electron" and s.density_equal_to is None), "electron")
        self.qneut_species = self.species_dict[qneut_type]
        if self.qneut_species.density_equal_to is not None:
            raise ValueError(f"The quasineutral species {qneut_type!r} cannot also set density.equal_to")

        # the qneut_type density profile can be set by quasineutrality,
        # so don't include it in count of evolved profiles
        if self.N_dens_profiles > 0 and self.qneut_species.evolve_density:  # last_evolving_ion_type is not None:
            self.N_dens_profiles = self.N_dens_profiles - 1
            self.species_dict[qneut_type].evolve_density = False
            self.n_evolve_keys.remove(qneut_type)

        # number of evolved profiles
        self.N_profiles = self.N_dens_profiles + self.N_temp_profiles

        # sanity checks
        assert adiabatic_species_count <= 1, "Error: cannot have more than one adiabatic species specified in [species] section. Specify adiabatic species in [model] instead."
        assert reference_species_count <= 1, "Error: cannot have more than one species set as reference species"
        assert bulk_ion_count <= 1, "Error: cannot have more than one species set as bulk ions"
        for t in self.n_equalto_keys:
            assert self.species_dict[t].density_equal_to in self.species_dict.keys(), f"Error: cannot set '{t}' density equal to non-existent species '{self.species_dict[t].density_equal_to}'"
        for t in self.T_equalto_keys:
            assert self.species_dict[t].temperature_equal_to in self.species_dict.keys(), f"Error: cannot set '{t}' temperature equal to non-existent species '{self.species_dict[t].temperature_equal_to}'"

        # create member labeling adiabatic species
        if adiabatic_species_count == 1:
            self.adiabatic_species = self.species_dict[adiabatic_type]
            self.has_adiabatic_species = True

        # create member labeling reference species
        if reference_species_count == 0:
            self.ref_species = self.species_dict[first_ion_type]
        else:
            self.ref_species = self.species_dict[reference_type]

        # create member labeling electron species
        self.electron = self.species_dict['electron']

        # create member labeling bulk ion species
        if bulk_ion_count == 0:
            # default bulk to first ion species in input file
            self.bulk_ion = self.species_dict[first_ion_type]
            bulk_ion_type = first_ion_type
        else:
            self.bulk_ion = self.species_dict[bulk_ion_type]
        self.other_ions_dict = OrderedDict(self.species_dict)
        del self.other_ions_dict['electron']
        del self.other_ions_dict[bulk_ion_type]

        # create member labeling impurity species
        if impurity_count == 1:
            self.impurity = self.species_dict[impurity_type]
        else:
            self.impurity = None

        self.first_ion_type = first_ion_type

        self.evolve_count = len(self.n_evolve_keys) + len(self.T_evolve_keys)

        self.clear_fluxes()
        self.clear_sources()

    def __getitem__(self, s_type):
        '''
        accessor that allows a particular element (Species) of the species_dict dictionary to be accessed via species[s_type] == species.species_dict[s_type]
        s_type is a String
        '''
        return self.species_dict[s_type]

    def remove(self, s_type, in_place=False):
        '''
        removes an element (Species) from the SpeciesDict
        s_type is a String
        if in_place == False, a new SpeciesDict object will be returned and the calling SpeciesDict will not be modified
        if in_place == True, the calling SpeciesDict will be modified
        '''
        if in_place:
            new_species = self
        else:
            new_species = copy.deepcopy(self)

        del new_species.species_dict[s_type]
        new_species.N_species -= 1
        if s_type in new_species.n_evolve_keys:
            new_species.n_evolve_keys.remove(s_type)
        if s_type in new_species.T_evolve_keys:
            new_species.T_evolve_keys.remove(s_type)

        return new_species

    def get_species_list(self):
        '''
        return list of all elements of species_dict. each element is a Species class member
        '''

        return self.species_dict.values()

    def print_info(self):
        info("  Species Information")
        info(f"    This calculation contains {[s for s in self.species_dict]} species.")
        info(f"    The '{self.bulk_ion.type}' species is treated as the bulk ions.")
        if self.impurity:
            info(f"    The '{self.impurity.type}' species is treated as the impurity.")
        info(f"    Evolving densities: {self.n_evolve_keys}")
        for t in self.n_equalto_keys:
            info(f"    The '{t}' density will be kept equal to {self.species_dict[t].density_equal_to_frac} * {self.species_dict[t].density_equal_to} density.")
        if self.qneut_species:
            info(f"    The '{self.qneut_species.type}' density will be set by quasineutrality.")
        info(f"    Zeff = {self.Zeff().profile}")
        info(f"    Evolving temperatures: {self.T_evolve_keys}")
        for t in self.T_equalto_keys:
            info(f"    The '{t}' temperature will be kept equal to {self.species_dict[t].temperature_equal_to_frac} * {self.species_dict[t].temperature_equal_to} temperature.")

        if self.has_adiabatic_species:
            info(f"    The '{self.adiabatic_species.type}' species will be treated adiabatically.")

        info(f"    Using '{self.ref_species.type}' as the reference species for turbulence calculations.")

        info(f"    Total number of (parallelizable) flux tube calculations per step = {(self.N_radial-1)*(1 + self.evolve_count)}.")

    def init_profiles_and_sources(self, geo, norms, imported=None):
        for s in self.get_species_list():
            s.init_profiles(self.grid, geo, norms, imported)

        # enforce equal_to densities and temperatures
        self.enforce_equal_to_density(fixed_T=True, init=True)
        self.enforce_equal_to_temperature(init=True)

        # if impurity species and Zeff_target, enforce
        if self.impurity is not None and self.impurity.Zeff_target is not None:
            self.enforce_Zeff_target(fixed_T=True)

        # Initialize the same linked/charge-dependent manifold used for every
        # trial, then store the constrained reference values for frozen profiles.
        temperatures = {s.type: s.T() for s in self.get_species_list()}
        self._enforce_evolution_constraints()
        for s in self.get_species_list():
            s.p_prof = s.n()*temperatures[s.type]
        self._enforce_evolution_temperatures()
        for s in self.get_species_list():
            s.n_init = GridProfile(s.n().profile.copy(), self.grid)
            s.T_init = GridProfile(s.T().profile.copy(), self.grid)
        if not self.profiles_are_physical():
            raise ValueError("Initial density and pressure must be positive and finite for every species")

        if self.evolve_count > 0:
            self.nt_vec = self.get_vec_from_profs()

        self.norms = norms

        # Compute beta_prime (to be used only if beta_prime is set
        # to be a constant rather than evolving with species)
        if geo.beta_prime_const:
            if geo.force_beta_prime_species:
                self.beta_prime = geo.beta_prime
            else:
                beta_ref = self.ref_species.beta(geo.Btor).toFluxProfile().profile
                ns, Ts, _ = self.get_profiles_on_flux_grid(normalize=True, a_minor=geo.a_minor)
                kns, kTs, _, _ = self.get_grads_on_flux_grid(pert_n=None, pert_T=None)
                self.beta_prime = -beta_ref * np.sum(ns*Ts*(kns+kTs), axis=0)
        self.Paux = GridProfile(0, self.grid)
        for s in self.get_species_list():
            s.init_aux_sources(self.grid, geo, norms, imported)
            self.Paux += s.Sp_labeled['aux']

    def get_vec_from_profs(self):
        '''
        copy and concatenate values from evolving species profiles into a numpy vector
        '''
        nt_vec = []

        for t in self.n_evolve_keys:
            nt_vec.append(self.species_dict[t].n().profile)

        for t in self.T_evolve_keys:
            nt_vec.append(self.species_dict[t].p().profile)

        nt_vec = np.concatenate(nt_vec) if nt_vec else np.empty(0)

        return nt_vec

    def enforce_quasineutrality(self, fixed_T=False):
        charge_density = np.zeros(self.N_radial)

        # compute charge density without the quasineutral species
        for s in self.get_species_list():
            if s.type != self.qneut_species.type:
                charge_density = charge_density + s.Z*s.n()

        # use quasineutrality to set density of last evolved species.
        # this needs to happen outside the above loop to ensure charge_density
        # has been computed correctly.
        # note that the charge_density does not include a contribution from the 'quasineutral' species,
        # so that we can compute the following
        if self.qneut_species:
            T = self.qneut_species.T()
            self.qneut_species.set_n(-charge_density/self.qneut_species.Z)

            # keep temperature fixed
            if fixed_T:
                self.qneut_species.set_p(self.qneut_species.n()*T)

    def enforce_Zeff_target(self, fixed_T=False):
        """Initialize one impurity and dilute the original ions proportionally."""
        impurity, electron = self.impurity, self.electron
        Z = impurity.Z
        target = impurity.Zeff_target
        others = [s for s in self.get_species_list() if s not in (electron, impurity)]
        ne = electron.n()
        charge = sum((s.Z*s.n() for s in others), GridProfile(0., self.grid))
        if (not np.all(np.isfinite(ne.profile) & (ne.profile > 0))
                or not np.allclose(charge.profile, ne.profile, rtol=1e-9, atol=0)):
            raise ValueError('Zeff initialization requires the non-impurity ions and electrons to balance charge first')
        if impurity.density_equal_to is not None or electron.density_equal_to is not None:
            raise ValueError('Zeff initialization cannot combine equal_to density for its impurity or electrons')
        if any(s.density_equal_to == impurity.type for s in others):
            raise ValueError('Zeff initialization cannot combine background ion density linked to its impurity')

        zbg = sum((s.Z**2*s.n() for s in others), GridProfile(0., self.grid))/ne
        # z_target = lambda*z_background + (1-lambda)*Z_impurity.
        if not np.isfinite(target) or not np.all((zbg.profile < target) & (target < Z)):
            raise ValueError('Zeff target must lie strictly between the background Zeff and impurity charge')
        dilution = (Z-target)/(Z-zbg)
        electron_links = [s for s in others if s.density_equal_to == electron.type]
        if electron_links and not np.allclose(dilution.profile, dilution[0], rtol=1e-12, atol=0):
            raise ValueError('Zeff target needs radial dilution incompatible with a constant equal_to electron density ratio')
        for s in electron_links:
            s.density_equal_to_frac *= dilution[0]
        impurity.set_n(ne*(1-dilution)/Z, fixed_T=fixed_T)
        for s in others:
            s.set_n(s.n()*dilution, fixed_T=fixed_T)

    def enforce_equal_to_density(self, fixed_T=False, init=False):
        for s in self.get_species_list():
            if s.density_equal_to is not None:
                T = s.T()
                n_eq = s.density_equal_to_frac * self[s.density_equal_to].n()
                s.n_prof = n_eq
                if fixed_T:
                    s.p_prof = s.n_prof * T

        if init:
            for s in self.get_species_list():
                if s.density_init_to is not None:
                    T = s.T()
                    n_eq = s.density_init_to_frac * self[s.density_init_to].n()
                    s.n_prof = n_eq
                    if fixed_T:
                        s.p_prof = s.n_prof * T

    def enforce_equal_to_temperature(self, init=False):
        for s in self.get_species_list():
            if s.temperature_equal_to is not None:
                p = s.p()
                T_eq = s.temperature_equal_to_frac * self[s.temperature_equal_to].T()
                s.p_prof = T_eq * s.n()
        if init:
            for s in self.get_species_list():
                if s.temperature_init_to is not None:
                    p = s.p()
                    T_eq = s.temperature_init_to_frac * self[s.temperature_init_to].T()
                    s.p_prof = T_eq * s.n()

    def _enforce_evolution_constraints(self):
        """Apply proportional profiles and quasineutrality at the same state.

        Express every density as n_s = a_s*n_q + b_s, then solve the
        charge constraint for n_q. This also handles proportional profiles
        linked to the charge-dependent species or listed in reverse order.
        """
        q = self.qneut_species
        density_terms = {q.type: (1.0, 0.0)}
        active = set()

        def density_terms_for(key):
            if key not in density_terms:
                if key in active:
                    raise ValueError("Cycle in species density equal_to constraints")
                active.add(key)
                s = self[key]
                if s.density_equal_to is None:
                    terms = (0.0, s.n())
                else:
                    a, b = density_terms_for(s.density_equal_to)
                    terms = (s.density_equal_to_frac*a, s.density_equal_to_frac*b)
                density_terms[key] = terms
                active.remove(key)
            return density_terms[key]

        charge_a = 0.0
        charge_b = GridProfile(0.0, self.grid)
        for s in self.get_species_list():
            a, b = density_terms_for(s.type)
            charge_a += s.Z*a
            charge_b += s.Z*b
        if charge_a == 0:
            raise ValueError("Density equal_to constraints make the quasineutral density underdetermined")
        n_q = -charge_b/charge_a
        for s in self.get_species_list():
            if s is q or s.density_equal_to is not None:
                a, b = density_terms[s.type]
                s.n_prof = a*n_q + b

    def _enforce_evolution_temperatures(self):
        active = set()
        temperatures = {}

        def temperature_for(key):
            if key not in temperatures:
                if key in active:
                    raise ValueError("Cycle in species temperature equal_to constraints")
                active.add(key)
                s = self[key]
                if s.temperature_equal_to is None:
                    T = s.T()
                else:
                    T = s.temperature_equal_to_frac*temperature_for(s.temperature_equal_to)
                    s.p_prof = T*s.n()
                temperatures[key] = T
                active.remove(key)
            return temperatures[key]

        for key in self.T_equalto_keys:
            temperature_for(key)

    def get_profs_from_vec(self, nt_vec):
        """Unpack independent n and p values; keep the boundary node fixed."""
        for i, key in enumerate(self.n_evolve_keys):
            start = i*self.N_radial
            self[key].n().profile[:-1] = nt_vec[start:start+self.N_radial-1]
        for s in self.get_species_list():
            if not s.evolve_density:
                s.n_prof = s.n_init
        self._enforce_evolution_constraints()

        for i, key in enumerate(self.T_evolve_keys, start=self.N_dens_profiles):
            start = i*self.N_radial
            self[key].p().profile[:-1] = nt_vec[start:start+self.N_radial-1]
        for s in self.get_species_list():
            if not s.evolve_temperature:
                s.p_prof = s.T_init*s.n()
        self._enforce_evolution_temperatures()

    def profile_scratch(self):
        """An isolated set of profile values; never modifies backend state."""
        scratch = copy.copy(self)
        scratch.species_dict = OrderedDict()
        for s in self.get_species_list():
            item = copy.copy(s)
            item.n_prof = GridProfile(s.n().profile.copy(), self.grid)
            item.p_prof = GridProfile(s.p().profile.copy(), self.grid)
            item.Sn_labeled = s.Sn_labeled.copy()
            item.Sp_labeled = s.Sp_labeled.copy()
            scratch.species_dict[s.type] = item
        for key in ("qneut_species", "ref_species", "electron", "bulk_ion", "impurity", "adiabatic_species"):
            item = getattr(self, key, None)
            if item is not None:
                setattr(scratch, key, scratch[item.type])
        scratch.other_ions_dict = OrderedDict((key, scratch[key]) for key in self.other_ions_dict)
        return scratch

    def profiles_are_physical(self):
        for s in self.get_species_list():
            for arr in (s.n().profile, s.p().profile):
                if not np.all(np.isfinite(arr) & (arr > 0)):
                    return False
        return True

    def get_profiles_on_grid(self, normalize=False, pert_n=None, pert_T=None, pert_idx=slice(None), rel_step=0.1):

        if pert_n is not None and pert_T is not None:
            raise ValueError("Perturb one independent profile at a time")
        idx = np.arange(self.N_radial)[pert_idx]
        state = self
        df = np.zeros(self.N_radial)
        if pert_n is not None or pert_T is not None:
            key = pert_n if pert_n is not None else pert_T
            keys = self.n_evolve_keys if pert_n is not None else self.T_evolve_keys
            if key not in keys or np.any(np.asarray(idx) == self.N_radial-1):
                raise ValueError("Profile perturbations require an independent, non-boundary degree of freedom")
            offset = 0 if pert_n is not None else self.N_dens_profiles
            start = (offset + keys.index(key))*self.N_radial
            y = self.get_vec_from_profs()
            df[idx] = rel_step*(self[key].n() if pert_n is not None else self[key].T()).profile[idx]
            y[start+idx] += df[idx] if pert_n is not None else df[idx]*self[key].n().profile[idx]
            state = self.profile_scratch()
            state.get_profs_from_vec(y)
            if not state.profiles_are_physical():
                raise ValueError("Profile perturbation produced nonpositive density or pressure")
        ns = np.stack([s.n().profile for s in state.get_species_list()])
        Ts = np.stack([s.T().profile for s in state.get_species_list()])
        if normalize:
            n_ref = self.ref_species.n().profile
            T_ref = self.ref_species.T().profile
            ns, Ts = ns/n_ref, Ts/T_ref
            if pert_n is not None:
                df /= n_ref
            if pert_T is not None:
                df /= T_ref
        return ns, Ts, df[pert_idx]

    def get_profiles_on_flux_grid(self, normalize=False, a_minor=1.0):

        ns = np.zeros((self.N_species, self.N_radial-1))
        Ts = np.zeros((self.N_species, self.N_radial-1))
        nus = np.zeros((self.N_species, self.N_radial-1))

        if normalize:
            # reference values for flux tube calculation
            n_ref = self.ref_species.n().toFluxProfile()  # 10^20 m^-3
            T_ref = self.ref_species.T().toFluxProfile()  # keV
            vt_ref = self.ref_species.norms.vT_ref*T_ref**0.5  # m/s

        for i, s in enumerate(self.get_species_list()):
            n = s.n().toFluxProfile()
            T = s.T().toFluxProfile()
            nu_ss = s.collision_frequency(s).toFluxProfile()

            if normalize:
                n = n/n_ref
                T = T/T_ref
                nu_ss = nu_ss*a_minor/vt_ref

            ns[i,:] = n
            Ts[i,:] = T
            nus[i,:] = nu_ss

        return ns, Ts, nus

    def get_grads_on_flux_grid(self, pert_n=None, pert_T=None, abs_step=None):
        """Return local gradient probes along the independent n and p directions."""
        all_species = self.get_species_list()
        shape = (self.N_species, self.N_radial-1)
        kns = np.zeros(shape)
        kts = np.zeros(shape)
        kps = np.zeros(shape)
        dkap = np.zeros(shape[1])
        rows = {}
        for i, s in enumerate(all_species):
            rows[s.type] = i
            kn, kt, kp = s.get_fluxgrads()
            kns[i] = kn.profile if hasattr(kn, "profile") else kn
            kts[i] = kt.profile if hasattr(kt, "profile") else kt
            kps[i] = kp.profile if hasattr(kp, "profile") else kp
        if pert_n is None and pert_T is None:
            return kns, kts, kps, dkap
        if pert_n is not None and pert_T is not None:
            raise ValueError("A flux probe must perturb either density or pressure, not both")

        key = pert_n if pert_n is not None else pert_T
        probed = self[key]
        base = kns[rows[key]] if pert_n is not None else kts[rows[key]]
        dkap = np.maximum(probed.rel_step*np.abs(base), probed.abs_step if abs_step is None else abs_step)
        if not np.all(np.isfinite(dkap) & (dkap > 0) & (base+dkap != base)):
            raise ValueError(f"The {key!r} gradient probe must be finite and nonzero; use abs_step > 0")
        dkn = np.zeros(shape)
        if pert_n is not None:
            dkap *= -1
            # The same linear density constraints also hold for dn/drho.
            # Convert the perturbation of dln(n)/drho into dn/drho, solve
            # the linear constraints, then convert each species back.
            q = self.qneut_species
            terms = {} if q is None else {q.type: (1.0, 0.0)}

            def coefficients(name):
                if name not in terms:
                    item = self[name]
                    link = getattr(item, "density_equal_to", None)
                    if link is None:
                        terms[name] = (0.0, float(name == key))
                    else:
                        a, b = coefficients(link)
                        terms[name] = (item.density_equal_to_frac*a, item.density_equal_to_frac*b)
                return terms[name]

            q_response = 0.0
            if q is not None:
                a = sum(item.Z*coefficients(item.type)[0] for item in all_species)
                b = sum(item.Z*coefficients(item.type)[1] for item in all_species)
                q_response = -b/a
            nk = probed.n().toFluxProfile().profile
            for item in all_species:
                a, b = coefficients(item.type)
                response = a*q_response + b
                if response != 0:
                    dkn[rows[item.type]] = response*nk/item.n().toFluxProfile().profile*dkap

        dkt = {}
        def temperature_response(item):
            if item.type not in dkt:
                link = getattr(item, "temperature_equal_to", None)
                if link is not None:
                    dkt[item.type] = temperature_response(self[link])
                elif item.type == pert_T:
                    dkt[item.type] = dkap
                elif getattr(item, "evolve_temperature", True):
                    dkt[item.type] = -dkn[rows[item.type]]
                else:
                    dkt[item.type] = np.zeros_like(dkap)
            return dkt[item.type]

        for item in all_species:
            i = rows[item.type]
            dt = temperature_response(item)
            kns[i] += dkn[i]
            kts[i] += dt
            kps[i] += dkn[i] + dt
        return kns, kts, kps, dkap

    def get_pressure_SI(self):
        '''
        return Trinity pressure, summed over species, in SI
        This is used for launching Equilibrium updates
        '''
        p_ref = self.norms.p_ref
        species_list = self.get_species_list()
        p_SI = np.sum([s.p_prof for s in species_list], axis=0) * p_ref

        return p_SI

    def get_masses(self, normalize=False, drop=None):
        '''
        return a numpy array of species masses. masses will be normalized to reference species mass if normalize=True
        '''
        if drop is not None:
            ms = np.zeros(self.N_species-1)
        else:
            ms = np.zeros(self.N_species)
        if normalize:
            m_ref = self.ref_species.mass
        else:
            m_ref = 1.0

        i = 0
        for s in self.get_species_list():
            if s.type != drop:
                ms[i] = s.mass/m_ref
                i += 1

        return ms

    def get_charges(self, normalize=False, drop=None):
        '''
        return a numpy array of species charges. charges will be normalized to reference species charge if normalize=True
        '''
        if drop is not None:
            Zs = np.zeros(self.N_species-1)
        else:
            Zs = np.zeros(self.N_species)
        if normalize:
            Z_ref = self.ref_species.Z
        else:
            Z_ref = 1.0

        i = 0
        for s in self.get_species_list():
            if s.type != drop:
                Zs[i] = s.Z/Z_ref
                i += 1

        return Zs

    def Zeff(self):
        Zeff = 0
        for s in self.get_species_list():
            if s.type != 'electron':
                Zeff += s.Z**2*s.n()
        Zeff = Zeff / self['electron'].n()
        return Zeff

    def get_types_ion_electron(self):
        ie_type = []
        for s in self.get_species_list():
            if s.type == "electron":
                ie_type.append("electron")
            else:
                ie_type.append("ion")

        return ie_type

    def clear_fluxes(self):
        for s in self.get_species_list():
            s.pflux = FluxProfile(0., self.grid)
            s.qflux = FluxProfile(0., self.grid)
            s.heat = FluxProfile(0., self.grid)
            for t in self.get_species_list():
                s.dpflux_dkn[t.type] = FluxProfile(0., self.grid)
                s.dpflux_dkT[t.type] = FluxProfile(0., self.grid)
                s.dqflux_dkn[t.type] = FluxProfile(0., self.grid)
                s.dqflux_dkT[t.type] = FluxProfile(0., self.grid)
                s.dheat_dkn[t.type] = FluxProfile(0., self.grid)
                s.dheat_dkT[t.type] = FluxProfile(0., self.grid)

    def add_flux(self, pflux_sj, qflux_sj, heat_sj=None, label=None):
        for i, s in enumerate(self.get_species_list()):
            s.pflux += FluxProfile(pflux_sj[i, :], self.grid)
            s.qflux += FluxProfile(qflux_sj[i, :], self.grid)
            if heat_sj is not None:
                s.heat += FluxProfile(heat_sj[i, :], self.grid)

            if label is not None:
                s.pflux_labeled[label] = FluxProfile(pflux_sj[i, :], self.grid)
                s.qflux_labeled[label] = FluxProfile(qflux_sj[i, :], self.grid)
                if heat_sj is not None:
                    s.heat_labeled[label] = FluxProfile(heat_sj[i, :], self.grid)
                else:
                    s.heat_labeled[label] = FluxProfile(0.0, self.grid)

    def add_dflux_dkn(self, stype, dpflux_dkn_sj, dqflux_dkn_sj, dheat_dkn_sj=None):
        for i, s in enumerate(self.get_species_list()):
            s.dpflux_dkn[stype] += FluxProfile(dpflux_dkn_sj[i, :], self.grid)
            s.dqflux_dkn[stype] += FluxProfile(dqflux_dkn_sj[i, :], self.grid)
            if dheat_dkn_sj is not None:
                s.dheat_dkn[stype] += FluxProfile(dheat_dkn_sj[i, :], self.grid)

    def add_dflux_dkT(self, stype, dpflux_dkT_sj, dqflux_dkT_sj, dheat_dkT_sj=None):
        for i, s in enumerate(self.get_species_list()):
            s.dpflux_dkT[stype] += FluxProfile(dpflux_dkT_sj[i, :], self.grid)
            s.dqflux_dkT[stype] += FluxProfile(dqflux_dkT_sj[i, :], self.grid)
            if dheat_dkT_sj is not None:
                s.dheat_dkT[stype] += FluxProfile(dheat_dkT_sj[i, :], self.grid)

    def clear_sources(self):
        for s in self.get_species_list():
            s.Sn = GridProfile(0, self.grid)
            s.Sp = GridProfile(0, self.grid)
            for t in self.get_species_list():
                s.dSn_dn[t.type] = GridProfile(0, self.grid)
                s.dSn_dT[t.type] = GridProfile(0, self.grid)
                s.dSp_dn[t.type] = GridProfile(0, self.grid)
                s.dSp_dT[t.type] = GridProfile(0, self.grid)

    def add_source(self, Sn_sj, Sp_sj, label=None):
        for i, s in enumerate(self.get_species_list()):
            s.Sn += GridProfile(Sn_sj[i, :], self.grid)
            s.Sp += GridProfile(Sp_sj[i, :], self.grid)

            if label is not None:
                s.Sn_labeled[label] = GridProfile(Sn_sj[i, :], self.grid)
                s.Sp_labeled[label] = GridProfile(Sp_sj[i, :], self.grid)

    def add_dS_dn(self, stype, dSn_dn_sj, dSp_dn_sj):
        for i, s in enumerate(self.get_species_list()):
            s.dSn_dn[stype] += GridProfile(dSn_dn_sj[i, :], self.grid)
            s.dSp_dn[stype] += GridProfile(dSp_dn_sj[i, :], self.grid)

    def add_dS_dT(self, stype, dSn_dT_sj, dSp_dT_sj):
        for i, s in enumerate(self.get_species_list()):
            s.dSn_dT[stype] += GridProfile(dSn_dT_sj[i, :], self.grid)
            s.dSp_dT[stype] += GridProfile(dSp_dT_sj[i, :], self.grid)

    def update_edge(self, physics):
        """Update independently controlled edge values, retaining n/T constraints."""
        trial = self.profile_scratch()
        temperatures = {s.type: s.T() for s in trial.get_species_list()}
        for s in trial.get_species_list():
            n, p, T = s.n(), s.p(), temperatures[s.type]
            if (s.temperature_equal_to is None and p[0]/p[-1] > physics.edge_p_ratio
                    and T[-1] < physics.edge_T_max):
                T[-1] = min(T[-1]*physics.edge_p_multiplier, physics.edge_T_max)
            if (s is not trial.qneut_species and s.density_equal_to is None
                    and n[0]/n[-1] > physics.edge_n_ratio and n[-1] < physics.edge_n_max):
                n[-1] = min(n[-1]*physics.edge_n_multiplier, physics.edge_n_max)

        trial._enforce_evolution_constraints()
        for s in trial.get_species_list():
            s.p_prof = s.n()*temperatures[s.type]
        trial._enforce_evolution_temperatures()
        if not trial.profiles_are_physical():
            raise ValueError('Edge control would create a nonphysical profile')
        changed = False
        for s in self.get_species_list():
            target = trial[s.type]
            nedge, pedge = target.n()[-1], target.p()[-1]
            changed |= not np.allclose((s.n()[-1], s.p()[-1]), (nedge, pedge), rtol=1e-12, atol=0)
        if not changed:
            return False
        for s in self.get_species_list():
            target = trial[s.type]
            s.n()[-1], s.p()[-1] = target.n()[-1], target.p()[-1]
            if not s.evolve_density:
                s.n_init[-1] = s.n()[-1]
            if not s.evolve_temperature:
                s.T_init[-1] = target.T()[-1]
        return True

    def update_source(self, physics, geometry):

        # if Q < 1 increase P_aux, if Q > 2 decrease P_aux

        # load options
        Q_threshold_low = physics.source_Q_threshold_low
        Q_threshold_high = physics.source_Q_threshold_high
        p_source_up_multiplier = physics.source_p_up_multiplier
        p_source_down_multiplier = physics.source_p_down_multiplier
        p_aux_max_MW = physics.source_p_aux_max_MW
        p_aux_min_MW = physics.source_p_aux_min_MW

        species = self.get_species_list()
        controlled = sum((s.Sp_labeled['aux'] for s in species), GridProfile(0., self.grid))
        controlled_MW = geometry.volume_integrate(controlled, active=True)*self.norms.P_ref_MWm3
        if not np.isfinite(controlled_MW) or controlled_MW <= 0:
            warn('Skipping auxiliary source control: no finite positive controllable auxiliary heating.')
            return

        fixed_labels = {'aux', 'alpha', 'rad', 'coll', 'heating'}
        external = sum((value for s in species for key, value in s.Sp_labeled.items()
                        if key not in fixed_labels), GridProfile(0., self.grid))
        external_MW = geometry.volume_integrate(external, active=True)*self.norms.P_ref_MWm3
        P_aux_MW = controlled_MW + external_MW
        P_fus_MW = geometry.volume_integrate(physics.compute_fusion_power(self), active=True)
        if not np.isfinite(P_aux_MW) or P_aux_MW <= 0 or not np.isfinite(P_fus_MW):
            warn('Skipping auxiliary source control: invalid fusion or auxiliary power.')
            return
        Q = P_fus_MW / P_aux_MW
        info(f"P_aux = {P_aux_MW:.2f} MW, P_fus = {P_fus_MW:.2f} MW, Q = {Q:.2f}, n0 = {self['electron'].n_prof[0]:.2f}")

        multiplier = 1.
        if Q < Q_threshold_low and P_aux_MW < p_aux_max_MW:
            multiplier = min(p_source_up_multiplier, (p_aux_max_MW-external_MW)/controlled_MW)
        elif Q > Q_threshold_high and P_aux_MW > p_aux_min_MW:
            multiplier = max(p_source_down_multiplier, (p_aux_min_MW-external_MW)/controlled_MW)
        if multiplier != 1.:
            for s in species:
                s.Sp_labeled['aux'] *= multiplier
            self.Paux = controlled*multiplier



class Species():

    def __init__(self, sp, grid):

        self.type = sp.get('type', "deuterium")
        self.is_adiabatic = sp.get('adiabatic', False)
        self.bulk_ion = sp.get('bulk', False)
        self.use_as_reference = sp.get('use_as_reference', self.bulk_ion)
        self.Zeff_target = sp.get('Zeff', None)
        self.impurity = sp.get('impurity', self.Zeff_target is not None)
        self.qneut = sp.get('qneut_species', False)
        density_parameters = sp.get('density', {})
        temperature_parameters = sp.get('temperature', {})

        # flags controlling density and temperature evolution
        self.evolve_density = density_parameters.get('evolve', True)
        self.evolve_temperature = temperature_parameters.get('evolve', True)
        self.density_equal_to = density_parameters.get('equal_to', None)
        self.temperature_equal_to = temperature_parameters.get('equal_to', None)
        self.rel_step = sp.get('rel_step', 0.5)
        self.abs_step = sp.get('abs_step', 0.5)
        if not np.isfinite(self.rel_step) or self.rel_step < 0 or not np.isfinite(self.abs_step) or self.abs_step < 0:
            raise ValueError("Flux gradient rel_step and abs_step must be finite and nonnegative")

        self.density_init_to = density_parameters.get('init_to', None)
        self.temperature_init_to = temperature_parameters.get('init_to', None)
        # note: the density_equal_to, temperature_equal_to, density_init_to, temperature_init_to parameters can be
        # either strings corresponding to a species, e.g.
        # equal_to = "electron"
        # or a string containing a multiplicative fraction times the species, e.g.
        # equal_to = "0.5 * electron"
        # the following function parses the string to handle both cases

        # thank you ChatGPT for helping write this function...
        def parse_string(input_str):
            # Define a regular expression pattern to match the numeric value and the unit
            pattern = r'(\d+(\.\d+)?([eE][+-]?\d+)?)\s*\*\s*(\w+)'

            # Use re.match to search for the pattern in the input string
            match = re.match(pattern, input_str)
            match2 = re.match(r'\w+', input_str)  # handle case with no multiplier

            if match:
                # Extract the numeric value and unit from the matched groups
                value = float(match.group(1))
                unit = match.group(4)

                return [value, unit]
            elif match2:
                value = 1.0
                unit = match2.group(0)
                return [value, unit]
            else:
                assert False, f"Error parsing string \"{input_str}\""

        # the init_to parameter can be used to initialize a profile
        # as some fraction of another profile. after initialization
        # the profiles will evolve independently
        if self.density_init_to is not None:
            (self.density_init_to_frac, self.density_init_to) = parse_string(self.density_init_to)
        if self.temperature_init_to is not None:
            (self.temperature_init_to_frac, self.temperature_init_to) = parse_string(self.temperature_init_to)

        # the equal_to parameter can be used to maintain a profile
        # as some fraction of another profile, even as the system evolves
        if self.density_equal_to is not None:
            (self.density_equal_to_frac, self.density_equal_to) = parse_string(self.density_equal_to)
        if self.temperature_equal_to is not None:
            (self.temperature_equal_to_frac, self.temperature_equal_to) = parse_string(self.temperature_equal_to)

        self.type_tag = sp.get('tag') or self.get_type_tag()  # for diagnostic output

        # physical parameters
        self.mass = sp.get('mass') or self.get_mass()  # mass in units of proton mass
        self.Z = sp.get('Z') or self.get_charge()      # charge in units of e

        # initial profiles
        # density parameters in units of 10^20 m^-3
        self.n_shape = density_parameters.get('shape', 'parabolic')
        # PARABOLIC PARAMETERS
        self.n_core = density_parameters.get('core', 4)
        self.n_volavg = density_parameters.get('volavg', None)
        self.n_lineavg = density_parameters.get('lineavg', None)
        self.n_alpha = density_parameters.get('alpha', 1)
        self.n_alpha1 = density_parameters.get('alpha1', 2)
        self.n_edge = density_parameters.get('edge', None)
        self.n_sep = density_parameters.get('sep', None)
        # GAUSSIAN
        self.n_height = density_parameters.get('height', 0)
        self.n_int = density_parameters.get('integrated', None)
        self.n_width = density_parameters.get('width', 0.1)
        self.n_center = density_parameters.get('center', 0)
        # EXPONENTIAL
        self.n_coefs = density_parameters.get('coefs', [0,0,0,0])
        if len(self.n_coefs) < 4:  # pad with zeros
            self.n_coefs.extend([0] * (4 - len(self.n_coefs)))
        # IMPORT
        self.n_import = density_parameters.get('import', False)
        self.n_import_key = density_parameters.get('key', None)
        # MANUAL
        self.n_profstr = density_parameters.get('profstr', None)
        # CUSTOM
        if self.n_shape == 'custom':
            # this allows a python function 'func' to be specified in the input file.
            # for example,
            # [[species]]
            # ...
            # density = {shape = 'custom', func = '''
            # def init(rho):
            #     return (1-rho**2)
            # ''', evolve = true}
            # ...
            script = density_parameters.get('func')
            namespace = {}
            exec(script, namespace)
            self.n_init_func = namespace.get('init')

        # temperature parameters in units of keV
        self.T_shape = temperature_parameters.get('shape', 'parabolic')
        # PARABOLIC
        self.T_core = temperature_parameters.get('core', 8)
        self.T_volavg = temperature_parameters.get('volavg', None)
        self.T_lineavg = temperature_parameters.get('lineavg', None)
        self.T_alpha = temperature_parameters.get('alpha', 1)
        self.T_alpha1 = temperature_parameters.get('alpha1', 2)
        self.T_edge = temperature_parameters.get('edge', None)
        self.T_sep = temperature_parameters.get('sep', None)
        # GAUSSIAN
        self.T_height = temperature_parameters.get('height', 0)
        self.T_int = temperature_parameters.get('integrated', None)
        self.T_width = temperature_parameters.get('width', 0.1)
        self.T_center = temperature_parameters.get('center', 0)
        # EXPONENTIAL
        self.T_coefs = temperature_parameters.get('coefs', [0,0,0,0])
        if len(self.T_coefs) < 4:  # pad with zeros
            self.T_coefs.extend([0] * (4 - len(self.T_coefs)))
        # IMPORT
        self.T_import = temperature_parameters.get('import', False)
        self.T_import_key = temperature_parameters.get('key', None)
        # MANUAL
        self.T_profstr = temperature_parameters.get('profstr', None)
        # CUSTOM
        if self.T_shape == 'custom':
            # this allows a python function 'func' to be specified in the input file.
            # for example,
            # [[species]]
            # ...
            # temperature = {shape = 'custom', func = '''
            # def init(rho):
            #     return (1-rho**2)
            # ''', evolve = true}
            # ...
            script = temperature_parameters.get('func')
            namespace = {}
            exec(script, namespace)
            self.T_init_func = namespace.get('init')

        # sources
        # density sources in units of 10^20 m^-3 s^-1
        # can be specified in input file as 'aux_particle_source' or 'density_source'
        density_source = sp.get('aux_particle_source', {})
        density_source = sp.get('density_source', density_source)  # deprecate?
        self.Sn_aux_import = density_source.get('import', False)
        self.Sn_aux_import_keys = density_source.get('keys', None)
        # set some smart defaults
        self.Sn_aux_shape = density_source.get('shape', 'gaussian')
        # parameters for gaussian shape
        self.Sn_aux_height = density_source.get('height', 0)  # units of 10^20 m^-3 s^-1
        self.Sn_aux_int = density_source.get('integrated', None)  # units of 10^20 s^-1
        self.Sn_aux_width = density_source.get('width', 0.1)
        self.Sn_aux_center = density_source.get('center', 0)
        # pressure sources in units of MW/m^3
        # can be specified in input file as 'aux_power_source' or 'pressure_source'
        pressure_source = sp.get('aux_power_source', {})
        pressure_source = sp.get('pressure_source', pressure_source)  # deprecate?
        self.Sp_aux_import = pressure_source.get('import', False)
        self.Sp_aux_import_keys = pressure_source.get('keys', None)
        self.Sp_aux_shape = pressure_source.get('shape', 'gaussian')
        self.Sp_aux_height = pressure_source.get('height', 0)  # units of MW/m^3
        self.Sp_aux_int = pressure_source.get('integrated', None)  # units of MW
        self.Sp_aux_width = pressure_source.get('width', 0.1)
        self.Sp_aux_center = pressure_source.get('center', 0)

        # parameters for ReLU flux model only
        n_relu = sp.get('density_relu_flux', {})
        self.n_relu_critical_gradient0 = n_relu.get('critical_gradient', 1.0)
        self.n_relu_critical_gradient0 = n_relu.get('critical_gradient0', self.n_relu_critical_gradient0)
        self.n_relu_critical_gradient1 = n_relu.get('critical_gradient1', 0.0)
        self.n_relu_flux_slope = n_relu.get('slope', 0.5)
        self.n_relu_noise_frac = n_relu.get('noise_frac', 0.0)
        p_relu = sp.get('pressure_relu_flux', {})
        self.p_relu_critical_gradient0 = p_relu.get('critical_gradient', 2.0)
        self.p_relu_critical_gradient0 = p_relu.get('critical_gradient0', self.p_relu_critical_gradient0)
        self.p_relu_critical_gradient1 = p_relu.get('critical_gradient1', 0.0)
        self.p_relu_flux_slope = p_relu.get('slope', 0.5)
        self.p_relu_flux_slope0 = p_relu.get('slope0', self.p_relu_flux_slope)
        self.p_relu_flux_slope1 = p_relu.get('slope1', 0.0)
        self.p_relu_noise_frac = p_relu.get('noise_frac', 0.0)

        # init flux profiles with zeros. these will be set by SpeciesDict.add_flux
        self.pflux = FluxProfile(0, grid)
        self.qflux = FluxProfile(0, grid)
        self.heat = FluxProfile(0, grid)

        # init flux jacobians as empty dictionaries. these will be set by SpeciesDict.add_dflux_dk*
        self.dpflux_dkn = {}
        self.dqflux_dkn = {}
        self.dheat_dkn = {}
        self.dpflux_dkT = {}
        self.dqflux_dkT = {}
        self.dheat_dkT = {}

        # labeled flux profiles for diagnostics. initialize as empty dictionaries. these will be set by SpeciesDict.add_fluxes
        self.pflux_labeled = {}
        self.qflux_labeled = {}
        self.heat_labeled = {}

        # init source profiles with zeros. these will be set by SpeciesDict.add_sources
        self.Sn = GridProfile(0, grid)
        self.Sp = GridProfile(0, grid)

        self.Sn_labeled = {}
        self.Sn_labeled['aux'] = GridProfile(0, grid)

        self.Sp_labeled = {}
        self.Sp_labeled['aux'] = GridProfile(0, grid)
        self.Sp_labeled['alpha'] = GridProfile(0, grid)
        self.Sp_labeled['rad'] = GridProfile(0, grid)
        self.Sp_labeled['heating'] = GridProfile(0, grid)
        self.Sp_labeled['coll'] = GridProfile(0, grid)

        # init source jacobians as empty dictionaries
        self.dSn_dn = {}
        self.dSn_dT = {}
        self.dSp_dn = {}
        self.dSp_dT = {}

    @staticmethod
    def _manual_profile(profstr, grid, label):
        """Read a two-row radius/value table without numpy.matrix."""
        from scipy.interpolate import PchipInterpolator
        try:
            rows = [list(map(float, row.replace(',', ' ').split())) for row in profstr.split(';')]
            x, val = np.asarray(rows, dtype=float)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError(f'{label} manual profile requires two semicolon-separated numeric rows') from error
        rho = grid.rho/grid.rho_edge
        if (x.size < 2 or not np.all(np.isfinite(x))
                or not np.all(np.diff(x) > 0) or not np.all(np.isfinite(val) & (val > 0))
                or x[0] > rho[0] or x[-1] < rho[-1]):
            raise ValueError(f'{label} manual profile requires positive values on a sorted finite radius covering the grid')
        return GridProfile(PchipInterpolator(x, val, extrapolate=False)(rho), grid)

    def init_profiles(self, grid, geo, norms, imported=None):
        # import initial density from file
        if self.n_import:
            assert imported.imported is not None, 'Error: must specify [import] section to use import = true for density'
            n = imported.get_density(key=self.n_import_key, species_mass=self.mass, species_charge=self.Z, tag=self.type_tag, n_edge=self.n_edge, norms=norms)
            self.n_prof = GridProfile(n, grid)
        # use analytic profile for initial density
        elif self.n_shape == 'parabolic':
            if self.n_edge is None:
                self.n_edge = 0.5  # default value

            if self.n_lineavg:
                n_prof = GridProfile((1 - (grid.rho/grid.rho_edge)**self.n_alpha1)**self.n_alpha, grid)
                self.n_prof = (self.n_lineavg - self.n_edge) / np.average(n_prof) * n_prof + self.n_edge
            elif self.n_volavg:
                n_prof = GridProfile((1 - (grid.rho/grid.rho_edge)**self.n_alpha1)**self.n_alpha, grid)
                self.n_prof = (self.n_volavg - self.n_edge) / geo.volume_average(n_prof) * n_prof + self.n_edge
            elif self.n_core and self.n_sep:
                self.n_prof = GridProfile((self.n_core - self.n_sep)*(1 - (grid.rho)**self.n_alpha1)**self.n_alpha + self.n_sep, grid)
            elif self.n_core:
                self.n_prof = GridProfile((self.n_core - self.n_edge)*(1 - (grid.rho/grid.rho_edge)**self.n_alpha1)**self.n_alpha + self.n_edge, grid)
        elif self.n_shape == 'gaussian':
            Gaussian = np.vectorize(self.Gaussian)
            info(f'Density is Gaussian with width {self.n_width}, center {self.n_center}, height {self.n_height}, int {self.n_int}')
            # particle source
            n_prof = GridProfile(Gaussian(grid.rho, A=1, sigma=self.n_width, x0=self.n_center), grid)  # Trinity units
            if self.n_int:
                info(f'Density using target int {self.n_int}')
                target = self.n_int
                volume_integral = geo.volume_integrate(n_prof, normalized=False, interpolate=False)
                self.n_prof = (target / volume_integral) * n_prof
            else:
                info(f'Density using height {self.n_height}')
                self.n_prof = self.n_height * n_prof
        elif self.n_shape == 'exponential':
            coefs = self.n_coefs
            x = grid.rho
            self.n_prof = GridProfile(coefs[0]*np.exp(-coefs[1]*x**2 - coefs[2]*x**4 - coefs[3]*x**6), grid)
            if self.n_edge:
                self.n_prof = self.n_prof - self.n_prof[-1] + self.n_edge
        elif self.n_shape == 'manual':
            self.n_prof = self._manual_profile(self.n_profstr, grid, 'density')
        elif self.n_shape == 'custom':
            self.n_prof = GridProfile(self.n_init_func(grid.rho), grid)
            del self.n_init_func  # this is necessary to allow Species class to be pickled

            # scale line-average or volume-average
            # note: n_edge (if present) will not be scaled
            if self.n_edge is None:
                self.n_edge = 0.0  # default value
            if self.n_lineavg:
                self.n_prof = (self.n_lineavg - self.n_edge) / np.average(self.n_prof) * self.n_prof + self.n_edge
            elif self.n_volavg:
                self.n_prof = (self.n_volavg - self.n_edge) / geo.volume_average(self.n_prof) * self.n_prof + self.n_edge
        else:
            raise ValueError('Invalid n profile shape in Trinity input')

        # import initial temperature from file
        if self.T_import:
            assert imported.imported is not None, 'Error: must specify [import] section to use import = true for temperature'
            T = imported.get_temperature(key=self.T_import_key, species_mass=self.mass, species_charge=self.Z, tag=self.type_tag, T_edge=self.T_edge, norms=norms)
            T_prof = GridProfile(T, grid)
        # use analytic profile for initial temperature
        elif self.T_shape == 'parabolic':
            if self.T_edge is None:
                self.T_edge = 2  # default value

            if self.T_lineavg:
                T_prof = GridProfile((1 - (grid.rho/grid.rho_edge)**self.T_alpha1)**self.T_alpha, grid)
                T_prof = (self.T_lineavg - self.T_edge) / np.average(T_prof) * T_prof + self.T_edge
            elif self.T_volavg:
                T_prof = GridProfile((1 - (grid.rho/grid.rho_edge)**self.T_alpha1)**self.T_alpha, grid)
                T_prof = (self.T_volavg - self.T_edge) / geo.volume_average(T_prof) * T_prof + self.T_edge
            elif self.T_core and self.T_sep:
                T_prof = GridProfile((self.T_core - self.T_sep)*(1 - (grid.rho)**self.T_alpha1)**self.T_alpha + self.T_sep, grid)
            elif self.T_core:
                T_prof = GridProfile((self.T_core - self.T_edge)*(1 - (grid.rho/grid.rho_edge)**self.T_alpha1)**self.T_alpha + self.T_edge, grid)
        elif self.T_shape == 'gaussian':
            Gaussian = np.vectorize(self.Gaussian)
            info(f'Temperature is Gaussian with width {self.T_width}, center {self.T_center}, height {self.T_height}, int {self.T_int}')
            # particle source
            T_prof = GridProfile(Gaussian(grid.rho, A=1, sigma=self.T_width, x0=self.T_center), grid)  # Trinity units
            if self.T_int:
                info(f'Temperature using target int {self.T_int}')
                target = self.T_int
                volume_integral = geo.volume_integrate(T_prof, normalized=False, interpolate=False)
                T_prof = (target / volume_integral) * T_prof
            else:
                info(f'Temperature using height {self.T_height}')
                T_prof = self.T_height * T_prof
        elif self.T_shape == 'exponential':
            coefs = self.T_coefs
            x = grid.rho
            T_prof = GridProfile(coefs[0]*np.exp(-coefs[1]*x**2 - coefs[2]*x**4 - coefs[3]*x**6), grid)
            if self.T_edge:
                T_prof = T_prof - T_prof[-1] + self.T_edge
        elif self.T_shape == 'manual':
            T_prof = self._manual_profile(self.T_profstr, grid, 'temperature')
        elif self.T_shape == 'custom':
            T_prof = GridProfile(self.T_init_func(grid.rho), grid)
            del self.T_init_func  # this is necessary to allow Species class to be pickled

            # scale line-average or volume-average
            # note: T_edge (if present) will not be scaled
            if self.T_edge is None:
                self.T_edge = 0.0  # default value.
            if self.T_lineavg:
                T_prof = (self.T_lineavg - self.T_edge) / np.average(T_prof) * T_prof + self.T_edge
            elif self.T_volavg:
                T_prof = (self.T_volavg - self.T_edge) / geo.volume_average(T_prof) * T_prof + self.T_edge
        else:
            raise ValueError('Invalid T profile shape in Trinity input')

        # pressure is quantity that is actually stored (not temperature)
        self.p_prof = self.n_prof * T_prof

        # save a copy of initial profiles separately
        self.n_init = self.n_prof
        self.T_init = T_prof
        self.p_init = self.p_prof

        self.norms = norms

    def n(self):
        return self.n_prof

    def p(self):
        return self.p_prof

    def T(self):
        return self.p_prof / self.n_prof

    def beta(self, Btor):
        # compute beta
        # p_prof is in units of 10^20 m^-3*keV
        # Btor is in units of T

        # convert p and B to cgs
        p_cgs = self.p_prof*1e17
        B_cgs = Btor*1e4
        return 4.03e-11*p_cgs/(B_cgs*B_cgs)

    def nu_equil(self, other):
        s = self
        u = other
        nu_su = self.collision_frequency(u)
        return self.norms.t_ref*8./(3*np.sqrt(np.pi))*nu_su*(u.mass/s.mass)**0.5*(u.mass/s.mass + u.T()/s.T())**-1.5

    def collision_frequency(self, other):
        # recall:
        # n_prof is in units of 10^20 m^-3
        # T_prof is in units of keV

        Z_s = self.Z
        m_s = self.mass
        n_s = self.n()  # 10^20 m^-3
        T_s = self.T()  # keV

        Z_u = other.Z
        m_u = other.mass
        n_u = other.n()  # 10^20 m^-3
        T_u = other.T()  # keV

        logLambda = self.logLambda(other)

        # nu_su = 4*pi*(q_s*q_u)**2*n_u*logLambda / (m_s**0.5*(2*T_s)**1.5)
        nu = 2.85e2*(Z_s*Z_u)**2/m_s**0.5*logLambda*n_u/T_s**1.5  # 1/s

        return nu

    def logLambda(self, other):
        # recall:
        # n_prof is in units of 10^20 m^-3
        # T_prof is in units of keV
        # the below formula is from the NRL formulary, which is in cgs units (except for temperatures in eV)

        Z_s = self.Z
        m_s = self.mass*m_proton_cgs
        n_s = self.n().profile*1e14  # 10^20 m^-3 -> cm^-3
        T_s = self.T().profile*1e3  # keV -> eV

        Z_u = other.Z
        m_u = other.mass*m_proton_cgs
        n_u = other.n().profile*1e14  # 10^20 m^-3 -> cm^-3
        T_u = other.T().profile*1e3  # keV -> eV

        if self.type == "electron" and other.type == "electron":  # ee
            lamb = 23.5 - np.log(n_s**0.5 / T_s**1.25) \
                - np.sqrt(1e-5 + (np.log(T_s) - 2)**2 / 16)
        elif self.type == "electron":  # ei
            lamb = 24.0 - np.log(n_s**0.5 / T_s)
        elif other.type == "electron":  # ie
            lamb = 24.0 - np.log(n_u**0.5 / T_u)
        else:  # ii
            lamb = 23.0 - np.log(
                (Z_s * Z_u) * (m_s + m_u) /
                (m_s * T_u + m_u * T_s) *
                (n_s * Z_s**2 / T_s + n_u * Z_u**2 / T_u)**0.5)
        return lamb

    def set_n(self, n, fixed_T=False):
        T = self.T()
        self.n_prof = n
        if fixed_T:
            self.p_prof = self.n_prof * T

    def set_T(self, T):
        self.p_prof = self.n_prof*T

    def set_p(self, p):
        self.p_prof = p

    def get_fluxgrads(self):
        kap_n_grid = -1*self.n_prof.log_gradient()
        kap_p_grid = -1*self.p_prof.log_gradient()
        kap_T_grid = kap_p_grid - kap_n_grid

        return kap_n_grid.toFluxProfile(), kap_T_grid.toFluxProfile(), kap_p_grid.toFluxProfile()

    # flux vector F_n (normalized particle flux)
    def Fn_plus(self, ref_species, geo, label=None):
        # returns FluxProfile with fluxes evaluated at j+1/2
        if label is not None:
            pflux = self.pflux_labeled[label]
        else:
            pflux = self.pflux
        if self.evolve_density:
            return (geo.area/geo.Btor**2).plus()*(ref_species.p()**1.5/ref_species.n()**0.5).plus()*pflux.plus()
        else:
            return 0*pflux

    def Fn_minus(self, ref_species, geo, label=None):
        # Shift the complete throughput to share identical faces and have an
        # exactly zero inner boundary (no particle flow through the axis).
        return self.Fn_plus(ref_species, geo, label=label).minus()

    def Fn_grid(self, ref_species, geo, label=None):
        # returns GridProfile with fluxes evaluated at j
        if label is not None:
            pflux = self.pflux_labeled[label]
        else:
            pflux = self.pflux
        if self.evolve_density:
            return (geo.area/geo.Btor**2*pflux).toGridProfile()*(ref_species.p()**1.5/ref_species.n()**0.5)
        else:
            return 0*pflux.toGridProfile()

    # flux vector F_p (normalized heat flux)
    def Fp_plus(self, ref_species, geo, label=None):
        # returns FluxProfile with fluxes evaluated at j+1/2
        if label is not None:
            qflux = self.qflux_labeled[label]
        else:
            qflux = self.qflux
        if self.evolve_temperature:
            return 2./3.*(geo.area/geo.Btor**2).plus()*(ref_species.p().plus()**2.5/ref_species.n().plus()**1.5)*qflux.plus()
        else:
            return 0*qflux

    def Fp_minus(self, ref_species, geo, label=None):
        # Shift the complete interface throughput: adjacent control volumes
        # must use the same value. Interpolation and nonlinear normalization
        # of p**2.5/n**1.5 do not commute.
        return self.Fp_plus(ref_species, geo, label=label).minus()

    def Fp_grid(self, ref_species, geo, label=None):
        # returns GridProfile with fluxes evaluated at j
        if label is not None:
            qflux = self.qflux_labeled[label]
        else:
            qflux = self.qflux
        if self.evolve_temperature:
            return 2./3.*(geo.area/geo.Btor**2*qflux).toGridProfile()*(ref_species.p()**2.5/ref_species.n()**1.5)
        else:
            return 0*qflux.toGridProfile()

    def coll_eq(self, species):
        s = self
        coll = 0*s.n()
        for u in species.get_species_list():
            coll += s.p()*s.nu_equil(u)*(s.n()*u.p()/(u.n()*s.p()) - 1)
        return coll

    def init_aux_sources(self, grid, geo, norms, imported=None):
        # sources
        # particle source
        if not self.evolve_density:
            self.Sn_labeled['aux'] = GridProfile(0, grid)
        elif self.Sn_aux_import:
            assert imported.imported is not None, 'Error: must specify [import] section to use import = true for density_source'
            aux_source_n = imported.get_density_source(keys=self.Sn_aux_import_keys, species_mass=self.mass, species_charge=self.Z, tag=self.type_tag, norms=norms)
            self.Sn_labeled['aux'] = GridProfile(aux_source_n, grid)
        elif self.Sn_aux_shape == 'gaussian':
            self.Sn_labeled['aux'] = self._gaussian_aux_source(
                grid, geo, center=self.Sn_aux_center, width=self.Sn_aux_width,
                height=self.Sn_aux_height, integrated=self.Sn_aux_int,
                unit_factor=norms.Sn_ref_SI20, label='aux_particle_source')
        else:
            raise ValueError('Invalid density_source profile shape in Trinity input')

        # power source
        if not self.evolve_temperature:
            self.Sp_labeled['aux'] = GridProfile(0, grid)
        elif self.Sp_aux_import:
            assert imported.imported is not None, 'Error: must specify [import] section to use import = true for pressure_source'
            aux_source_p = imported.get_pressure_source(keys=self.Sp_aux_import_keys, species_mass=self.mass, species_charge=self.Z, tag=self.type_tag, norms=norms)
            self.Sp_labeled['aux'] = GridProfile(aux_source_p, grid)
        elif self.Sp_aux_shape == 'gaussian':
            self.Sp_labeled['aux'] = self._gaussian_aux_source(
                grid, geo, center=self.Sp_aux_center, width=self.Sp_aux_width,
                height=self.Sp_aux_height, integrated=self.Sp_aux_int,
                unit_factor=norms.P_ref_MWm3, label='aux_power_source')
        else:
            raise ValueError('Invalid pressure_source profile shape in Trinity input')

    @staticmethod
    def _gaussian_aux_source(grid, geo, *, center, width, height, integrated, unit_factor, label):
        """Average the Gaussian over each evolved shell using its discrete Jacobian."""
        amplitude = integrated if integrated is not None else height
        if amplitude == 0:
            return GridProfile(0., grid)
        if not (np.isfinite(amplitude) and np.isfinite(center) and np.isfinite(width) and width > 0):
            raise ValueError(f'{label}: a nonzero Gaussian requires finite amplitude and center and positive finite width.')

        # The far face of the last evolved cell is a midpoint; rho_edge is a
        # Dirichlet node, not a cell.
        faces = np.r_[0., grid.midpoints]
        if width < grid.drho:
            warn(f'{label}: Gaussian width {width:g} is smaller than drho={grid.drho:g}; '
                 'shell averages preserve deposited strength but the radial shape is not resolved.')

        # Differences of erf in the core, and erfc in the tails to avoid
        # subtracting two values rounded to ±1. The omitted scale cancels
        # for integrated targets, but matters for a prescribed peak height.
        with np.errstate(over='ignore'):
            z = (faces - center) / width / np.sqrt(2)
        lower, upper = z[:-1], z[1:]
        mass = erf(upper) - erf(lower)
        right = lower > 0.5
        left = upper < -0.5
        mass[right] = erfc(lower[right]) - erfc(upper[right])
        mass[left] = erfc(-upper[left]) - erfc(-lower[left])
        if integrated is None:
            average = (mass * width) * np.sqrt(np.pi/2) / grid.drho
            return GridProfile(np.r_[average * (height / unit_factor), 0.], grid)

        if not 0. <= center <= faces[-1]:
            # An explicit integrated target means power/particles deposited
            # in the simulated domain, not the full Gaussian in all space.
            retained = 0.5 * mass.sum()
            exterior = (f'{label}: integrated Gaussian center {center:g} is outside the evolved '
                        f'interval [0, {faces[-1]:g}]; rho_edge={grid.rho[-1]:g} is a fixed boundary node.')
            # A many-sigma exterior tail is not a usable deposition shape.
            min_fraction = np.sqrt(np.finfo(float).eps)
            if retained <= min_fraction:
                raise ValueError(f'{exterior} The radial Gaussian fraction inside ({retained:.3g}) '
                                 f'is <= {min_fraction:.3g}; move the center inward or increase width.')
            warn(f'{exterior} Only {retained:.3%} of the radial Gaussian lies inside; '
                 'renormalizing that part to the requested active-domain integral.')

        source = GridProfile(np.r_[mass, 0.], grid)
        volume = geo.volume_integrate(source, active=True)
        if not np.isfinite(volume) or volume <= 0.:
            raise ValueError(f'{label}: cannot normalize a Gaussian with zero or invalid evolved-shell weight.')
        return source * (integrated / unit_factor / volume)

    # for a particle and heat sources
    def Gaussian(self, x, A=2, sigma=.3, x0=0):
        exp = - ((x - x0) / sigma)**2 / 2
        return A * np.e ** exp

    def get_mass(self):
        '''
        Look-up table for species mass in units of proton mass.
        '''
        _dict = {'hydrogen': 1.0,
                 'deuterium': 2.0,
                 'tritium': 3.0,
                 'helium': 4.0,
                 'boron': 10.811,
                 'carbon': 12.011,
                 'oxygen': 15.999,
                 'neon': 20.18,
                 'krypton': 83.798,
                 'tungsten': 183.84,
                 'electron': 0.000544617021}
        key = self.type
        if key in _dict:
            return _dict[key]
        else:
            assert False, f"species '{key}' has unknown mass. use mass parameter in input file (with units of proton mass)."

    def get_charge(self):
        '''
        Look-up table for species charge in units of e.
        '''
        _dict = {'hydrogen': 1,
                 'deuterium': 1,
                 'tritium': 1,
                 'helium': 2,
                 'boron': 5,
                 'carbon': 6,
                 'oxygen': 8,
                 'neon': 10,
                 'krypton': 36,
                 'tungsten': 72,
                 'electron': -1}
        key = self.type
        if key in _dict:
            return float(_dict[key])
        else:
            assert False, f"species '{key}' has unknown charge. use Z parameter in input file (with units of e)."

    def get_type_tag(self):
        '''
        Look-up table for short tag name for diagnostic output
        '''
        _dict = {'hydrogen': 'H',
                 'deuterium': 'D',
                 'tritium': 'T',
                 'helium': 'He',
                 'boron': 'B',
                 'carbon': 'C',
                 'oxygen': 'O',
                 'neon': 'Ne',
                 'krypton': 'Kr',
                 'tungsten': 'W',
                 'electron': 'e'}
        key = self.type
        if key in _dict:
            return _dict[key]
        else:
            return key  # use full name
