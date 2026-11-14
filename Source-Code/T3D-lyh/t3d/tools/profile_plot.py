import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from netCDF4 import Dataset
import adios2
import itertools
from os import getcwd
from enum import Enum

from t3d import Profiles, Grid


class plotter:

    def __init__(self):

        # for backwards compat to before t3d packaging
        sys.modules['Profiles'] = Profiles
        sys.modules['Grid'] = Grid

        np.set_printoptions(linewidth=500)

        self.grid_lines = False
        self.savefig = False

        # Dictionary of plotting functions for the legacy panel plot
        self.panel_plots = {
            "density": self.plot_density_profiles,
            "temperature": self.plot_temperature_profiles,
            "pressure": self.plot_pressure_profiles,
            "power_balance": self.plot_power_balance,
            "heat_flux": self.plot_heat_flux,
            "particle_balance": self.plot_particle_balance,
            "particle_flux": self.plot_particle_flux,
            "fusion_power_density": self.plot_fusion_power_density,
            "radiation": self.plot_radiation_profiles,
            "exchange": self.plot_exchange,
            "power_source": self.plot_auxiliary_power_source,
            "q": self.plot_q,
            "particle_source": self.plot_auxiliary_particle_source,
            "gamma": self.plot_gamma,
        }

        # Dictionary of all available plotting functions
        self.plotting_functions = self.panel_plots.copy()
        self.plotting_functions["state_profiles"] = self.plot_state_profiles
        self.plotting_functions["flux"] = self.plot_flux
        self.plotting_functions["source_total"] = self.plot_source_total
        self.plotting_functions["sink_terms"] = self.plot_sink_terms

    def list_available_plots(self):
        ''' Function to list available plots '''
        print("\nList of available plots:")
        for plot in self.plotting_functions:
            print(f"  {plot}")

    def read_output(self, output_data_file: str):
        """Function to read the output data which could be a numpy pickle file or in NetCDF4 or ADIOS2 format"""

        output_data_file = str(output_data_file)
        self.output_data_file = output_data_file

        class ReadableOutputFormat(Enum):
            PICKLE = 0
            NETCDF4 = 1
            ADIOS2 = 2

        def _read_data(group_name, variable_name, *, as_array=True, verbose=False):
            """
            Read data from a group in a data container.

            Parameters:
            group_name (str): Name of the group containing the variable.
            variable_name (str): Name of the variable to read.
            as_array (bool, optional): Whether to return the data as a NumPy array. Defaults to True.
            verbose (bool, optional): Whether to print the variable name and value. Defaults to False.

            Returns:
            Union[Any, np.ndarray]: The read data, either as a single value, a list of strings, or a NumPy array.
            """
            try:
                match output_format:
                    case ReadableOutputFormat.PICKLE:
                        if not as_array:
                            try:
                                val = data[group_name][variable_name]
                            except (KeyError, TypeError) as e:
                                val = data[variable_name]
                        else:
                            try:
                                val = data[group_name][variable_name]
                            except (KeyError, TypeError) as e:
                                val = data[variable_name]
                            if not (isinstance(val, list) and all(isinstance(v, str) for v in val)):
                                val = np.array(val)

                    case ReadableOutputFormat.NETCDF4:
                        if not as_array:
                            val = data[group_name].variables[variable_name][...]
                            if isinstance(val, np.ndarray) and val.size == 1:
                                val = val.item()
                        else:
                            val = data[group_name].variables[variable_name][:]
                            if isinstance(val, (np.ndarray)) and all(isinstance(v, str) for v in val):
                                val = val.tolist()

                    case ReadableOutputFormat.ADIOS2:
                        try:
                            val = data.read_attribute(f"{group_name}/{variable_name}")
                        except Exception:
                            val = data.read(f"{group_name}/{variable_name}")

                    case _:  # default case
                        raise ValueError("Unknown file format")

            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"Failed to read {variable_name} from {group_name}: {e}")

            if verbose:
                print(f'{variable_name} = {val}')

            return val

        # Define mapping of file extensions to their format identifier, output file root, and the data loader function
        file_handlers = {
            "log.npy": (ReadableOutputFormat.PICKLE, lambda fin: (fin[:-7], np.load(fin, allow_pickle=True).tolist())),
            "npy":     (ReadableOutputFormat.PICKLE, lambda fin: (fin[:-3], np.load(fin, allow_pickle=True).tolist())),
            "nc":      (ReadableOutputFormat.NETCDF4, lambda fin: (fin[:-2], Dataset(fin, "r", format="NETCDF4"))),
            "bp":      (ReadableOutputFormat.ADIOS2, lambda fin: (fin[:-2], adios2.FileReader(fin))),
            "bp/":     (ReadableOutputFormat.ADIOS2, lambda fin: (fin[:-3], adios2.FileReader(fin))),
        }

        # Determine file type and load data
        file_extension = next((ext for ext in file_handlers if output_data_file.endswith(ext)), None)
        if file_extension:
            output_format, data_loader = file_handlers[file_extension]
            froot, data = data_loader(output_data_file)
        else:
            raise ValueError("Unknown file format")
        self.fileout = getcwd() + '/' + froot[:-1]  # Added -1 to remove '.' at the end

        # Get the filename
        match output_format:
            case ReadableOutputFormat.PICKLE:
                self.filename = data['trinity_infile']
            case ReadableOutputFormat.NETCDF4:
                self.filename = data.infile
            case ReadableOutputFormat.ADIOS2:
                self.filename = data.read_attribute('infile')
            case _:  # default case
                raise ValueError("Unknown file format")

        # Norms
        self.t_ref = _read_data('norms', 't_ref', as_array=False)
        try:
            self.rho_star = _read_data('norms', 'rho_star', as_array=False)
        except KeyError:
            self.rho_star = 1 / _read_data('norms', 'gyro_scale', as_array=False)
        self.P_ref_MWm3 = _read_data('norms', 'P_ref_MWm3', as_array=False)
        self.Sn_ref_SI20 = _read_data('norms', 'Sn_ref_SI20', as_array=False)

        # Time
        self.time = _read_data('time', 't') * self.t_ref
        self.N = len(self.time)
        try:
            self.t_iter_idx = _read_data('time', 't_iter_idx')
            self.t_step_idx = list(_read_data('time', 't_step_idx'))
            self.t_rms = _read_data('time', 't_rms')
            # Row 0 of a physical step is the accepted profile from the
            # previous step, not the last Newton iterate at its old time.
            first = {}
            for i, (step, iteration) in enumerate(zip(self.t_step_idx, self.t_iter_idx)):
                if iteration == 0:
                    first.setdefault(step, i)
            self.t_step_idx_converged = np.asarray(list(first.values()), dtype=int)
        except (KeyError, ValueError, TypeError):
            self.t_iter_idx = np.zeros(self.N, dtype=int)
            self.t_step_idx = list(range(self.N))
            self.t_step_idx_converged = np.arange(self.N, dtype=int)
            self.t_rms = np.full(self.N,np.nan)
        try:
            self.t_diagnostics_valid = np.asarray(_read_data('time', 't_diagnostics_valid'), dtype=bool)
        except (KeyError, ValueError, TypeError):
            self.t_diagnostics_valid = np.ones(self.N,dtype=bool)
        try:
            self.t_arrival_order = np.asarray(_read_data('time', 't_arrival_order'), dtype=int)
        except (KeyError, ValueError, TypeError):
            self.t_arrival_order = None  # legacy output did not record order

        # Grid
        try:
            self.rho = _read_data('grid', 'rho')
        except KeyError:
            self.rho = _read_data('grid', 'rho_axis')
        try:
            self.midpoints = _read_data('grid', 'midpoints')
        except KeyError:
            self.midpoints = _read_data('grid', 'mid_axis')
        try:
            self.flux_label = _read_data('grid', 'flux_label', as_array=False)
        except Exception:
            self.flux_label = _read_data('norms', 'flux_label', as_array=False)
        if self.flux_label == 'rminor':
            self.rad_label = r'$r/a$'
        elif self.flux_label == 'torflux':
            self.rad_label = r'$\rho_{tor}$'
        self.drho = _read_data('grid', 'drho', as_array=False)

        # Geo
        self.area = _read_data('geo', 'area')
        self.grho = _read_data('geo', 'grho')
        self.G_fac = _read_data('geo', 'G_fac')
        self.a_minor = _read_data('geo', 'a_minor', as_array=False)

        # Geo
        self.Btor = _read_data('geo', 'Btor', as_array=False)
        self.a_minor = _read_data('geo', 'a_minor', as_array=False)
        self.R_major = _read_data('geo', 'R_major', as_array=False)
        self.AspectRatio = self.R_major / self.a_minor

        # Bulk ion tag
        try:
            self.ion_tag = _read_data('species', 'bulk_ion_tag', as_array=False)
        except Exception:
            # Special NetCDF4 backwards compatibility case
            self.ion_tag = data['species'].variables['tags'].bulk_ion_tag

        # Specie profiles
        self.n = _read_data('species', 'n_e')
        self.pi = _read_data('species', 'p_' + self.ion_tag)
        self.pe = _read_data('species', 'p_e')
        self.Ti = _read_data('species', 'T_' + self.ion_tag)
        self.Te = _read_data('species', 'T_e')
        self.aLn = _read_data('species', 'aLn_' + self.ion_tag)
        self.aLpi = _read_data('species', 'aLp_' + self.ion_tag)
        self.aLpe = _read_data('species', 'aLp_e')
        self.aLTi = _read_data('species', 'aLT_' + self.ion_tag)
        self.aLTe = _read_data('species', 'aLT_e')

        # Fluxes
        try:
            self.flux_model_labels = _read_data('species', 'flux_model_labels')
        except Exception:
            self.flux_model_labels = ['turb', 'neo']  # backwards compat
        self.Gamma = {}
        self.Qi = {}
        self.Qe = {}
        self.Gamma['tot'] = _read_data('species', 'pflux_' + self.ion_tag)
        self.Qi['tot'] = _read_data('species', 'qflux_' + self.ion_tag)
        self.Qe['tot'] = _read_data('species', 'qflux_e')
        for label in self.flux_model_labels:
            self.Gamma[label] = _read_data('species', 'pflux_' + label + '_' + self.ion_tag)
            self.Qi[label] = _read_data('species', 'qflux_' + label + '_' + self.ion_tag)
            self.Qe[label] = _read_data('species', 'qflux_' + label + '_e')

        # Density sources, in code units, multiply by Sn_ref_SI20
        try:
            self.particle_source_labels = list(_read_data('species', 'particle_source_labels'))
        except Exception:
            self.particle_source_labels = ['aux']  # backwards compat
        if not self.particle_source_labels:
            self.particle_source_labels = ['aux']  # backwards compat
        self.particle_source_labels.append('tot')
        self.Sn_i = {}
        self.Sn_e = {}
        for label in self.particle_source_labels:
            try:
                self.Sn_i[label] = _read_data('species', 'Sn_' + label + '_' + self.ion_tag)
                self.Sn_e[label] = _read_data('species', 'Sn_' + label + '_e')
            except Exception:
                pass  # Some older files don't have Sn_tot_e/i
        self.Sn_int_SI20_i = {}
        self.Sn_int_SI20_e = {}
        for label in self.particle_source_labels:
            try:
                self.Sn_int_SI20_i[label] = _read_data('species', 'Sn_' + label + '_int_SI20_' + self.ion_tag)
                self.Sn_int_SI20_e[label] = _read_data('species', 'Sn_' + label + '_int_SI20_e')
            except Exception:
                pass  # Some older files don't have Sn_tot_int_MW_e/i
        self.Sn_cumint_SI20_i = {}
        self.Sn_cumint_SI20_e = {}
        for label in self.particle_source_labels:
            try:
                self.Sn_cumint_SI20_i[label] = _read_data('species', 'Sn_' + label + '_cumint_SI20_' + self.ion_tag)
                self.Sn_cumint_SI20_e[label] = _read_data('species', 'Sn_' + label + '_cumint_SI20_e')
            except Exception:
                pass  # Some older files don't have these outputs

        # Power sources, in code units, multiply by P_ref_MWm3 to get MW/m^3
        try:
            self.power_source_labels = list(_read_data('species', 'power_source_labels'))
        except Exception:
            self.power_source_labels = ['aux', 'alpha', 'rad', 'heating', 'coll']  # backwards compat
        if not self.power_source_labels:
            self.power_source_labels = ['aux', 'alpha', 'rad', 'heating', 'coll']  # backwards compat
        self.power_source_labels.append('tot')
        self.Sp_i = {}
        self.Sp_e = {}
        for label in self.power_source_labels:
            try:
                self.Sp_i[label] = _read_data('species', 'Sp_' + label + '_' + self.ion_tag)
                self.Sp_e[label] = _read_data('species', 'Sp_' + label + '_e')
            except Exception:
                pass  # Some older files don't have Sp_tot_e/i
        self.Sp_int_MW_i = {}
        self.Sp_int_MW_e = {}
        for label in self.power_source_labels:
            try:
                self.Sp_int_MW_i[label] = _read_data('species', 'Sp_' + label + '_int_MW_' + self.ion_tag)
                self.Sp_int_MW_e[label] = _read_data('species', 'Sp_' + label + '_int_MW_e')
            except Exception:
                pass  # Some older files don't have Sp_tot_int_e/i
        self.Sp_aux_int_MW_tot = self.Sp_int_MW_e['aux'] + self.Sp_int_MW_i['aux']
        self.Sp_cumint_MW_i = {}
        self.Sp_cumint_MW_e = {}
        for label in self.power_source_labels:
            self.Sp_cumint_MW_i[label] = _read_data('species', 'Sp_' + label + '_cumint_MW_' + self.ion_tag)
            self.Sp_cumint_MW_e[label] = _read_data('species', 'Sp_' + label + '_cumint_MW_e')

        # Alpha heating
        self.Palpha_MWm3 = _read_data('species', 'Palpha_MWm3')
        self.Palpha_int_MW = _read_data('species', 'Palpha_int_MW')

        # Power balance
        self.Q_MW_i = {}
        self.Q_MW_e = {}
        self.Q_MW_i['tot'] = _read_data('species', 'Q_MW_' + self.ion_tag)
        self.Q_MW_e['tot'] = _read_data('species', 'Q_MW_e')
        for label in self.flux_model_labels:
            self.Q_MW_i[label] = _read_data('species', 'Q_' + label + '_MW_' + self.ion_tag)
            self.Q_MW_e[label] = _read_data('species', 'Q_' + label + '_MW_e')

        # Particle balance
        self.Gam_SI20_i = {}
        self.Gam_SI20_e = {}
        self.Gam_SI20_i['tot'] = _read_data('species', 'Gam_SI20_' + self.ion_tag)
        self.Gam_SI20_e['tot'] = _read_data('species', 'Gam_SI20_e')
        for label in self.flux_model_labels:
            self.Gam_SI20_i[label] = _read_data('species', 'Gam_' + label + '_SI20_' + self.ion_tag)
            self.Gam_SI20_e[label] = _read_data('species', 'Gam_' + label + '_SI20_e')

        # Zero-d / fusion metrics
        self.N_SI20_e = _read_data('zeroD', 'N_SI20_e')
        self.N_SI20_i = _read_data('zeroD', 'N_SI20_' + self.ion_tag)
        self.N_tot_SI20 = _read_data('zeroD', 'Ntot_SI20')
        self.W_MJ_e = _read_data('zeroD', 'W_MJ_e')
        self.W_MJ_i = _read_data('zeroD', 'W_MJ_' + self.ion_tag)
        self.Wtot_MJ = _read_data('zeroD', 'Wtot_MJ')

        # Close the data file if NetCDF4 or ADIOS2
        if output_format != ReadableOutputFormat.PICKLE:
            data.close()

    def list_profile_times(self):
        ''' Function to list the available profile times and index number that are not from Newton iterations '''
        for i in range(self.N):
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            print(f'Profile #{i} at time {t_curr:.6e} s')

    def plot_density_profiles(self, axs):
        ''' Function to plot the density profiles '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            # plot profiles
            label = None
            if i == 0 or i == self.tidx[-1]:
                label = f'$n_e$, $t =${t_curr:.2f} s'
            axs.plot(self.rho, self.n[i], '.-', color=self.green_map[i], label=label)
        nmax = np.max(self.n)
        axs.set_ylim(bottom=0, top=1.5 * nmax)
        axs.set_xlim(left=0.0)
        axs.set_xlabel(self.rad_label)
        axs.set_title(r'density [10$^{20}$ m$^{-3}$]')
        leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_temperature_profiles(self, axs):
        ''' Function to plot the temperature profiles '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            # plot profiles
            ti_label, te_label = None, None
            if i == 0 or i == self.tidx[-1]:
                ti_label = f'$T_i$, $t =${t_curr:.2f} s'
                te_label = f'$T_e$, $t =${t_curr:.2f} s'
            axs.plot(self.rho, self.Ti[i], '.-', color=self.warm_map[i], label=ti_label)
            if i == 0 or self.plot_electrons:
                axs.plot(self.rho, self.Te[i], '.-', color=self.cool_map[i], label=te_label)

        axs.set_ylim(bottom=0)
        axs.set_xlim(left=0.0)
        axs.set_xlabel(self.rad_label)
        axs.set_title('temperature [keV]')
        leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_pressure_profiles(self, axs):
        ''' Function to plot the pressure profiles '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            # plot profiles
            axs.plot(self.rho, self.pi[i], '.-', color=self.warm_map[i])
            if self.plot_electrons:
                axs.plot(self.rho, self.pe[i], '.-', color=self.cool_map[i])
        axs.set_ylim(bottom=0)
        axs.set_xlim(left=0.0)
        axs.set_xlabel(self.rad_label)
        axs.set_title(r'pressure [10$^{20}$m$^{-3}$ keV]')
        axs.grid(self.grid_lines)

    def plot_power_balance(self, axs):
        ''' Function to plot the power balance '''
        i = self.tidx[-1]
        # Ions
        axs.plot(self.midpoints, self.Q_MW_i['tot'][i], 'o-', color=self.warm_map[i], fillstyle='none', label='$Q_i\\cdot A_{{surf}}$ (total)')
        marker = itertools.cycle(('d', 's', 'v', 'P', '*', 'X'))
        for label in self.flux_model_labels:
            axs.plot(self.midpoints, self.Q_MW_i[label][i], '-', marker=next(marker), color=self.warm_map[i], fillstyle='none', label=f'$Q_i\\cdot A_{{surf}}$ ({label})')
        axs.plot(self.midpoints, self.Sp_cumint_MW_i['tot'][i][:-1], 'o:', color=self.warm_map[i], label=r'$\int S_{p,i}$')
        # Electrons
        axs.plot(self.midpoints, self.Q_MW_e['tot'][i], 'o-', color=self.cool_map[i], fillstyle='none', label='$Q_e\\cdot A_{{surf}}$ (total)')
        marker = itertools.cycle(('d', 's', 'v', 'P', '*', 'X'))
        for label in self.flux_model_labels:
            axs.plot(self.midpoints, self.Q_MW_e[label][i], '-', marker=next(marker), color=self.cool_map[i], fillstyle='none', label=f'$Q_e\\cdot A_{{surf}}$ ({label})')
        axs.plot(self.midpoints, self.Sp_cumint_MW_e['tot'][i][:-1], 'o:', color=self.cool_map[i], label=r'$\int S_{p,e}$')
        # Plot layout
        _, top = axs.get_ylim()
        axs.set_ylim(bottom=0, top=1.6 * top)
        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        axs.set_title('power balance [MW]')
        numitems = len(list(axs.get_legend_handles_labels()[1]))
        nrows = int(numitems/2)
        ncols = int(np.ceil(numitems / float(nrows)))
        leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=ncols, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_heat_flux(self, axs):
        ''' Function to plot the heat flux '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            axs.plot(self.midpoints, self.Qi['tot'][i], 'x-', color=self.warm_map[i])
            axs.plot(self.midpoints, self.Qe['tot'][i], 'x-', color=self.cool_map[i])
        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        axs.set_title('heat flux [GB]')
        axs.grid(self.grid_lines)

    def plot_particle_balance(self, axs):
        ''' Function to plot the particle balance '''
        i = self.tidx[-1]
        axs.plot(self.midpoints, self.Gam_SI20_e['tot'][i], 'o-', color=self.green_map[i], fillstyle='none', label=r'$\Gamma_e\cdot A_{{surf}}$')
        marker = itertools.cycle(('d', 's', 'v', 'P', '*', 'X'))
        for label in self.flux_model_labels:
            axs.plot(self.midpoints, self.Gam_SI20_e[label][i], '-', marker=next(marker), color=self.green_map[i], fillstyle='none', label=f'$\\Gamma_e\\cdot A_{{surf}}$ ({label})')
        axs.plot(self.midpoints, self.Sn_cumint_SI20_e['tot'][i][:-1], 'o:', color=self.green_map[i], label=r'$\int S_{ne}$')
        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        axs.set_title('particle balance [10$^{20}$/s]')
        leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_particle_flux(self, axs):
        ''' Function to plot the particle flux '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            axs.plot(self.midpoints, self.Gamma['tot'][i], 'x-', color=self.green_map[i])
        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        axs.set_title('particle flux [GB]')
        axs.grid(self.grid_lines)

    def plot_fusion_power_density(self, axs):
        ''' Function fusion power density '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue

            spi_label, spe_label, sptot_label = None, None, None
            if i == self.tidx[-1]:
                spi_label, spe_label, sptot_label = '$S_{p,i}$', '$S_{p,i}$', '$S_{p,{tot}}$'
            axs.plot(self.rho, self.Sp_i['alpha'][i] * self.P_ref_MWm3, '.-', color=self.warm_map[i], label=spi_label)
            axs.plot(self.rho, self.Sp_e['alpha'][i] * self.P_ref_MWm3, '.-', color=self.cool_map[i], label=spe_label)
            axs.plot(self.rho, self.Palpha_MWm3[i], '.-', color=self.green_map[i], label=sptot_label)

        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        axs.set_title('fusion power density \n [MW/m$^{3}$]')
        title = (r'$P_{\alpha} = $' + f'{self.Palpha_int_MW[i]:.2f} MW\n' +
                 r'$P_{fus} = $' + f'{5.03 * self.Palpha_int_MW[i]:.2f} MW')
        leg = axs.legend(loc='best', title=title, fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_radiation_profiles(self, axs):
        ''' Function to plot the radiation profiles '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue

            spi_label, spe_label = None, None
            if i == self.tidx[-1]:
                spi_label, spe_label = '$S_{p,i}$', '$S_{p,e}$'
            axs.plot(self.rho, self.Sp_i['rad'][i] * self.P_ref_MWm3, '.-', color=self.warm_map[i], label=spi_label)
            axs.plot(self.rho, self.Sp_e['rad'][i] * self.P_ref_MWm3, '.-', color=self.cool_map[i], label=spe_label)

        axs.set_title('radiation [MW/m$^{3}$]')
        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        try:
            leg = axs.legend(loc='best', title=r'$P_{rad} = $' + f"{self.Sp_int_MW_e['rad'][-1]:.2f} MW", fancybox=False, shadow=False, ncol=1, fontsize=8)
            leg.get_frame().set_edgecolor('k')
            leg.get_frame().set_linewidth(0.65)
        except:
            pass
        axs.grid(self.grid_lines)

    def plot_exchange(self, axs):
        ''' Function to plot the collisional and turbulent exchange terms '''
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue

            scoli_label, scole_label = None, None
            sturbi_label, sturbe_label = None, None
            if i == self.tidx[-1]:
                scoli_label, scole_label = r'$S^\mathrm{coll}_{p,i}$', r'$S^\mathrm{coll}_{p,e}$'
                sturbi_label, sturbe_label = r'$S^\mathrm{turb}_{p,i}$', r'$S^\mathrm{turb}_{p,e}$'
            axs.plot(self.rho, self.Sp_i['coll'][i] * self.P_ref_MWm3, '.-', color=self.warm_map[i], label=scoli_label)
            axs.plot(self.rho, self.Sp_e['coll'][i] * self.P_ref_MWm3, '.-', color=self.cool_map[i], label=scole_label)
            axs.plot(self.rho, self.Sp_i['heating'][i] * self.P_ref_MWm3, '.:', color=self.warm_map[i], label=sturbi_label)
            axs.plot(self.rho, self.Sp_e['heating'][i] * self.P_ref_MWm3, '.:', color=self.cool_map[i], label=sturbe_label)

        axs.set_title('collisional and turbulent \n exchange [MW/m$^{3}$]')
        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        leg = axs.legend(loc='best', title=r'$P^{coll}_i = $' + f"{self.Sp_int_MW_i['coll'][-1]:.2f} MW", fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_auxiliary_power_source(self, axs):
        ''' Function to plot auxiliary power source, convert from Trinity units to MW/m3 '''
        i = self.tidx[-1]
        axs.plot(self.rho, self.Sp_i['aux'][i] * self.P_ref_MWm3, '.-', color=self.warm_map[i], label='$S_{p,i}$')
        axs.plot(self.rho, self.Sp_e['aux'][i] * self.P_ref_MWm3, '.-', color=self.cool_map[i], label='$S_{p,e}$')
        _, top = axs.get_ylim()
        axs.set_ylim(bottom=0, top=1.5 * top)
        axs.set_xlim(left=0.0)
        axs.set_title('auxiliary power source \n [MW/m$^{3}$]')
        axs.set_xlabel(self.rad_label)
        leg = axs.legend(loc='best', title='$P_{aux} =$' + f"{self.Sp_aux_int_MW_tot[i]:.2f} MW", fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_q(self, axs):
        ''' Function to plot q '''
        r = self.ridx
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            axs.plot(self.aLpi[i][r] - self.aLn[i][r], self.Qi['tot'][i][r], '.', color=self.warm_map[i])
            axs.plot(self.aLpe[i][r] - self.aLn[i][r], self.Qe['tot'][i][r], '.', color=self.cool_map[i])
        axs.set_title(r'$Q(L_T)$ [GB]')
        axs.set_xlabel('$a/L_T$')
        # leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        # leg.get_frame().set_edgecolor('k')
        # leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_auxiliary_particle_source(self, axs):
        ''' Function to plot the auxiliary particle source '''
        i = self.N - 1
        axs.plot(self.rho, self.Sn_e['aux'][i] * self.Sn_ref_SI20, '.-', color=self.green_map[i], label='$S_{n}$')
        _, top = axs.get_ylim()
        axs.set_ylim(bottom=0, top=1.5 * top)
        axs.set_xlim(left=0.0)
        axs.set_xlabel(self.rad_label)
        axs.set_title('auxiliary particle source \n [10$^{20}$/(m$^{3}$ s)]')
        axs.grid(self.grid_lines)

    def plot_gamma(self, axs):
        ''' Function to plot gamma '''
        r = self.ridx
        for i in self.tidx:
            t_curr = self.time[i]
            if i > 0 and not self.show_newton_iterations:
                if t_curr == self.time[i - 1]:
                    # skip repeated time from Newton iteration
                    continue
            axs.plot(self.aLn[i][r], self.Gamma['tot'][i][r], '.', color=self.green_map[i])
        axs.set_xlabel('$a/L_n$')
        axs.set_title(r'$\Gamma(L_n)$ [GB]')
        axs.grid(self.grid_lines)

    def plot_state_profiles(self, axs, profile='Ti', t_stop=-1):
        ''' Function to plot the density or temperature profiles '''

        if t_stop < 0:
            t_stop = self.N

        t_old = -1
        for ti in np.arange(t_stop):
            t_curr = self.time[ti]
            if (t_curr == t_old):
                # skip majority of prints
                continue
            else:
                t_old = t_curr
                t = ti
            # plot profiles
            label = None
            if profile == 'Ti':
                if t == 0 or t == t_stop-1:
                    label = f'$T_i$, $t =${self.time[t]:.2f} s'
                axs.plot(self.rho, self.Ti[t], '.-', color=self.warm_map[t], label=label)
                axs.set_title('temperature [keV]')

            if profile == 'Te':
                if t == 0 or t == t_stop-1:
                    label = f'$T_e$, $t =${self.time[t]:.2f} s'
                axs.plot(self.rho, self.Te[t], '.-', color=self.cool_map[t], label=label)
                axs.set_title('temperature [keV]')

            if profile == 'ne':
                if t == 0 or t == t_stop-1:
                    label = f'$n_e$, $t =${self.time[t]:.2f} s'
                axs.plot(self.rho, self.n[t], '.-', color=self.green_map[t], label=label)
                axs.set_title(r'density [10$^{20}$ m$^{-3}$]')

        _, top = axs.get_ylim()
        axs.set_ylim(bottom=0, top=1.5*top)
        axs.set_xlim(left=0.0)
        axs.set_xlabel(self.rad_label)
        leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_flux(self, axs, profile='Ti', t_stop=-1):
        '''
        This function used to plot the flux from GX
        now it plots the total flux

        A future version could break out parts in a FOR loop on label
        '''

        if t_stop < 0:
            t_stop = self.N

        t_old = -1
        for ti in np.arange(t_stop):
            t_curr = self.time[ti]
            if (t_curr == t_old):
                # skip majority of prints
                continue
            else:
                t_old = t_curr
                t = ti

            if profile == 'Ti':
                axs.plot(self.aLpi[t] - self.aLn[t], self.Qi['tot'][t], '.', color=self.warm_map[t])
                axs.set_title(r'$Q(L_{T_i})$ [GB]')
                axs.set_xlabel('$a/L_{{T_i}}$')

            if profile == 'Te':
                axs.plot(self.aLpe[t] - self.aLn[t], self.Qe['tot'][t], '.', color=self.cool_map[t])
                axs.set_title(r'$Q(L_{T_e})$ [GB]')
                axs.set_xlabel('$a/L_{{T_e}}$')

            if profile == 'ne':
                axs.plot(self.aLn[t], self.Gamma['tot'][t], '.', color=self.green_map[t])
                axs.set_xlabel('$a/L_n$')
                axs.set_title(r'$\Gamma(L_n)$ [GB]')
        # leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        # leg.get_frame().set_edgecolor('k')
        # leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_source_total(self, axs, profile='ne', t=-1):
        ''' Plots total source and sink terms, for each profile '''

        if t < 0:
            t = self.N - 1
        elif t == 0:
            return
        else:
            t = t-1

        if profile == 'Ti':

            Spi_aux = self.Sp_i['aux'][t] * self.P_ref_MWm3
            Spi_tot = self.Sp_int_MW_i['tot'][t]  # for some reason Sp_aux is excluded?
            axs.plot(self.rho, Spi_tot + Spi_aux, 'x--', color=self.warm_map[t], label=r'$S_{p,i}^{tot}$')

            axs.plot(self.midpoints, self.Q_MW_i['tot'][t], 'o:', color=self.warm_map[t], fillstyle='none', label='$-Q_i^{tot}$')
            axs.set_title('ion total source [MW]')

        if profile == 'Te':
            Spe_aux = self.Sp_e['aux'][t] * self.P_ref_MWm3
            Spe_tot = self.Sp_int_MW_e['tot'][t]  # for some reason Sp_aux is excluded?
            axs.plot(self.rho, Spe_tot + Spe_aux, 'x--', color=self.cool_map[t], label=r'$S_{p,e}^{tot}$')

            axs.plot(self.midpoints, self.Q_MW_e['tot'][t], 'o:', color=self.cool_map[t], fillstyle='none', label='$-Q_e^{tot}$')
            axs.set_title('electron total source [MW]')

        if profile == 'ne':
            axs.plot(self.midpoints, self.Sn_cumint_SI20_e['tot'][t][:-1], 'x--', color=self.green_map[t], label=r'$S_n$')
            axs.plot(self.midpoints, self.Gam_SI20_e['tot'][t], 'o:', color=self.green_map[t], fillstyle='none', label=r'$-\Gamma$')
            axs.set_title('ne total source [10$^{20}$/s]')

        axs.set_xlabel(self.rad_label)
        axs.set_xlim(left=0.0)
        leg = axs.legend(loc='best', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def plot_sink_terms(self, axs, profile='Ti', t=-1):
        ''' Plots source and sink term-by-term, for each profile '''

        if t < 0:
            t = self.N - 1
        elif t == 0:
            return
        else:
            t = t-1

        if profile == 'Ti':
            marker = itertools.cycle(('d', 's', 'v', 'P', '*', 'X'))
            for label in ['tot'] + self.flux_model_labels:
                axs.plot(self.midpoints,
                         self.Q_MW_i[label][t],
                         '-', marker=next(marker),
                         color=self.warm_map[t],
                         fillstyle='full',
                         label=rf'$Q_{{p,i}}^{{\mathrm{{{label}}}}}$')

            marker = itertools.cycle(('d', 's', 'v', 'P', '*', 'X'))
            for label in self.power_source_labels:
                axs.plot(self.rho,
                         self.Sp_cumint_MW_i[label][t] * self.P_ref_MWm3,
                         '-', marker=next(marker),
                         color=self.cool_map[t],
                         fillstyle='full',
                         label=rf'$S_{{p,i}}^{{\mathrm{{{label}}}}}$')

            axs.set_title('ion sink terms [MW]')

        if profile == 'Te':
            marker = itertools.cycle(('d', 's', 'v', 'P', '*', 'X'))
            for label in ['tot'] + self.flux_model_labels:
                axs.plot(self.midpoints,
                         self.Q_MW_e[label][t],
                         '-', marker=next(marker),
                         color=self.warm_map[t],
                         fillstyle='full',
                         label=rf'$Q_{{p,e}}^{{\mathrm{{{label}}}}}$')

            marker = itertools.cycle(('d', 's', 'v', 'P', '*', 'X'))
            for label in self.power_source_labels:
                axs.plot(self.rho,
                         self.Sp_cumint_MW_e[label][t] * self.P_ref_MWm3,
                         '-', marker=next(marker),
                         color=self.cool_map[t],
                         fillstyle='full',
                         label=rf'$S_{{p,e}}^{{\mathrm{{{label}}}}}$')

            axs.set_title('electron sink terms [MW]')

        if profile == 'ne':
            axs.plot(self.midpoints, self.Gam_SI20_e['tot'][t], 'o-', color=self.green_map[t], fillstyle='none', label=r'$\Gamma^{\mathrm{turb}}$')
            axs.set_title('ne sink terms [10$^{20}$/s]')

        # _, top = axs.get_ylim()
        axs.set_xlabel(self.rad_label)
        # axs.set_ylim(bottom=0, top=2*top)
        axs.set_xlim(left=0.0)
        leg = axs.legend(loc='upper left', fancybox=False, shadow=False, ncol=1, fontsize=8)
        leg.get_frame().set_edgecolor('k')
        leg.get_frame().set_linewidth(0.65)
        axs.grid(self.grid_lines)

    def create_color_maps(self):
        if len(self.tidx) == 1:
            self.warm_map = pylab.cm.autumn(np.linspace(0.25, 0.25, self.N))
            self.cool_map = pylab.cm.Blues(np.linspace(1, 1, self.N))
            self.green_map = pylab.cm.YlGn(np.linspace(1, 1, self.N))
        else:
            self.warm_map = pylab.cm.autumn(np.linspace(1, 0.25, self.N))
            self.cool_map = pylab.cm.Blues(np.linspace(0.25, 1, self.N))
            self.green_map = pylab.cm.YlGn(np.linspace(0.25, 1, self.N))

    def make_plots(self, plots, panel=None, figsize=(4, 4), title=None):
        """
        Create individual plots or combine them into a panel of subplots.

        Parameters:
            plots (list of str): List of plots to create.
            panel (list of ints): List of number of columns and rows, if None do not create panel
            figsize (tuple): Size of the figure
        """
        for plot in plots:
            if plot not in self.plotting_functions:
                raise ValueError(f"Specified plot function, {plot}, not in list of available plots")

        self.plot_electrons = True

        # set up color maps
        self.create_color_maps()

        if not panel:
            for plot in plots:
                _, axs = plt.subplots(1, 1, figsize=figsize)
                plot_function = self.plotting_functions.get(plot.lower())
                plot_function(axs)

                if self.savefig:
                    figname = self.fileout + "-" + plot + ".pdf"
                    print("Saving figure: " + figname)
                    plt.savefig(figname)

        else:
            _, axs = plt.subplots(nrows=panel[0], ncols=panel[1], figsize=figsize)
            axs = axs.flatten()
            for ax, plot_function in zip(axs, plots):
                plot_function = self.plotting_functions.get(plot_function)
                plot_function(ax)
            # Hide any unused subplots
            for ax in axs[len(plots):]:
                ax.axis('off')
            plt.suptitle(title)
            plt.tight_layout()
            if self.savefig:
                figname = self.fileout + ".pdf"
                print("Saving figure: " + figname)
                plt.savefig(figname)

        if not self.savefig:
            plt.show()

    def plot_panel(self):
        ''' Create the legacy plot panel '''

        rlabel = rf'[{self.filename}]'
        _, axs = plt.subplots(2, 7, figsize=(65, 8))

        plt.suptitle(rlabel)

        self.plot_electrons = True

        # set up color maps
        self.create_color_maps()

        for i, (_, plot) in enumerate(self.panel_plots.items()):
            plot(axs[int(i / 7), i % 7])

        # Create the plot panel
        plt.tight_layout()
        plt.subplots_adjust(left=0.05,
                            bottom=0.1,
                            right=0.95,
                            top=0.9,
                            wspace=0.4,
                            hspace=0.4)

        if self.savefig:
            figname = self.fileout + "-panel.pdf"
            print("Saving figure: "+figname)
            plt.savefig(figname)
        else:
            plt.show()


def main():

    import argparse

    plots = plotter()

    description = "This program creates plots from the T3D pickle file .npy, NetCDF4 .nc file, or ADIOS2 .bp output"
    epilog = '''\
Usage: to create the plot panel
  t3d-plot [trinity.log.npy]
  t3d-plot [trinity.nc]
  t3d-plot [trinity.bp]

Usage: to create indvidual plots
  t3d-plot [trinity.log.npy] -p density temperature etc
  t3d-plot [trinity.nc] -p density temperature etc
  t3d-plot [trinity.bp] -p density temperature etc
'''
    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filename",
                        help="Specify plot file ")
    parser.add_argument("-p", "--plots",
                        nargs="+",
                        help="Space separated list of plots",
                        default=None)
    parser.add_argument("-t", "--tidx", nargs="+",
                        help="Space separated list of time indices",
                        default=None)
    parser.add_argument("-r", "--ridx", nargs="+",
                        help="Space separated list of rho indices",
                        default=None)
    parser.add_argument("-l", "--list",
                        help="List available plot routines",
                        action="store_true")
    parser.add_argument("-g", "--grid",
                        help="Include grid lines in plot",
                        action="store_true")
    parser.add_argument("-s", "--savefig",
                        help="Save plots to file rather than interactive display",
                        action="store_true")
    parser.add_argument("-i", "--show_newton_iterations", "--show-newton-iterations",
                        help="Plot intermediate Newton iterations",
                        action="store_true", default=False)
    parser.add_argument("--list-times",
                        help="List available profile times and numbers",
                        action="store_true")
    parser.add_argument("--panel",
                        nargs=2,
                        type=int,
                        help="Create a panel plot from a list of plots with (n,m) for rows and columns",
                        default=None)
    parser.add_argument("--figsize",
                        nargs=2,
                        type=int,
                        help="Specify the figure size",
                        default=(4, 4))
    parser.add_argument("--title",
                        help="Specify the figure title for custom panel plots only",
                        default=None)

    args = parser.parse_args()

    if args.list:
        plots.list_available_plots()
        exit(0)

    if args.grid:
        plots.grid_lines = True

    if args.savefig:
        plots.savefig = True

    plots.read_output(args.filename)

    if args.tidx:
        plots.tidx = list(map(int, args.tidx))
    else:
        plots.tidx = np.arange(len(plots.time))

    if args.ridx:
        plots.ridx = list(map(int, args.ridx))
    else:
        plots.ridx = slice(None)

    plots.show_newton_iterations = args.show_newton_iterations

    if args.list_times:
        plots.list_profile_times()
        exit(0)

    if args.panel:
        if not args.plots:
            raise ValueError("Panel plots requires list of plots (-p or --plots)")

    if args.plots is None:
        plots.plot_panel()
    else:
        plots.make_plots(args.plots,
                         args.panel,
                         figsize=(args.figsize[0], args.figsize[1]),
                         title=args.title)


if __name__ == '__main__':
    '''
        This program plots Trinity3D output read from the .npy output
        file and stored in a python dictionary.
    '''

    main()
