import os
import json
from datetime import datetime, time
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
from collections import defaultdict
import copy
from netCDF4 import Dataset
import adios2
from datetime import date
import getpass
from t3d.Logbook import info, errr
from t3d.Profiles import GridProfile

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        errr("Please install tomli or upgrade to Python >= 3.11")
        exit()


@contextmanager
def _atomic_output(filename):
    """Leave the previous checkpoint intact if the new write fails."""
    path = Path(filename)
    temporary = NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False)
    tmp_path = Path(temporary.name)
    temporary.close()
    try:
        yield tmp_path
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _toml_value(value):
    """Write the TOML value types produced by tomllib without losing order."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False).replace(chr(127), r"\u007f")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(x) for x in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{_toml_value(k)} = {_toml_value(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"Unsupported TOML value {value!r}")


def _toml_text(data):
    # Root key/value assignments have to precede every table header; other
    # values can be expressed with TOML's inline tables and arrays.
    scalars = []
    tables = []
    for key, value in data.items():
        key = _toml_value(key)
        if isinstance(value, dict):
            tables.append("[" + key + "]\n" +
                          "".join(f"{_toml_value(k)} = {_toml_value(v)}\n" for k,v in value.items()))
        elif isinstance(value, list) and value and all(isinstance(x, dict) for x in value):
            for table in value:
                tables.append("[[" + key + "]]\n" +
                              "".join(f"{_toml_value(k)} = {_toml_value(v)}\n" for k,v in table.items()))
        else:
            scalars.append(f"{key} = {_toml_value(value)}\n")
    return "\n".join(["".join(scalars), *tables]).lstrip("\n")


class Input:
    """
    This class handles Trinity input files and parameters
    Parameters are set from input file if they exist, otherwise use defaults
    """

    def __init__(self, fin):
        inputs = self.read_input(fin)

        self.input_dict = inputs

    def read_input(self, fin):
        """
        Read TOML input file
        """

        # A filename is the normal input. Retain the inline TOML API, but
        # never reinterpret a malformed or inaccessible file as TOML text.
        if isinstance(fin, str) and ("\n" in fin or "\r" in fin) and not os.path.isfile(fin):
            return tomllib.loads(fin)
        with open(fin, mode="rb") as f:
            return tomllib.load(f)

    def write(self, fout="temp.in", overwrite=False):
        if os.path.exists(fout) and not overwrite:
            info(f"  input exists, skipping write {fout}")
            return
        text = _toml_text(self.input_dict)
        with _atomic_output(fout) as tmp:
            tmp.write_text(text, encoding="utf-8")
        info(f"  wrote input: {fout}")

    def pretty_print(self, entry=""):
        """Print the current input or a single named entry as valid TOML."""
        data = self.input_dict if not entry else {entry: self.input_dict[entry]}
        print(_toml_text(data), end="")


class Logger:
    """
    This class manages T3D output by doing the following:

    -   Initialization data and run data gets saved to a data dictionary, this
        data dictionary gets saved to a numpy pickle file which can be used
        for importing data into another T3D run and for plotting purposes

    -   The name of the output file can be set in the [log] section of the
        T3D input toml file, default name is the input file name with .log.npy

    -   This class also includes output options for ADIOS2 and NetCDF4. These
        will also get written at every step. ADIOS2 output is on by default
        and NetCDF4 is off. These can be controlled in the T3D input toml file
        in the [log] section with
                output_adios2 = true/false
                output_netcdf = true/false
        The default names for these formats are the input file name with a .bp
        extension for ADIOS2 and .nc extension for NetCDF4

    -   Another dictionary is created to facilitate setting units and
        dimensions in the ADIOS2/NetCDF4 output. This dictionary is created
        once at the beginning of the run as the output is dependent of the
        species included in the T3D run

    -   To add new variables to T3D output, the developer needs to include the
        data in the log dictionary in either the store_system or save functions
        and add the units/description in the store_output_metadata function
    """

    def __init__(self, infile, inputs, grid, time, species, geo, physics, norms, flux_models, source_models=None):

        # read log parameters from input file
        log_parameters = inputs.get('log', {})
        default_stem = str(Path(infile).with_suffix(""))
        stem = str(log_parameters.get('f_save', default_stem))
        # A legacy f_save value ending in a known output suffix is also a
        # stem: never point NetCDF, BP and pickle writers at the same path.
        for suffix in ('.log.npy', '.npy', '.nc', '.bp'):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        if not stem or not Path(stem).name or stem.endswith(os.sep):
            raise ValueError("log.f_save must name an output stem")
        self.f_save = stem+'.log.npy'
        self.nc_save = stem+'.nc'
        self.bp_save = stem+'.bp'
        self.output_netcdf = log_parameters.get('output_netcdf', False)
        self.output_adios2 = log_parameters.get('output_adios2', True)
        self.n_write = log_parameters.get('n_write', 1)
        if type(self.n_write) is not int or self.n_write < 1:
            raise ValueError("log.n_write must be a positive integer")

        # saves TRINITY data as a nested dictionary
        log = defaultdict(list)  # note: log IS the saved object in .npy output

        log['trinity_infile'] = infile

        log['species_types'] = [s.type for s in species.get_species_list()]
        log['species_tags'] = [s.type_tag for s in species.get_species_list()]
        log['bulk_ion_tag'] = species.bulk_ion.type_tag
        log['n_evolve_species'] = species.n_evolve_keys
        log['T_evolve_species'] = species.T_evolve_keys
        log['flux_model_labels'] = [f.label for f in flux_models.get_models_list()]
        tags = log['species_tags']
        if len(tags) != len(set(tags)) or 'i' in tags:
            raise ValueError("Diagnostic species tags must be unique; 'i' is reserved for the ion sum.")
        compounds = [f'{label}_{tag}' for label in log['flux_model_labels'] for tag in tags]
        if len(compounds) != len(set(compounds)) or any(key in tags for key in compounds):
            raise ValueError("Output names from flux labels and species tags collide.")
        particles = dict.fromkeys(key for s in species.get_species_list() for key in s.Sn_labeled)
        powers = dict.fromkeys(key for s in species.get_species_list() for key in s.Sp_labeled)
        models = tuple(source_models.get_sources_list()) if source_models is not None else ()
        self.external_power_labels = []
        for model in models:
            label = model.label
            if not isinstance(label, str) or not label or '/' in label or label in particles or label in powers or label == 'tot':
                raise ValueError(f'External source label {label!r} is invalid or conflicts with another diagnostic label.')
            # Reserve a label across both channels, even if a channel is disabled.
            particles[label] = bool(model.particle_source)
            powers[label] = bool(model.power_source)
            if model.power_source:
                self.external_power_labels.append(label)
        log['particle_source_labels'] = [key for key, enabled in particles.items() if enabled is not False]
        log['power_source_labels'] = [key for key, enabled in powers.items() if enabled is not False]
        # Compound names must also be unique: a custom tag like
        # 'int_MW_D' would otherwise alias the integral for tag 'D'.
        for channel, labels, integral_unit in (
            ('Sn', log['particle_source_labels'], 'SI20'),
            ('Sp', log['power_source_labels'], 'MW'),
        ):
            names = set()
            for label in [*labels, 'tot']:
                for variant, variant_tags in (
                    (label, [*tags, 'i']),
                    (f'{label}_int_{integral_unit}', tags),
                    (f'{label}_cumint_{integral_unit}', tags),
                ):
                    for tag in variant_tags:
                        key = f'{channel}_{variant}_{tag}'
                        if key in names:
                            raise ValueError(f'Diagnostic species/source names collide at {key!r}.')
                        names.add(key)

        self.log = log
        self.geo = geo
        self.grid = grid
        self.norms = norms
        self.physics = physics
        self.species = species

        self.auxiliary_heating = bool(self.external_power_labels)
        for s in species.get_species_list():
            if s.Sp_aux_height > 0 or s.Sp_aux_int or s.Sp_aux_import:
                self.auxiliary_heating = True

        self.store_system(grid, time, geo, norms)

        self.store_output_metadata()

    def store_system(self, grid, time, geo, norms):
        """Saves run settings for reproducing runs."""

        # time step info lives here
        time_settings = {}
        time_settings['alpha'] = time.alpha
        time_settings['dtau'] = time.dtau
        time_settings['N_steps'] = time.N_steps
        time_settings['max_newton_iter'] = time.max_newton_iter
        self.log['time'] = time_settings

        # grid info lives here
        grid_settings = {}
        grid_settings['N_radial'] = grid.N_radial
        grid_settings['rho_edge'] = grid.rho_edge
        grid_settings['rho_inner'] = grid.rho_inner
        grid_settings['rho'] = grid.rho
        grid_settings['midpoints'] = grid.midpoints
        grid_settings['drho'] = grid.drho
        grid_settings['flux_label'] = grid.flux_label
        self.log['grid'] = grid_settings
        self.log['grid_obj'] = grid

        geo_settings = {}
        geo_settings['area'] = geo.area.profile.copy()
        geo_settings['grho'] = geo.grho.profile.copy()
        geo_settings['area_grid'] = geo.area_grid.profile.copy()
        geo_settings['grho_grid'] = geo.grho_grid.profile.copy()
        geo_settings['G_fac'] = geo.G_fac.profile.copy()
        geo_settings['Btor'] = geo.Btor
        geo_settings['B0'] = geo.B0
        geo_settings['a_minor'] = geo.a_minor
        geo_settings['R_major'] = geo.a_minor * geo.AspectRatio
        self.log['geo'] = geo_settings
        self.log['geo_obj'] = geo

        norm_settings = {}
        norm_settings['p_ref'] = norms.p_ref
        norm_settings['Btor'] = norms.Btor
        norm_settings['a_minor'] = norms.a_meters
        norm_settings['vT_ref'] = norms.vT_ref
        norm_settings['rho_star'] = norms.rho_star
        norm_settings['t_ref'] = norms.t_ref
        norm_settings['P_ref_MWm3'] = norms.P_ref_MWm3
        norm_settings['Sn_ref_SI20'] = norms.Sn_ref_SI20
        self.log['norms'] = norm_settings

    def store_output_metadata(self):
        """
        This function creates dictionaries summarizing output variable types,
        dimensions, units, and description. This information is included in
        the ADIOS2 and NetCDF4 output. The keys in the dictionaries are the
        variables names used in T3D and the output files for easy association.
        The item for each key is another dictionary which contains the data
        type, the dimensions (if an array), the units, and a description of
        the variable. Note that the dimensions given by tuples of strings of
        the dimension variable are needed by the NetCDF4 output but not for
        ADIOS2.
        """

        # Function to simplify dictionary entry
        def _dict_entry(*, data_type=np.float64, description, dims=None, units=None) -> dict:
            """
            Create a metadata dictionary entry for a variable

            Parameters
            ----------
            description (str) : Description of the variable
            data_type : Data type of the variable (default: numpy.float64)
            dims : Dimension names for the variable (optional)
            units : Units of the variable (optional)

            Returns
            -------
            dict
                Dictionary containing metadata for the variable
            """

            entry = {
                'data_type': data_type,
                'description': description,
            }

            if dims:
                entry['dims'] = dims
            if units:
                entry['units'] = units

            return entry

        # Initialize the norms dict which contains reference parameters used for normalization
        self.norms_dict = {
            't_ref': _dict_entry(units='s', description='Reference time given by the minor radius divided by the reference thermal velocity / rhostar**2'),
            'rho_star': _dict_entry(units='-', description='The ratio of the gyroradius to the minor radius'),
            'P_ref_MWm3': _dict_entry(units='MW m^-3', description='Reference pressure source'),
            'Sn_ref_SI20': _dict_entry(units='10^20 m^-3 s^-1', description='Reference particle source')
        }

        # Initialize the time dict which contains time-stepping related output including convergence
        self.time_dict = {
            't': _dict_entry(dims=('n_steps', ), units='[t_ref]', description='Normalized time array'),
            't_iter_idx': _dict_entry(data_type=np.int32, dims=('n_steps', ), description='Iteration index'),
            't_step_idx': _dict_entry(data_type=np.int32, dims=('n_steps', ), description='Time-step index'),
            't_rms': _dict_entry(data_type=np.float64, dims=('n_steps', ), description='Relative RMS Newton update'),
            't_implicit_rms': _dict_entry(dims=('n_steps', ), description='Relative RMS time-discrete defect at the evaluated iterate, scaled by the leading time coefficient'),
            't_steady_rms': _dict_entry(dims=('n_steps', ), units='1/[t_ref]', description='Relative RMS semidiscrete transport rate at the evaluated iterate'),
            't_arrival_order': _dict_entry(data_type=np.int32, dims=('n_steps', ), description='Integrator order of the interval leading to this state: 0 if no derivative or a controller jump, 1 for first order, 2 for BDF2'),
            't_G_fac': _dict_entry(dims=('n_steps','n_radial'), units='-', description='Geometry factor on the recorded profile grid'),
            't_geometry_file': _dict_entry(data_type=str, dims=('n_steps', ), description='Equilibrium file for each recorded state'),
            't_diagnostics_valid': _dict_entry(data_type=np.int32, dims=('n_steps', ), description='1 if flux and source diagnostics were evaluated at the recorded profiles; 0 if unavailable')
        }

        # Initialize the grid dict which contains grid related output
        self.grid_dict = {
            'flux_label': _dict_entry(data_type=str, description='String to denote the flux label: rminor = r/a or torflux = sqrt(toroidal_flux/toroidal_flux_LCFS)'),
            'rho': _dict_entry(dims=('n_radial', ), units='-', description='Flux surface grid'),
            'midpoints': _dict_entry(dims=('n_radial_m_1', ), units='-', description='Midpoints of the flux surface grid, where flux models are evaluated'),
            'drho': _dict_entry(units='-', description='Grid spacing')
        }

        # Initialize the geo dict which contains geometry related output
        self.geo_dict = {
            'Btor': _dict_entry(units='T', description='Toroidal magnetic field'),
            'B0': _dict_entry(units='T', description='Magnetic field'),
            'a_minor': _dict_entry(units='m', description='Minor radius'),
            'R_major': _dict_entry(units='m', description='Major radius'),
            'area': _dict_entry(dims=('n_radial_m_1', ), units='-', description='Surface area of flux surfaces on the midpoints grid, normalized by the minor radius squared'),
            'grho': _dict_entry(dims=('n_radial_m_1', ), units='-', description='<|grad rho|> on the midpoints grid, normalized by the minor radius'),
            'area_grid': _dict_entry(dims=('n_radial', ), units='-', description='Surface area of flux surfaces on the rho grid, normalized by the minor radius squared'),
            'grho_grid': _dict_entry(dims=('n_radial', ), units='-', description='<|grad rho|> on the rho grid, normalized by the minor radius'),
            'G_fac': _dict_entry(dims=('n_radial', ), units='-', description='Geometry factor given by the ratio of grho_grid over area_grid')
        }

        # Initilialize the species dict which contains species related output including profiles, gradients, fluxes, and sources
        self.species_dict = {
            'species_types': _dict_entry(data_type=str, dims=('n_species'), description='List of specie types'),
            'species_tags': _dict_entry(data_type=str, dims=('n_species'),  description='List of specie tags'),
            'bulk_ion_tag': _dict_entry(data_type=str, description='Bulk ion tag'),
            'flux_model_labels': _dict_entry(data_type=str, dims=('n_flux_models'), description='List of flux model labels'),
            'particle_source_labels': _dict_entry(data_type=str, dims=('n_particle_sources'), description='List of particle source labels'),
            'power_source_labels': _dict_entry(data_type=str, dims=('n_power_sources'), description='List of power source labels'),
            'n_evolve_species': _dict_entry(data_type=str, dims=('num_n_evolve_species'), description='List of species for which the density is being evolved'),
            'T_evolve_species': _dict_entry(data_type=str, dims=('num_T_evolve_species'), description='List of species for which the temperature is being evolved')
        }

        # Profiles
        _dims = ('n_steps', 'n_radial')
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            self.species_dict['n_'+tag] = _dict_entry(dims=_dims, units='10^20 m^-3', description=f'{s} density profile on rho grid')
            self.species_dict['p_'+tag] = _dict_entry(dims=_dims, units='10^20 m^-3 keV', description=f'{s} pressure profile on rho grid')
            self.species_dict['T_'+tag] = _dict_entry(dims=_dims, units='keV', description=f'{s} temperature profile on rho grid')
        # Fluxes
        _dims = ('n_steps', 'n_radial_m_1')
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            self.species_dict['pflux_'+tag] = _dict_entry(dims=_dims, units='[GB]', description=f'GyroBohm-normalized total {s} particle flux profile on midpoints grid')
            self.species_dict['qflux_'+tag] = _dict_entry(dims=_dims, units='[GB]', description=f'GyroBohm-normalized total {s} heat flux profile on midpoints grid')
            self.species_dict['heat_'+tag] = _dict_entry(dims=_dims, units='[GB]', description=f'GyroBohm-normalized total {s} collisional heating profile on midpoints grid')
            for label in self.log['flux_model_labels']:
                tag2 = label + '_' + tag
                self.species_dict['pflux_'+tag2] = _dict_entry(dims=_dims, units='[GB]', description=f'GyroBohm-normalized {s} particle flux profile from {label} on midpoints grid')
                self.species_dict['qflux_'+tag2] = _dict_entry(dims=_dims, units='[GB]', description=f'GyroBohm-normalized {s} heat flux profile from {label} on midpoints grid')
                self.species_dict['heat_'+tag2] = _dict_entry(dims=_dims, units='[GB]', description=f'GyroBohm-normalized {s} collisional heating profile from {label} on midpoints grid')
        # Gradients
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            self.species_dict['aLn_'+tag] = _dict_entry(dims=_dims, units='-', description=f'Normalized {s} density gradient, a/L_n = -1/n dn/drho, on midpoints grid')
            self.species_dict['aLp_'+tag] = _dict_entry(dims=_dims, units='-', description=f'Normalized {s} pressure gradient, a/L_p = -1/p dp/drho, on midpoints grid')
            self.species_dict['aLT_'+tag] = _dict_entry(dims=_dims, units='-', description=f'Normalized {s} temperature gradient, a/L_T = -1/T dT/drho, on midpoints grid')
        # Particle source terms
        _particle_source = defaultdict(lambda: 'external', {'aux': 'auxiliary'})
        _dims = ('n_steps', 'n_radial')
        _units = '[Sn_ref_SI20]'
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            for label in self.log['particle_source_labels']:
                tag2 = label + '_' + tag
                self.species_dict['Sn_'+tag2] = _dict_entry(dims=_dims, units=_units, description=f'Normalized {s} {_particle_source[label]} particle source on rho grid')
            self.species_dict['Sn_tot_'+tag] = _dict_entry(dims=_dims, units=_units, description=f'Normalized total {s} particle source on rho grid')
        for label in self.log['particle_source_labels']:
            self.species_dict['Sn_' + label + '_i'] = _dict_entry(dims=_dims, units=_units, description=f'Normalized ion-summed {_particle_source[label]} particle source on rho grid')
        self.species_dict['Sn_tot_i'] = _dict_entry(dims=_dims, units=_units, description='Normalized ion-summed total particle source on rho grid')
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            for label in self.log['particle_source_labels']:
                tag2 = label + '_cumint_SI20_' + tag
                self.species_dict['Sn_'+tag2] = _dict_entry(dims=_dims, units='10^20 s^-1', description=f'Cumulative {s} {_particle_source[label]} particle source over evolved shells')
            self.species_dict['Sn_tot_cumint_SI20_'+tag] = _dict_entry(dims=_dims, units='10^20 s^-1', description=f'Cumulative {s} total particle source over evolved shells')
        _dims = ('n_steps', )
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            for label in self.log['particle_source_labels']:
                tag2 = label + '_int_SI20_' + tag
                self.species_dict['Sn_'+tag2] = _dict_entry(dims=_dims, units='10^20 s^-1', description=f'Integrated {s} {_particle_source[label]} particle source over evolved shells')
            self.species_dict['Sn_tot_int_SI20_'+tag] = _dict_entry(dims=_dims, units='10^20 s^-1', description=f'Integrated {s} total particle source over evolved shells')
        # Power source terms
        _power_source_descriptions = defaultdict(lambda: 'external power source', {
            'aux': 'auxiliary power source',
            'alpha': 'deposited alpha heating',
            'rad': 'power source (sink) from radiation',
            'heating': 'power source from turbulent heating',
            'coll': 'power source (sink) from collisional equilibration',
        })
        _dims = ('n_steps', 'n_radial')
        _units = '[P_ref_MWm3]'
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            for label in self.log['power_source_labels']:
                tag2 = label + '_' + tag
                self.species_dict['Sp_'+tag2] = _dict_entry(dims=_dims, units=_units, description=f'Normalized {s} {_power_source_descriptions[label]} on rho grid')
            self.species_dict['Sp_tot_'+tag] = _dict_entry(dims=_dims, units=_units, description=f'Normalized total {s} power source on rho grid')
        for label in self.log['power_source_labels']:
            self.species_dict['Sp_'+label+'_i'] = _dict_entry(dims=_dims, units=_units, description=f'Normalized ion-summed {_power_source_descriptions[label]} on rho grid')
        self.species_dict['Sp_tot_i'] = _dict_entry(dims=_dims, units=_units, description='Normalized ion-summed total power source on rho grid')
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            for label in self.log['power_source_labels']:
                tag2 = label + '_cumint_MW_' + tag
                self.species_dict['Sp_'+tag2] = _dict_entry(dims=_dims, units='MW', description=f'Integrated {s} {_power_source_descriptions[label]} interior to each flux surface on rho grid')
            self.species_dict['Sp_tot_cumint_MW_'+tag] = _dict_entry(dims=_dims, units='MW', description=f'Integrated {s} total power source interior to each flux surface on rho grid')
        _dims = ('n_steps', )
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            for label in self.log['power_source_labels']:
                tag2 = label + '_int_MW_' + tag
                self.species_dict['Sp_'+tag2] = _dict_entry(dims=_dims, units='MW', description=f'Integrated {s} {_power_source_descriptions[label]}')
            self.species_dict['Sp_tot_int_MW_'+tag] = _dict_entry(dims=_dims, units='MW', description=f'Integrated {s} total power source')

        # Particle fluxes in physical units
        _dims = ('n_steps', 'n_radial_m_1')
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            self.species_dict['Gam_SI20_'+tag] = _dict_entry(dims=_dims, units='10^20 / s', description=f'Dimensional {s} particle flux on midpoint grid')
            for label in self.log['flux_model_labels']:
                self.species_dict['Gam_'+label+'_SI20_'+tag] = _dict_entry(dims=_dims, units='10^20 / s', description=f'Dimensional {s} particle flux from {label} on midpoint grid')

        # Heat fluxes in physical units
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            self.species_dict['Q_MW_'+tag] = _dict_entry(dims=_dims, units='MW', description=f'Dimensional {s} heat flux on midpoint grid')
            for label in self.log['flux_model_labels']:
                self.species_dict['Q_'+label+'_MW_'+tag] = _dict_entry(dims=_dims, units='MW', description=f'Dimensional {s} heat flux from {label} on midpoint grid')

        # Alpha heating
        _dims = ('n_steps', 'n_radial')
        self.species_dict['Palpha_MWm3'] = _dict_entry(dims=_dims, units='MW m^-3', description='Modeled deposited alpha heating power density')
        self.species_dict['Palpha_birth_MWm3'] = _dict_entry(dims=_dims, units='MW m^-3', description='Unattenuated D-T alpha birth power density')
        self.species_dict['Palpha_cumint_MW'] = _dict_entry(dims=_dims, units='MW', description='Cumulative modeled alpha deposition over evolved shells')
        self.species_dict['Palpha_int_MW'] = _dict_entry(dims=('n_steps', ), units='MW', description='Modeled alpha deposition integrated over evolved shells')
        self.species_dict['p_alpha'] = _dict_entry(dims=_dims, units='10^20 m^-3 keV', description='Alpha particle pressure profile on rho grid')
        self.species_dict['beta_alpha'] = _dict_entry(dims=_dims, units='-', description='Alpha particle beta profile on rho grid')
        if self.physics.Zeff_prof:
            self.species_dict['Zeff'] = _dict_entry(dims=_dims, units='-', description='Effective charge, Z_effective on rho grid')

        # Initiliaze the zerod dict which contains integrated related output
        self.zerod_dict = {}
        _dims = ('n_steps', )
        for s, tag in zip(self.log['species_types'], self.log['species_tags']):
            self.zerod_dict['n_volavg_'+tag] = _dict_entry(dims=_dims, units='10^20 m^-3', description=f'Volume-averaged {s} density')
            self.zerod_dict['n_lineavg_'+tag] = _dict_entry(dims=_dims, units='10^20 m^-3', description=f'Line-averaged {s} density')
            self.zerod_dict['npeak_'+tag] = _dict_entry(dims=_dims, units='-', description=f'{s} density peaking factor, f = n_0/<n>_vol')
            self.zerod_dict['T_volavg_'+tag] = _dict_entry(dims=_dims, units='keV', description=f'Volume-averaged {s} temperature')
            self.zerod_dict['T_lineavg_'+tag] = _dict_entry(dims=_dims, units='keV', description=f'Line-averaged {s} temperature')
            self.zerod_dict['Tpeak_'+tag] = _dict_entry(dims=_dims, units='-', description=f'{s} temperature peaking factor, f = T_0/<T>_vol')
            self.zerod_dict['N_SI20_'+tag] = _dict_entry(dims=_dims, units='particles', description=f'Number of {s} particles in evolved shells')
            self.zerod_dict['W_MJ_'+tag] = _dict_entry(dims=_dims, units='MJ', description=f'Volume integral of {s} pressure')
            self.zerod_dict['Pheat_MW_'+tag] = _dict_entry(dims=_dims, units='MW', description=f'Net power from all {s} sources over evolved shells')
            self.zerod_dict['Ploss_MW_'+tag] = _dict_entry(dims=_dims, units='MW', description=f'Outward {s} heat flow through the last transport face')
        self.zerod_dict['Ntot_SI20'] = _dict_entry(dims=_dims, units='particles', description='Total number of particles in evolved shells')
        self.zerod_dict['Wtot_MJ'] = _dict_entry(dims=_dims, units='MJ', description='Volume integral of species-summed pressure')
        self.zerod_dict['beta_vol'] = _dict_entry(dims=_dims, units='-', description='Volume averaged total beta')
        self.zerod_dict['Pheat_MW'] = _dict_entry(dims=_dims, units='MW', description='Species-summed net source power over evolved shells')
        self.zerod_dict['Ploss_MW'] = _dict_entry(dims=_dims, units='MW', description='Species-summed outward heat flow through the last transport face')
        self.zerod_dict['Paux_MW'] = _dict_entry(dims=_dims, units='MW', description='Prescribed and external auxiliary heating over evolved shells')
        self.zerod_dict['Palpha_int_MW'] = _dict_entry(dims=_dims, units='MW', description='Modeled alpha deposition integrated over evolved shells')
        self.zerod_dict['Palpha_birth_int_MW'] = _dict_entry(dims=_dims, units='MW', description='Unattenuated alpha birth power integrated over evolved shells')
        self.zerod_dict['Pfus_MW'] = _dict_entry(dims=_dims, units='MW', description='D-T birth power (neutrons and alphas) integrated over evolved shells')
        self.zerod_dict['Qfus'] = _dict_entry(dims=_dims, units='-', description='Fusion gain')
        if self.physics.alpha_heating or self.auxiliary_heating:
            self.zerod_dict['tauE'] = _dict_entry(dims=_dims, units='s', description='Steady-state estimate Wtot divided by net auxiliary, deposited alpha and radiative power')
            if self.geo.geo_option == "vmec" or self.geo.geo_option == "desc" or self.geo.geo_option == "basic":
                self.zerod_dict['tauE_ISS04'] = _dict_entry(dims=_dims, units='s', description='Prediction of energy confinement time according to tauE_ISS04')
                self.zerod_dict['f_renorm'] = _dict_entry(dims=_dims, units='-', description='f = tauE / tauE_ISS04, assuming steady-state')
                self.zerod_dict['n_sudo'] = _dict_entry(dims=_dims, units='10^20 m^-3', description='Sudo density limit')
                self.zerod_dict['f_sudo_line'] = _dict_entry(dims=_dims, units='-', description='f = <n_e>_line/n_sudo')
                self.zerod_dict['f_sudo_vol'] = _dict_entry(dims=_dims, units='-', description='f = <n_e>_vol/n_sudo')
                self.zerod_dict['f_sudo_edge'] = _dict_entry(dims=_dims, units='-', description='f = n_edge/n_sudo')

    def save(self, species, solver, time, final=False):
        """Function to save the data from the next time-step to the data log dictionary"""

        # save time markers
        self.log['t'].append(time.time)
        self.log['t_iter_idx'].append(time.iter_idx)
        self.log['t_step_idx'].append(time.step_idx)
        self.log['t_rms'].append(time.rms)
        # A final record may refresh transport at a new state; a Newton
        # defect cannot be carried from the prior iterate into that row.
        self.log['t_implicit_rms'].append(np.nan if final else getattr(time, 'implicit_rms', np.nan))
        self.log['t_steady_rms'].append(np.nan if final else getattr(time, 'steady_rms', np.nan))
        self.log['t_arrival_order'].append(getattr(time, 'arrival_order', 0) if time.iter_idx == 0 else 0)
        self.log['t_geometry_file'].append(str(getattr(self.geo, 'geo_file', '')))
        self.log['t_G_fac'].append(self.geo.G_fac.profile.copy())
        valid = time.diagnostics_current
        self.log['t_diagnostics_valid'].append(np.int32(valid))

        def diagnostic(value):
            if valid:
                return value
            if hasattr(value, "profile"):
                return type(value)(np.full_like(value.profile, np.nan, dtype=float), value.grid)
            return np.full_like(value, np.nan, dtype=float) if np.ndim(value) else np.nan

        self.log['species'].append(copy.deepcopy(species))

        N_tot = 0
        W_tot = 0
        Pheat_tot = 0
        Ploss_tot = 0
        Paux_tot = 0
        beta_vol = 0
        zero = GridProfile(0., self.grid)
        ion_particles = {label: zero for label in [*self.log['particle_source_labels'], 'tot']}
        ion_powers = {label: zero for label in [*self.log['power_source_labels'], 'tot']}

        gradients = species.get_grads_on_flux_grid()[:3]
        for index, s in enumerate(species.get_species_list()):
            tag = s.type_tag
            self.log['n_'+tag].append(s.n().profile.copy())
            self.log['p_'+tag].append(s.p().profile.copy())
            self.log['T_'+tag].append(s.T().profile.copy())
            raw = (s.pflux, s.qflux, s.heat)
            flux_names = ('pflux', 'qflux', 'heat')
            values = s.flux_diagnostics_gb(species.ref_species, self.geo) if valid else raw
            for name, value in zip(flux_names, values):
                self.log[name+'_'+tag].append(diagnostic(value))
            for label in self.log['flux_model_labels']:
                values = s.flux_diagnostics_gb(species.ref_species, self.geo, label=label) if valid else raw
                for name, value in zip(flux_names, values):
                    self.log[name+'_'+label+'_'+tag].append(diagnostic(value))

            for name, values in zip(('aLn', 'aLT', 'aLp'), gradients):
                self.log[name+'_'+tag].append(values[index].copy())

            # Source totals use the same evolved shells as the transport equations.
            particle_source_labels = self.log['particle_source_labels']
            for label in particle_source_labels:
                labeled_source = diagnostic(s.Sn_labeled.get(label, zero))
                self.log['Sn_'+label+'_'+tag].append(labeled_source)
                self.log['Sn_'+label+'_int_SI20_'+tag].append(self.geo.volume_integrate(labeled_source, active=True)*self.norms.Sn_ref_SI20)
                self.log['Sn_'+label+'_cumint_SI20_'+tag].append(self.geo.volume_integrate(labeled_source, profile=True, active=True)*self.norms.Sn_ref_SI20)
                if s.type != 'electron':
                    ion_particles[label] += labeled_source
            total_source = diagnostic(s.Sn)
            self.log['Sn_tot_'+tag].append(total_source)
            if s.type != 'electron':
                ion_particles['tot'] += total_source
            self.log['Sn_tot_int_SI20_'+tag].append(self.geo.volume_integrate(total_source, active=True)*self.norms.Sn_ref_SI20)
            self.log['Sn_tot_cumint_SI20_'+tag].append(self.geo.volume_integrate(total_source, profile=True, active=True)*self.norms.Sn_ref_SI20)
            power_source_labels = self.log['power_source_labels']
            for label in power_source_labels:
                labeled_source = diagnostic(s.Sp_labeled.get(label, zero))
                self.log['Sp_'+label+'_'+tag].append(labeled_source)
                self.log['Sp_'+label+'_int_MW_'+tag].append(self.geo.volume_integrate(labeled_source, active=True)*self.norms.P_ref_MWm3)
                self.log['Sp_'+label+'_cumint_MW_'+tag].append(self.geo.volume_integrate(labeled_source, profile=True, active=True)*self.norms.P_ref_MWm3)
                if s.type != 'electron':
                    ion_powers[label] += labeled_source
            total_source = diagnostic(s.Sp)
            self.log['Sp_tot_'+tag].append(total_source)
            if s.type != 'electron':
                ion_powers['tot'] += total_source
            self.log['Sp_tot_int_MW_'+tag].append(self.geo.volume_integrate(total_source, active=True)*self.norms.P_ref_MWm3)
            self.log['Sp_tot_cumint_MW_'+tag].append(self.geo.volume_integrate(total_source, profile=True, active=True)*self.norms.P_ref_MWm3)

            # Do not access unlabeled backend data on runs with no flux evaluation.
            unavailable = np.full(len(self.grid.midpoints), np.nan)
            VprimeQ_s = (1.5*s.Fp_plus(species.ref_species, self.geo)*self.norms.P_ref_MWm3*self.norms.a_meters**3
                         if valid else unavailable)  # MW
            self.log['Q_MW_'+tag].append(VprimeQ_s)
            self.log['Gam_SI20_'+tag].append(
                s.Fn_plus(species.ref_species, self.geo)*self.norms.Sn_ref_SI20*self.norms.a_meters**3
                if valid else unavailable)
            for label in self.log['flux_model_labels']:
                self.log['Q_'+label+'_MW_'+tag].append(
                    1.5*s.Fp_plus(species.ref_species, self.geo, label=label)*self.norms.P_ref_MWm3*self.norms.a_meters**3
                    if valid else unavailable)
                self.log['Gam_'+label+'_SI20_'+tag].append(
                    s.Fn_plus(species.ref_species, self.geo, label=label)*self.norms.Sn_ref_SI20*self.norms.a_meters**3
                    if valid else unavailable)

            # various integrated quantities

            N_s = self.geo.volume_integrate(s.n(), active=True) * 1e20
            self.log["N_SI20_"+tag].append(N_s)
            N_tot += N_s

            # stored energy
            W_s = self.geo.volume_integrate(1.5*s.p(), active=True)*self.norms.p_ref/1e6  # MJ
            self.log['W_MJ_'+tag].append(W_s)
            W_tot += W_s

            Paux_s = sum(self.log[f'Sp_{label}_int_MW_{tag}'][-1]
                         for label in ['aux', *self.external_power_labels])
            Pheat_s = self.log['Sp_tot_int_MW_'+tag][-1]
            self.log['Pheat_MW_'+tag].append(Pheat_s)
            Ploss_s = VprimeQ_s[-1]
            self.log['Ploss_MW_'+tag].append(Ploss_s)
            Pheat_tot += Pheat_s
            Ploss_tot += Ploss_s
            Paux_tot += Paux_s

            beta_vol += self.geo.volume_average(s.beta(self.geo.Btor))

            n_volavg_s = self.geo.volume_average(s.n())  # 10^20/m^3
            self.log['n_volavg_'+tag].append(n_volavg_s)
            n_lineavg_s = np.average(s.n().profile)  # 10^20/m^3
            if s.type == "electron":
                ne_volavg = n_volavg_s
                ne_avg = n_lineavg_s
            self.log['n_lineavg_'+tag].append(n_lineavg_s)
            npeak_s = s.n()[0]/n_volavg_s
            self.log['npeak_'+tag].append(npeak_s)

            T_volavg_s = self.geo.volume_average(s.T())  # keV
            self.log['T_volavg_'+tag].append(T_volavg_s)
            T_lineavg_s = np.average(s.T().profile)  # keV
            self.log['T_lineavg_'+tag].append(T_lineavg_s)
            Tpeak_s = s.T()[0]/T_volavg_s
            self.log['Tpeak_'+tag].append(Tpeak_s)

        for label, data in ion_particles.items():
            self.log[f'Sn_{label}_i'].append(data)
        for label, data in ion_powers.items():
            self.log[f'Sp_{label}_i'].append(data)

        Palpha = self.physics.compute_total_alpha_heating(species)
        Palpha_birth = self.physics.compute_alpha_birth_power(species)
        self.log['Palpha_MWm3'].append(Palpha)
        self.log['Palpha_birth_MWm3'].append(Palpha_birth)
        self.log['Palpha_cumint_MW'].append(self.geo.volume_integrate(Palpha, profile=True, active=True))
        Palpha_tot = self.geo.volume_integrate(Palpha, active=True)
        Palpha_birth_tot = self.geo.volume_integrate(Palpha_birth, active=True)
        Pfus = 5.03*Palpha_birth_tot
        self.log['Palpha_int_MW'].append(Palpha_tot)
        self.log['Palpha_birth_int_MW'].append(Palpha_birth_tot)
        self.log['Pfus_MW'].append(Pfus)

        # fusion metrics
        self.log['Ntot_SI20'].append(N_tot)
        self.log['Wtot_MJ'].append(W_tot)
        self.log['Pheat_MW'].append(Pheat_tot)
        self.log['Ploss_MW'].append(Ploss_tot)
        self.log['Paux_MW'].append(Paux_tot)
        self.log['beta_vol'].append(beta_vol)
        Qfus = Pfus / Paux_tot if np.isfinite(Paux_tot) and Paux_tot > 0 else np.nan
        self.log['Qfus'].append(Qfus)

        # Compute energy confinement time and related metrics only
        # when alpha or external heating sources are present
        if self.physics.alpha_heating or self.auxiliary_heating:
            tauE = self.physics.compute_tauE(species) if valid else np.nan
            self.log['tauE'].append(tauE)
            if final and valid:
                info("\n Final confinement time and related metrics:", color='yellow')
                info(f"   Wtot = {W_tot:.3f} MJ", color='yellow')
                info(f"   Pheat = {Pheat_tot:.3f} MW", color='yellow')
                info(f"   Ploss = {Ploss_tot:.3f} MW", color='yellow')
                info(f"   Palpha = {self.log['Palpha_int_MW'][-1]:.3f} MW", color='yellow')
                info(f"   Q_fus = {Qfus:.3f}", color='yellow')
                info(f"   tauE = {tauE:.3e} s", color='yellow')
                info(f"   beta_vol = {beta_vol*100:.2f} %", color='yellow')

            if self.geo.geo_option == "vmec" or self.geo.geo_option == "desc" or self.geo.geo_option == 'basic':
                R = self.geo.AspectRatio*self.geo.a_minor
                a = self.geo.a_minor
                P = self.physics.compute_Pheat(species) if valid else np.nan
                B = np.abs(self.geo.B0)

                tauE_ISS04 = self.physics.compute_ISS04(species) if valid else np.nan
                self.log['tauE_ISS04'].append(tauE_ISS04)
                self.log['f_renorm'].append(tauE/tauE_ISS04)

                n_sudo = np.sqrt(P*B/(16*R*a**2)) if P > 0 and B > 0 else np.nan  # 10^20/m^3
                self.log['n_sudo'].append(n_sudo)
                f_sudo_lineavg = ne_avg/n_sudo
                self.log['f_sudo_line'].append(f_sudo_lineavg)
                f_sudo_vol = ne_volavg/n_sudo
                self.log['f_sudo_vol'].append(f_sudo_vol)
                f_sudo_edge = species['electron'].n()[-1]/n_sudo
                self.log['f_sudo_edge'].append(f_sudo_edge)
                if final and valid:
                    info(f"tauE_ISS04 = {tauE_ISS04:.3e} s, f_renorm = {tauE/tauE_ISS04:.3f}, n_sudo = {n_sudo:.3f} 10^20/m^3, f_sudo_line = {f_sudo_lineavg:.3f}, f_sudo_vol = {f_sudo_vol:.3f}, f_sudo_edge = {f_sudo_edge:.3f}")

        # compute alpha pressure
        p_alpha = self.physics.compute_alpha_pressure(species)
        beta_alpha = 4.03e-11*p_alpha*1e17/(self.geo.Btor*1e4)**2
        self.log['p_alpha'].append(p_alpha)
        self.log['beta_alpha'].append(beta_alpha)

        if self.physics.Zeff_prof:
            self.log['Zeff'].append(self.physics.Zeff_prof)

        try:
            self.log['solver_jacobian'].append(solver.jacob)
            self.log['solver_rhs'].append(solver.rhs)
        except:
            pass

    def export(self, f_save=None):
        """Exports the numpy pickle file which contains data only and not units or description information"""
        if f_save is None:
            f_save = self.f_save
        f_save = str(f_save)
        if not f_save.endswith(".npy"):
            f_save += ".npy"
        with _atomic_output(f_save) as tmp, tmp.open("wb") as file:
            np.save(file, self.log)

    def write_netCDF4(self, infile, f_save=None):
        """Function to create the NetCDF output"""

        # Function for writing variables to the NetCDF file
        def _netcdf4_write(group, group_dict, log_group=None):
            """
            Write data to a NetCDF4 group.

            Parameters:
            group (netCDF4.Group): The NetCDF4 group to write the data to.
            group_dict (dict): A dictionary containing the dataset information, including the data type, dimensions, units, and description.
            log_group (str, optional): The name of the group in the log dictionary to use for the data. If not provided, the function will use the dataset name as the key in the log dictionary.

            Returns:
            None
            """
            for dataset_name, dataset_info in group_dict.items():
                if log_group:
                    data = self.log[log_group][dataset_name]
                else:
                    data = self.log[dataset_name]

                # fixes netcdf4 comatibility with both numpy2 and numpy1
                dt = dataset_info['data_type']
                if dt in (np.integer, np.signedinteger, np.unsignedinteger):
                    dt = np.int32

                if 'dims' in dataset_info:
                    variable = group.createVariable(dataset_name, dt, dataset_info['dims'])
                    if 'units' in dataset_info:
                        variable.units = dataset_info['units']
                    if 'description' in dataset_info:
                        variable.description = dataset_info['description']
                    if dt != str:
                        group[dataset_name][:] = data
                    else:
                        for i in range(len(data)):
                            group[dataset_name][i] = data[i]
                else:
                    variable = group.createVariable(dataset_name, dt)
                    if 'units' in dataset_info:
                        variable.units = dataset_info['units']
                    if 'description' in dataset_info:
                        variable.description = dataset_info['description']
                    group[dataset_name][...] = data

        # Exit immediately if NetCDF4 output is not needed
        if not self.output_netcdf:
            return

        # Set the output filename
        if f_save is None:
            f_save = self.nc_save

        with _atomic_output(f_save) as tmp, Dataset(tmp, mode='w', format='NETCDF4') as ncf:
            # Attributes
            ncf.title = "T3D netCDF4 output file"
            ncf.history = "Created " + date.today().strftime("%Y/%m/%d") + " by " + getpass.getuser()

            # Copy the input file to the output NetCDF4 file
            ncf.infile = infile
            with open(infile) as f:
                inputs = f.read()
                ncf.inputs = inputs  # This dumps the entire T3D input file

            # Global dimensions
            _ = ncf.createDimension('n_steps', len(self.log['t']))
            _ = ncf.createDimension('n_radial', self.log['grid']['N_radial'])
            _ = ncf.createDimension('n_radial_m_1', self.log['grid']['N_radial'] - 1)

            # Add scalars and norms
            norms = ncf.createGroup('norms')
            _netcdf4_write(norms, self.norms_dict, log_group='norms')

            # Create time group
            times = ncf.createGroup('time')
            _netcdf4_write(times, self.time_dict, log_group=None)

            # Create grid group
            grid = ncf.createGroup('grid')
            _netcdf4_write(grid, self.grid_dict, log_group='grid')

            # Create geo group
            geo = ncf.createGroup('geo')
            _netcdf4_write(geo, self.geo_dict, log_group='geo')

            # Create species and profiles group
            species = ncf.createGroup('species')
            _ = species.createDimension('n_species', len(self.log['species_types']))
            _ = species.createDimension('n_flux_models', len(self.log['flux_model_labels']))
            _ = species.createDimension('n_particle_sources', len(self.log['particle_source_labels']))
            _ = species.createDimension('n_power_sources', len(self.log['power_source_labels']))
            _ = species.createDimension('num_n_evolve_species', len(self.log['n_evolve_species']))
            _ = species.createDimension('num_T_evolve_species', len(self.log['T_evolve_species']))
            _netcdf4_write(species, self.species_dict, log_group=None)

            # Create zerod group
            zerod = ncf.createGroup('zeroD')
            _netcdf4_write(zerod, self.zerod_dict, log_group=None)


    def write_adios2(self, infile, f_save=None):
        """This function writes T3D output to ADIOS2 bp format

        Arguments:
        infile -- t3d input file name which gets dumped to the adios2 output
        f_save -- name of the adios2 output
        """

        def _adios2_write(group, group_dict, log_group=None):
            """Function to write adios2 variables and associated attributes (units and description)"""
            for key in group_dict.keys():
                label = group + '/' + key
                if log_group:
                    var = self.log[log_group][key]
                else:
                    var = self.log[key]
                if var is None:
                    continue
                if 'dims' in group_dict[key].keys() and group_dict[key]['data_type'] != str:
                    values = np.array(var)
                    fh.write(label, values, values.shape, [0]*values.ndim, values.shape)
                else:
                    fh.write_attribute(label, var)
                for k in ['units', 'description']:
                    if k in group_dict[key].keys():
                        fh.write_attribute(label + '/' + k, group_dict[key][k])

        # Exit immediately if ADIOS2 output is not needed
        if not self.output_adios2:
            return

        # Set the output filename
        if f_save is None:
            f_save = self.bp_save

        # ADIOS output stream
        fh = adios2.Stream(f_save, "w")

        # Output the metadata
        fh.write_attribute('title', 'T3D ADIOS2 output')
        fh.write_attribute('history', f'Created {date.today().strftime("%Y/%m/%d")} by {getpass.getuser()}')

        # Output the input file location and contents
        fh.write_attribute('infile', infile)
        with open(infile) as f:
            inputs = f.read()
            fh.write_attribute('inputs', inputs)  # This dumps the entire T3D input file

        _adios2_write('norms', self.norms_dict, log_group='norms')
        _adios2_write('time', self.time_dict, log_group=None)
        _adios2_write('grid', self.grid_dict, log_group='grid')
        _adios2_write('geo', self.geo_dict, log_group='geo')
        _adios2_write('species', self.species_dict, log_group=None)
        _adios2_write('zeroD', self.zerod_dict, log_group=None)

        fh.close()
