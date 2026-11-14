import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from collections import defaultdict
from netCDF4 import Dataset
import adios2

from t3d.Profiles import FluxProfile, GridProfile
from t3d.constants import ELEMENTARY_CHARGE
from t3d.Logbook import warn, errr


class Import():

    def __init__(self, inputs, grid):
        self.grid = grid

        # read import parameters from input file
        import_parameters = inputs.get('import', {})
        requested = import_parameters.get('type')
        if requested is not None and not isinstance(requested, str):
            raise ValueError('import.type must be a string.')
        aliases = {'t3d': 'trinity', 'text': 'ascii', 'txt': 'ascii', 'columns': 'ascii'}
        self.type = requested.strip().lower() if requested is not None else None
        self.type = aliases.get(self.type, self.type)
        self.file = import_parameters.get('file', '')
        if self.type is None:
            self.imported = None
        elif self.type == "transp":
            transp_time = import_parameters.get('transp_time', 0.0)
            self.imported = TranspReader(self.file, time=transp_time, flux_label=grid.flux_label)
        elif self.type == "trinity":
            trinity_frame = import_parameters.get('frame', -1)
            self.imported = TrinityReader(grid, self.file, trinity_frame)
        elif self.type == "jetto":
            jetto_time = import_parameters.get('jetto_time', import_parameters.get('jett0_time'))
            self.imported = JettoReader(self.file, time=jetto_time, flux_label=grid.flux_label)
        elif self.type == "ascii":
            columns = import_parameters.get('columns', [])  # list of keys corresponding to columns in file
            divide_by = import_parameters.get('divide_by', [])  # list of factors to divide columns by to get expected units
            if len(divide_by) == 0:
                divide_by = [1 for c in columns]
            if len(divide_by) != len(columns):
                raise ValueError("import.columns and import.divide_by lists must be the same length.")
            self.imported = AsciiColumnReader(grid, self.file, columns=columns, divide_by=divide_by)
        else:
            raise ValueError(f'Invalid import type {requested!r}; expected trinity, ascii, transp or jetto.')

    def get(self, key):
        return self.imported.data[key]

    def keys(self):
        return [k for k in self.imported.data.keys()]

    @staticmethod
    def _check_profile(rho, value):
        rho = np.ma.asarray(rho, dtype=float).filled(np.nan)
        value = np.ma.asarray(value, dtype=float).filled(np.nan)
        if (rho.ndim != 1 or rho.size < 2 or value.shape != rho.shape
                or not np.all(np.isfinite(rho)) or not np.all(np.diff(rho) > 0)):
            raise ValueError("Profile input requires a strictly increasing finite 1D radial grid and matching values.")
        if not np.all(np.isfinite(value)):
            raise ValueError("Profile input contains nonfinite values.")
        return rho, value

    def get_density(self, key=None, species_mass=2.0, species_charge=None, tag=None, interpolate=True, n_edge=None, norms=None):
        rho, n = self.imported.get_density(key=key, species_mass=species_mass, species_charge=species_charge, tag=tag, norms=norms)

        rho, n = self._check_profile(rho, n)
        # interpolate onto Trinity grid
        if interpolate:
            f = interp1d(rho, n, bounds_error=False, fill_value='extrapolate')
            n = f(self.grid.rho)

        if n_edge is not None:
            n = np.array(n, copy=True)
            n[-1] = n_edge

        return n

    def get_temperature(self, key=None, species_mass=2.0, species_charge=None, tag=None, interpolate=True, T_edge=None, norms=None):
        rho, T = self.imported.get_temperature(key=key, species_mass=species_mass, species_charge=species_charge, tag=tag, norms=norms)

        rho, T = self._check_profile(rho, T)
        # interpolate onto Trinity grid
        if interpolate:
            f = interp1d(rho, T, bounds_error=False, fill_value='extrapolate')
            T = f(self.grid.rho)

        if T_edge is not None:
            T = np.array(T, copy=True)
            T[-1] = T_edge

        return T

    def get_density_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, interpolate=True, norms=None):
        rho, Sn = self.imported.get_density_source(keys=keys, species_mass=species_mass, species_charge=species_charge, tag=tag, norms=norms)

        return self._source_on_grid(rho, Sn, interpolate)

    def get_pressure_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, interpolate=True, norms=None):
        rho, Sp = self.imported.get_pressure_source(keys=keys, species_mass=species_mass, species_charge=species_charge, tag=tag, norms=norms)

        return self._source_on_grid(rho, Sp, interpolate)

    def get_particle_flux(self, keys=None, tag=None):
        """Physical particle throughput on faces, in 10^20 s^-1."""
        return self._flux_on_faces('particle', keys, tag)

    def get_heat_flux(self, keys=None, tag=None):
        """Physical energy throughput on faces, in MW."""
        return self._flux_on_faces('heat', keys, tag)

    def _flux_on_faces(self, kind, keys, tag):
        if self.type not in ('ascii', 'trinity'):
            raise ValueError("Flux import requires [import] type='columns' or 'trinity'.")
        rho, values = self.imported.get_face_flux(kind, keys, tag)
        rho, values = self._check_profile(rho, values)
        if rho[0] < 0. or rho[-1] > 1.:
            raise ValueError("Flux import rho must be in [0, 1].")
        if rho[0] == 0. and values[0] != 0.:
            raise ValueError(f"Imported {kind} flux for {tag} must be zero at the magnetic axis.")
        faces = self.grid.midpoints
        tol = 16*np.finfo(float).eps
        if faces[0] < rho[0]-tol or faces[-1] > rho[-1]+tol:
            raise ValueError(
                f"Imported {kind} flux for {tag} must cover all transport faces "
                f"([{faces[0]:.8g}, {faces[-1]:.8g}]); extrapolation is not allowed.")
        return np.interp(faces, rho, values)

    def _source_on_grid(self, rho, profile, interpolate):
        rho, profile = np.asarray(rho), np.asarray(profile)
        if (rho.ndim != 1 or rho.size < 2 or profile.shape != rho.shape
                or not np.all(np.isfinite(profile)) or not np.all(np.isfinite(rho))
                or not np.all(np.diff(rho) > 0)):
            raise ValueError('Source import requires finite 1D profiles on a strictly increasing radial grid.')
        if not interpolate:
            return profile.copy()
        if self.type == 'trinity':
            if np.array_equal(rho, self.grid.rho):
                return profile.copy()
            # Restart samples are already cell values, not pointwise data.
            # Map by radial shell overlap without extrapolating beyond
            # the saved evolved interval. This does not remap geometry/V'.
            old_faces = np.r_[0., (rho[:-1] + rho[1:])/2]
            new_faces = np.r_[0., self.grid.midpoints]
            widths = np.diff(new_faces)
            overlap = np.maximum(0., np.minimum(new_faces[1:, None], old_faces[None, 1:])
                                 - np.maximum(new_faces[:-1, None], old_faces[None, :-1]))
            return np.r_[overlap @ profile[:-1] / widths, 0.]
        # Analytic integrals of the same piecewise-linear interpolant
        # (and linear endpoint extrapolation), avoiding adaptive quadrature
        # across potentially hundreds of derivative discontinuities.
        dr = np.diff(rho)
        slopes = np.diff(profile)/dr
        cumsum = np.r_[0., np.cumsum(.5*(profile[1:] + profile[:-1])*dr)]
        edges = np.r_[self.grid.rho[0]-self.grid.drho/2, self.grid.rho+self.grid.drho/2]
        j = np.clip(np.searchsorted(rho, edges, side='right')-1, 0, rho.size-2)
        offset = edges-rho[j]
        primitive = cumsum[j] + offset*(profile[j]+.5*offset*slopes[j])
        return np.diff(primitive)/np.diff(edges)

    def get_geometry(self, geo, interpolate=True):

        self.imported.get_geometry(geo, self.grid, interpolate=interpolate)

    def get_Zeff(self, key=None, interpolate=True):
        rho, Z = self.imported.get_Zeff(key=key)
        if not hasattr(Z, '__len__'):
            Z = np.ones(len(rho))*Z

        # interpolate onto Trinity grid
        if interpolate:
            f = interp1d(rho, Z, bounds_error=False, fill_value='extrapolate')
            Z = f(self.grid.rho)

        return Z


class TrinityReader():
    """Read data from a Trinity or Trinity-like .npy, .nc, or .bp file."""

    PICKLE = 0
    NETCDF4 = 1
    ADIOS2 = 2

    def __init__(self, grid, fin, frame=-1):

        self.grid = grid
        self.frame = -1 if frame is None else frame
        if not isinstance(self.frame, (int, np.integer)) or isinstance(self.frame, bool):
            raise ValueError('Trinity restart frame must be an integer.')
        fin = str(fin)
        self.output_format = None
        self.data = None
        self.rho = None

        if fin.endswith(".npy"):
            self.output_format = self.PICKLE
            self.data = np.load(fin, allow_pickle=True).tolist()
            self.rho = self.data["grid"].get("rho", self.data["grid"].get("rho_axis"))

        elif fin.endswith(".nc"):
            self.output_format = self.NETCDF4
            self.data = Dataset(fin, "r", format="NETCDF4")
            grid_group = self.data['grid']
            key = 'rho' if 'rho' in grid_group.variables else 'rho_axis'
            self.rho = grid_group[key][:]

        elif fin.endswith(".bp") or fin.endswith(".bp/"):
            self.output_format = self.ADIOS2
            self.data = adios2.FileReader(fin)
            name = 'grid/rho' if 'grid/rho' in self.data.available_variables() else 'grid/rho_axis'
            self.rho = self.data.read(name)

        else:
            raise ValueError("Unrecognized Trinity restart format; expected .npy, .nc or .bp.")

        self.rho = self._numeric_data(self.rho)
        if (self.rho.ndim != 1 or self.rho.size < 2 or not np.all(np.isfinite(self.rho))
                or not np.all(np.diff(self.rho) > 0)):
            raise ValueError('Saved Trinity restart grid must be finite and strictly increasing.')
        saved_label = self._read_static('grid', 'flux_label', default=None)
        if saved_label is not None and saved_label != grid.flux_label:
            raise ValueError(f'Restart radial coordinate {saved_label!r} does not match grid {grid.flux_label!r}.')

    def __del__(self):
        """Ensure proper cleanup of open files."""
        if self.output_format != self.PICKLE and self.data is not None:
            self.data.close()

    def __getstate__(self):
        """Prepare object state for pickling."""
        state = self.__dict__.copy()
        state.pop('data', None)  # Do not pickle the data file
        return state

    def __setstate__(self, state):
        """Restore object state from pickled data."""
        self.__dict__.update(state)
        self.data = None  # Did not pickle the data file

    @staticmethod
    def _numeric_data(value):
        if np.ma.isMaskedArray(value):
            return value.astype(float).filled(np.nan)
        return np.asarray(value)

    def _read_static(self, group, key, default=None):
        if self.output_format == self.PICKLE:
            val = self.data.get(group, {}).get(key, default)
        elif self.output_format == self.NETCDF4:
            group_obj = self.data.groups.get(group)
            val = group_obj[key][...] if group_obj is not None and key in group_obj.variables else default
        else:
            full_key = f'{group}/{key}'
            val = self.data.read_attribute(full_key) if full_key in self.data.available_attributes() else default
        if val is None:
            return None
        value = np.asarray(val)
        if value.size != 1:
            raise ValueError(f'Restart metadata {group}/{key} must contain a single value.')
        scalar = value.item()
        return scalar.decode() if isinstance(scalar, bytes) else scalar

    def read_data(self, group_name, variable_name):
        """Read data from the appropriate file format."""
        try:
            match self.output_format:
                case self.PICKLE:
                    # Legacy time histories are top-level; static grouped
                    # data belongs to its group and is not a time history.
                    val = self.data[variable_name]
                    if not (isinstance(val, list) and all(isinstance(v, str) for v in val)):
                        val = np.array(val)

                case self.NETCDF4:
                    group = self.data.groups.get(group_name)
                    if not group:
                        raise KeyError(f"Group {group_name} not found in NetCDF file.")
                    val = group.variables.get(variable_name)
                    if val is None:
                        raise KeyError(f"Variable {variable_name} not found in group {group_name}.")
                    val = val[:]
                    if np.ma.isMaskedArray(val):
                        val = self._numeric_data(val)

                case self.ADIOS2:
                    val = self.data.read(f"{group_name}/{variable_name}")
                    if val is None:
                        raise KeyError(f"Variable {group_name}/{variable_name} not found in ADIOS2 file.")

                case _:
                    raise ValueError("Unknown file format")

            return val[self.frame]

        except KeyError as e:
            raise KeyError(f"Error reading {group_name} and {variable_name}: {e}") from e

    def _fetch_species_data(self, key, tag, prefix):
        """Helper function to fetch species data with a default key pattern."""
        var = key if key else f"{prefix}_{tag}"
        value = self._numeric_data(self.read_data("species", var))
        if value.shape != self.rho.shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Restart profile {var} is unavailable at frame {self.frame} or has the wrong radial shape.")
        return self.rho, value

    def get_density(self, key=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        """Get density profile."""
        return self._fetch_species_data(key, tag, "n")

    def get_temperature(self, key=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        """Get temperature profile."""
        return self._fetch_species_data(key, tag, "T")

    def _restart_source(self, key, tag, prefix, norm_key, norms):
        if key is not None and not isinstance(key, str):
            raise ValueError('A Trinity restart source expects one profile name.')
        rho, value = self._fetch_species_data(key, tag, prefix)
        # Never reinterpret a different normalization as equivalent physical power.
        ref_old = self._read_static('norms', norm_key)
        if ref_old is None:
            t_ref = self._read_static('norms', 't_ref')
            if t_ref is not None and np.isfinite(t_ref) and t_ref > 0.:
                ref_old = ((1e20 * 1e3 * ELEMENTARY_CHARGE / 1e6)
                           if norm_key == 'P_ref_MWm3' else 1.) / t_ref
        if ref_old is None or not np.isfinite(ref_old) or ref_old <= 0.:
            raise ValueError(f'Trinity restart is missing a valid saved source normalization {norm_key}.')
        ref_new = 1. if norms is None else getattr(norms, norm_key)
        if not np.isfinite(ref_new) or ref_new <= 0.:
            raise ValueError(f'Trinity restart target reference {norm_key} must be finite and positive.')
        return rho, np.asarray(value)*(ref_old/ref_new)

    def get_density_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        return self._restart_source(keys, tag, 'Sn_aux', 'Sn_ref_SI20', norms)

    def get_pressure_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        return self._restart_source(keys, tag, 'Sp_aux', 'P_ref_MWm3', norms)

    def get_face_flux(self, kind, keys, tag):
        # Only diagnostics with physical throughput units, never qflux/pflux [GB].
        prefix, unit = ('Gam', 'SI20') if kind == 'particle' else ('Q', 'MW')
        name = keys if keys is not None else f'{prefix}_{unit}_{tag}'
        if not isinstance(name, str) or not name.startswith(prefix+'_') or f'_{unit}_' not in name:
            raise ValueError(f"Trinity {kind} flux requires one physical diagnostic name ({prefix}_{unit}_<tag>).")
        try:
            valid = self.read_data('time', 't_diagnostics_valid')
        except KeyError:
            pass  # Older official outputs predate the validity field.
        else:
            if np.asarray(valid).size != 1 or np.asarray(valid).item() != 1:
                raise ValueError(f"Trinity flux diagnostics are unavailable at frame {self.frame}.")
        if self.output_format == self.PICKLE:
            faces = self.data['grid'].get('midpoints')
        elif self.output_format == self.NETCDF4:
            grid = self.data['grid']
            faces = grid['midpoints'][:] if 'midpoints' in grid.variables else None
        else:
            faces = self.data.read('grid/midpoints') if 'grid/midpoints' in self.data.available_variables() else None
        if faces is None:
            faces = (self.rho[:-1] + self.rho[1:])/2
        return faces, self.read_data('species', name)

    def get_geometry(self, geo, grid, interpolate=True):
        if self.output_format != self.PICKLE:
            raise NotImplementedError('Importing a geometry object requires a Trinity .npy output file.')
        if not np.array_equal(grid.rho, self.rho):
            raise ValueError('Importing a saved Trinity geometry requires the original radial grid.')
        saved = self.data['geo_obj']
        if getattr(saved,'geo_option',None) != 'miller':
            raise ValueError("T3D geometry restart currently requires a saved Miller geometry.")
        # Copy only imported data, not the old run's flux executables,
        # grids or controllers. The new input owns those runtime objects.
        metrics = ('area', 'area_grid', 'grho', 'grho_grid', 'rhotor', 'rminor',
                   'rmaj', 'rmajor_prime', 'qsf', 'shat', 'kappa', 'tri', 'delta',
                   'kappa_prime', 's_kappa', 'delta_prime', 's_delta',
                   'beta_prime_alpha', 'B_ref', 'a_ref')
        for name in metrics:
            if hasattr(saved,name):
                value = getattr(saved,name)
                if isinstance(value,(GridProfile,FluxProfile)):
                    value = type(value)(value.profile.copy(),grid)
                setattr(geo,name,value)
        for name in ('Btor','B0','a_minor','AspectRatio','iota23'):
            if hasattr(saved,name):
                setattr(geo,name,getattr(saved,name))
        if geo.force_beta_prime_species:
            geo.beta_prime = np.array(self.data['species'][self.frame].beta_prime)


class AsciiColumnReader():
    '''
    Read generic columned data from an ascii text file
    '''

    def __init__(self, grid, fin, columns=[], divide_by=[]):

        self.grid = grid
        if not columns or len(set(columns)) != len(columns):
            raise ValueError('ASCII import requires unique column names.')
        if len(divide_by) != len(columns) or not np.all(np.isfinite(divide_by)) or np.any(np.asarray(divide_by) == 0.):
            raise ValueError('ASCII divide_by must contain one finite nonzero divisor per column.')
        self.columns = columns
        self.divide_by = divide_by
        self.data = defaultdict(list)
        numlines = 0
        with open(fin) as f:
            for line_index, line in enumerate(f, 1):
                data = line.partition('#')[0].split()
                if not data:
                    continue
                if len(data) < len(columns):
                    raise ValueError(f'Not enough values for configured ASCII columns at line {line_index}.')
                for i, k in enumerate(columns):
                    self.data[k].append(float(data[i])/divide_by[i])
                numlines += 1

        if 'rho' not in self.data.keys():
            warn("A radial coordinate 'rho' was not specified in the import.columns list. Assuming rho = [0, ..., 1]")
            self.data['rho'] = np.linspace(0, 1, num=numlines)

    def get_density(self, key=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        if key:
            n = np.array(self.data[key])
            return self.data['rho'], n
        elif tag:
            n = np.array(self.data[f'n_{tag}'])
            return self.data['rho'], n
        else:
            return None, None

    def get_temperature(self, key=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        if key:
            T = np.array(self.data[key])
            return self.data['rho'], T
        elif tag:
            T = np.array(self.data[f'T_{tag}'])
            return self.data['rho'], T
        else:
            return None, None

    def _columns_source(self, keys, tag, prefix, reference):
        if keys is None:
            if tag is None:
                raise ValueError('ASCII source requires keys or a species tag.')
            keys = [f'{prefix}_{tag}']
        keys = [keys] if isinstance(keys, str) else list(keys)
        if not keys or any(key not in self.data for key in keys):
            missing = [key for key in keys if key not in self.data]
            raise ValueError(f'Missing ASCII source column(s): {missing or "no columns requested"}.')
        return self.data['rho'], np.sum([self.data[key] for key in keys], axis=0)/reference

    def get_density_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        return self._columns_source(keys, tag, 'Sn_aux', 1. if norms is None else norms.Sn_ref_SI20)

    def get_pressure_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        return self._columns_source(keys, tag, 'Sp_aux', 1. if norms is None else norms.P_ref_MWm3)


    def get_face_flux(self, kind, keys, tag):
        if 'rho' not in self.columns:
            raise ValueError("ASCII flux import requires an explicit 'rho' column.")
        prefix = 'Gam_SI20' if kind == 'particle' else 'Q_MW'
        names = [f'{prefix}_{tag}'] if keys is None else ([keys] if isinstance(keys, str) else keys)
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name or name == 'rho' for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("Flux import keys must be a column name or a list of unique flux column names.")
        return self._columns_source(names, tag, prefix, 1.)


class JettoReader():
    '''
    Read data from a JETTO file
    '''
    def __init__(self, fin, time=None, flux_label='rminor'):
        self.time = time
        self.flux_label = flux_label
        self.finput = fin
        with Dataset(fin, mode='r') as f:
            if time is None:
                self.time_index = -1
            else:
                timeline = np.ma.asarray(f['time'][:],dtype=float).filled(np.nan)
                if not np.isfinite(self.time) or not np.all(np.isfinite(timeline)):
                    raise ValueError("JETTO time stamps and requested time must be finite.")
                self.time_index = np.argmin(abs(self.time - timeline))
            if flux_label == "rminor":
                rho = (f['R'][self.time_index, :] - f['RI'][self.time_index, :]) / 2
                rho = rho / rho[-1]
            elif flux_label == "polflux":
                psi = f['PSI'][self.time_index, :].data
                rho = (psi - psi[0]) / (psi[-1] - psi[0])
            elif flux_label in ("torflux", "rhotor"):
                rho = f['X'][self.time_index, :].data
            else:
                raise ValueError(
                    f"Unrecognized JETTO radial coordinate {flux_label!r}.")
            self.rho = rho

    def get_density(self, key=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        with Dataset(self.finput, mode='r') as f:
            if key:
                n = np.ma.asarray(f[key][self.time_index, :],dtype=float).filled(np.nan)
            else:
                raise NotImplementedError(
                    "JettoReader requires species key to be specified")

        # convert to Trinity units, 10^20 m^-3
        n = n * 1e-20

        return self.rho, n

    def get_temperature(self, key=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        with Dataset(self.finput, mode='r') as f:
            if key:
                T = np.ma.asarray(f[key][self.time_index, :],dtype=float).filled(np.nan)
            else:
                raise NotImplementedError(
                    "JettoReader requires species key to be specified")

        # convert to Trinity units, keV
        T = T * 1e-3

        return self.rho, T

    def get_density_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        with Dataset(self.finput, mode='r') as f:
            if keys is None:
                raise NotImplementedError(
                    "JettoReader requires species keys to be specified")
            Sn_tot = 0
            for key in ([keys] if isinstance(keys, str) else keys):
                Sn = np.ma.asarray(f[key][self.time_index, :],dtype=float).filled(np.nan)
                Sn_tot += Sn

        # convert to Trinity units
        # JETTO sources are in 1/(m^3 s)
        Sn_tot = Sn_tot * 1e-20 / (1. if norms is None else norms.Sn_ref_SI20)

        return self.rho, Sn_tot

    def get_pressure_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        with Dataset(self.finput, mode='r') as f:
            if keys is None:
                raise NotImplementedError(
                    "JettoReader requires species keys to be specified")
            Sp_tot = 0
            for key in ([keys] if isinstance(keys, str) else keys):
                Sp = np.ma.asarray(f[key][self.time_index, :],dtype=float).filled(np.nan)
                Sp_tot += Sp

        # convert to Trinity units
        # JETTO sources are in W/m^3
        Sp_tot = Sp_tot * 1e-6 / (1. if norms is None else norms.P_ref_MWm3)

        return self.rho, Sp_tot

    def get_geometry(self, geo, grid, interpolate=True):
        with Dataset(self.finput, mode='r') as f:
            # 0d quantities
            aminor = (f['R'][self.time_index, -1] - f['RI'][self.time_index, -1]) / 2
            rgeo = (f['R'][self.time_index, -1] + f['RI'][self.time_index, -1]) / 2
            B_T = np.abs(f['F'][self.time_index, -1] / rgeo)
            rgeo = rgeo / aminor

            # 1d quantities
            rmajor = (f['R'][self.time_index, :] + f['RI'][self.time_index, :]) / 2
            rmajor = rmajor / aminor
            rhotor = f['X'][self.time_index, :].data
            rminor = (f['R'][self.time_index, :] - f['RI'][self.time_index, :]) / 2
            rminor = rminor / aminor
            if self.flux_label == 'rminor':
                rho = rminor
                drho_drhotor = np.gradient(rminor, rhotor)
            elif self.flux_label == 'torflux':
                rho = rhotor
                drho_drhotor = np.ones(rhotor.shape)

            # Beta' from alpha
            palpha = 2/3 * f['WALD'][self.time_index, :].data
            p_prime = np.gradient(palpha, rminor)
            beta_prime_alpha = 8 * 1e-7 * np.pi * p_prime / B_T**2

            qsf = f['Q'][self.time_index, :].data
            kappa = f['ELO'][self.time_index, :].data
            tri = f['TRI'][self.time_index, :].data
            delta = np.arcsin(tri)  # GS2 definition of delta
            area = f['SURF'][self.time_index, :].data / aminor**2

            rmajor_prime = np.gradient(rmajor, rhotor) / drho_drhotor
            kappa_prime = np.gradient(kappa, rhotor) / drho_drhotor
            s_kappa = rho * np.gradient(kappa, rhotor) / drho_drhotor / kappa
            delta_prime = np.gradient(delta, rhotor) / drho_drhotor  # GS2 delta_prime
            s_delta = rho * np.gradient(tri, rhotor) / drho_drhotor
            shat = (rho/rhotor) / drho_drhotor * f['SH'][self.time_index, :].data
            # This is grad rho where rho is the jetto
            # radial coordinate
            grho = np.gradient(rho, rminor) / np.sqrt(
                f['GROQ'][self.time_index, :].data)

            geo.a_minor = aminor
            geo.a_ref = aminor
            geo.Btor = B_T
            geo.a_minor = aminor
            geo.AspectRatio = rgeo
        if interpolate:

            f = interp1d(rho, area)
            geo.area = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, grho)
            geo.grho = FluxProfile(f(grid.midpoints), grid)
            geo.grho_grid = GridProfile(f(grid.rho), grid)

            f = interp1d(rho, rhotor)
            geo.rhotor = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, rminor)
            geo.rminor = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, rmajor)
            geo.rmaj = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, rmajor_prime)
            geo.rmajor_prime = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, qsf)
            geo.qsf = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, shat)
            geo.shat = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, kappa)
            geo.kappa = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, tri)
            geo.tri = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, delta)
            geo.delta = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, kappa_prime)
            geo.kappa_prime = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, s_kappa)
            geo.s_kappa = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, delta_prime)
            geo.delta_prime = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, s_delta)
            geo.s_delta = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, beta_prime_alpha)
            geo.beta_prime_alpha = FluxProfile(f(grid.midpoints), grid)


class TranspReader():
    '''
    Read data from a TRANSP u-file, which can be used for initial profiles, sources, geometry etc
    '''

    def __init__(self, fin, time=None, flux_label='torflux'):

        self.time = time
        self.flux_label = flux_label

        fin = Path(fin)
        if fin.suffix.lower() in {'.nc', '.cdf'}:
            self.read_nc(fin, time=self.time)
        else:
            self.read_ascii(fin, time=self.time)

    def read_nc(self, fin2d, time=None):
        raise NotImplementedError('TRANSP NetCDF import is not implemented; use a TRANSP ASCII U-file.')

    def read_ascii(self, fin2d, time=None):
        fin2d = Path(fin2d)
        if not fin2d.stem.endswith('2d'):
            raise ValueError("TRANSP ASCII profile filename stem must end in '2d'.")
        fin0d = fin2d.with_name(fin2d.stem[:-2]+'0d'+fin2d.suffix)
        fin1d = fin2d.with_name(fin2d.stem[:-2]+'1d'+fin2d.suffix)

        self.meta = {}
        self.data = {}

        # read 0d data
        with open(fin0d) as f:
            datain = f.readlines()

        label = datain[0].strip().split(',')
        value = datain[1].strip().split(',')

        for j in np.arange(len(label)):

            self.data[label[j]] = value[j]

        # read 1d (time) data
        with open(fin1d) as f:
            datain = f.readlines()

        block_length1d = 0
        for d in datain:
            block_length1d += 1
            if d[:10] == '**********':
                block_length1d += 1
                break

        N_blocks = len(datain) // block_length1d
        for j in np.arange(N_blocks):

            start = j*block_length1d
            end = start + block_length1d

            block = datain[start:end]
            obj = TData1D(block, block_length1d, time=time)

            self.meta[obj.label] = obj
            self.data[obj.label] = obj.data

        # read 2d (time, rho) data
        with open(fin2d) as f:
            datain = f.readlines()

        block_length2d = 0
        for d in datain:
            block_length2d += 1
            if d[:10] == '**********':
                block_length2d += 1
                break

        N_blocks = len(datain) // block_length2d
        for j in np.arange(N_blocks):

            start = j*block_length2d
            end = start + block_length2d

            block = datain[start:end]
            obj = TData2D(block, block_length2d, time=time)

            self.meta[obj.label] = obj
            self.data[obj.label] = obj.data

    def find_ion_species_index(self, species_mass, species_charge=None):
        # Match both TRANSP identity fields (isobars can share a mass).
        # Never substitute a copy of NE for a missing ion profile.
        candidates = []
        for i in range(1, 10):
            try:
                mass = float(self.data[f'NM{i}A'])
                charge = float(self.data[f'NM{i}Z'])
            except (KeyError, ValueError, TypeError):
                continue
            if mass > 0 and charge > 0:
                candidates.append((i, mass, charge))
        if not candidates and 'NM1' in self.data:
            candidates = [(1, 2., 1.)]  # legacy lone thermal deuterium channel
        matches = [i for i, m, z in candidates if np.isclose(m, species_mass, rtol=1e-3, atol=0)
                   and (species_charge is None or np.isclose(z, species_charge, rtol=1e-3, atol=0))]
        if not matches:
            raise ValueError(f"No matching TRANSP ion for mass={species_mass}, charge={species_charge}; specify a profile key.")
        if len(matches) > 1:
            raise ValueError(f"TRANSP ion identity is ambiguous for mass={species_mass}, charge={species_charge}; specify charge or key.")
        return matches[0]

    def _profile_on_file_grid(self, key):
        if key not in self.meta or key not in self.data:
            raise ValueError(f"TRANSP profile {key!r} not found.")
        return np.asarray(self.meta[key].rho), np.asarray(self.data[key])

    def _radial_coordinate(self, rhotor):
        if self.flux_label != 'rminor':
            return rhotor
        axis, minor = self._profile_on_file_grid('RMINOR')
        # Existing U-file exports generally use the same toroidal grid.
        if np.array_equal(axis, rhotor):
            return minor/float(self.data['AMIN'])
        # Some U-files store geometry and thermodynamic channels at
        # different centers near the magnetic axis. r(0)=0 is exact.
        if axis[0] > 0:
            axis, minor = np.r_[0., axis], np.r_[0., minor]
        return interp1d(axis, minor, bounds_error=False, fill_value='extrapolate')(rhotor)/float(self.data['AMIN'])

    def get_density(self, key=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        if key is None:
            key = 'NE' if species_charge == -1 else f'NM{self.find_ion_species_index(species_mass, species_charge)}'
        rho, n = self._profile_on_file_grid(key)
        return self._radial_coordinate(rho), n*1e-20

    def get_temperature(self, key=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        rho, T = self._profile_on_file_grid(key or ('TE' if species_charge == -1 else 'TI'))
        return self._radial_coordinate(rho), T*1e-3

    def _sum_source(self, keys, species_mass, species_charge, electron, ion, negative):
        automatic = keys is None
        if automatic:
            if species_charge == -1:
                keys = electron
            else:
                keys = ion if self.find_ion_species_index(species_mass, species_charge) == 1 else []
        keys = [keys] if isinstance(keys, str) else list(keys)
        total, rho = None, None
        for key in keys:
            if automatic and key not in self.data:
                continue  # optional diagnostics in TRANSP exports
            axis, values = self._profile_on_file_grid(key)
            if total is None:
                rho, total = axis, np.zeros_like(values, dtype=float)
            elif not np.array_equal(rho, axis) or total.shape != values.shape:
                raise ValueError(f"TRANSP source {key!r} does not share the radial grid of the other channels.")
            total += -values if key in negative else values
        if total is None:
            rho, values = self._profile_on_file_grid('NE')
            total = np.zeros_like(values, dtype=float)
        return self._radial_coordinate(rho), total

    def get_density_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        rho, total = self._sum_source(keys, species_mass, species_charge, ['SBE', 'SNBIE', 'DNER'],
                                      ['SNBII', 'SWALL', 'DNER'], {'DNER'})
        return rho, total*1e-20/(1. if norms is None else norms.Sn_ref_SI20)

    def get_pressure_source(self, keys=None, species_mass=2.0, species_charge=None, tag=None, norms=None):
        rho, total = self._sum_source(keys, species_mass, species_charge,
                                      ['QNBIE', 'QICRHE', 'QECHE', 'QLHE', 'QIBWE', 'QWALLE', 'QOHM', 'DWER'],
                                      ['QNBII', 'QICRHI', 'QECHI', 'QLHI', 'QIBWI', 'QWALLI', 'DWIR'], {'DWER', 'DWIR'})
        return rho, total*1e-6/(1. if norms is None else norms.P_ref_MWm3)

    def get_geometry(self, geo, grid, interpolate=True):

        # read in geometry quantities
        # 0d quantities
        aminor = self.data['AMIN']
        B_T = self.data['BT']
        rgeo = self.data['RGEO'] / aminor

        # 1d quantities, on rhotor grid
        rhotor = self.meta['RMINOR'].rho
        rminor = self.data['RMINOR']/aminor
        if self.flux_label == 'rminor':
            rho = rminor
            drho_drhotor = np.gradient(rminor, rhotor)
        elif self.flux_label == 'torflux':
            rho = rhotor
            drho_drhotor = np.ones(rhotor.shape)

        rmajor = self.data['RMAJOR']/aminor
        qsf = self.data['Q']
        kappa = self.data['KAPPAR']
        delta = np.arcsin(self.data['DELTAR'])
        area = self.data['SURF']/aminor**2

        # dF/drho = dF/drhotor * drhotor/drho = dF/drhotor / (drho/drhotor)
        rmajor_prime = np.gradient(rmajor, rhotor)/drho_drhotor
        kappa_prime = np.gradient(kappa, rhotor)/drho_drhotor
        delta_prime = np.gradient(delta, rhotor)/drho_drhotor
        shat = np.gradient(qsf, rhotor)/drho_drhotor*rho/qsf
        grho = self.data['GRHO1'] * aminor * drho_drhotor

        geo.Btor = B_T
        geo.a_minor = aminor
        geo.AspectRatio = rgeo

        if interpolate:
            f = interp1d(rho, area, bounds_error=False, fill_value='extrapolate')
            geo.area = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, grho, bounds_error=False, fill_value='extrapolate')
            geo.grho = FluxProfile(f(grid.midpoints), grid)
            geo.grho_grid = GridProfile(f(grid.rho), grid)

            f = interp1d(rho, rhotor, bounds_error=False, fill_value='extrapolate')
            geo.rhotor = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, rminor, bounds_error=False, fill_value='extrapolate')
            geo.rminor = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, rmajor, bounds_error=False, fill_value='extrapolate')
            geo.rmaj = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, rmajor_prime, bounds_error=False, fill_value='extrapolate')
            geo.rmajor_prime = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, qsf, bounds_error=False, fill_value='extrapolate')
            geo.qsf = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, shat, bounds_error=False, fill_value='extrapolate')
            geo.shat = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, kappa, bounds_error=False, fill_value='extrapolate')
            geo.kappa = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, delta, bounds_error=False, fill_value='extrapolate')
            geo.delta = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, kappa_prime, bounds_error=False, fill_value='extrapolate')
            geo.kappa_prime = FluxProfile(f(grid.midpoints), grid)

            f = interp1d(rho, delta_prime, bounds_error=False, fill_value='extrapolate')
            geo.delta_prime = FluxProfile(f(grid.midpoints), grid)

    def get_Zeff(self, key=None):
        if key:
            rhotor, Z = self.meta[key].rho, self.data[key]
        else:
            try:
                rhotor, Z = self.meta['ZEFFR'].rho, self.data['ZEFFR']
            except:
                rhotor, Z = self.meta['NE'].rho, self.data['ZEFF']

        if self.flux_label == "rminor":
            try:
                rmin = self.data['RMINOR']/self.data['AMIN']
            except:
                rmin = self.data['RMINOR']/np.expand_dims(self.data['AMIN'], axis=1)
            rho = rmin
        else:
            rho = rhotor

        return rho, Z


class TData():
    @staticmethod
    def read_header(block):

        header_dict = {}
        for line in block:
            val, key = line.strip().split(';')
            tag = key.split('-')[1]  # get piece in between "-tag-"
            header_dict[tag] = val

        return header_dict

    @staticmethod
    def parse_line(line, j=13):

        N = len(line)
        return [line[idx:idx + j] for idx in range(0, N, j)]

    @staticmethod
    def parse(line, j=13):

        i = 1
        num = line[i:i+j]
        arr = []
        while num.strip():
            arr.append(num)

            # set next round
            i += j
            num = line[i:i+j]

        return arr

    @staticmethod
    def read_block(block):

        data = []
        for line in block:
            try:
                # reads most lines
                row = np.array(line.strip().split(' '), float)
            except:
                # reads lines with consectutive negs
                row = np.array(TData.parse(line), float)
                # np.array(parse_line(line[1:-1]), float)

            for n in row:
                data.append(n)

        return np.array(data)


class TData1D(TData):
    def __init__(self, datain, block_length, time=None, header_length=7, footer_length=3, label_length=15):

        start = header_length
        header = datain[:start]
        head = TData.read_header(header)
        len_data = int(head['# OF PTS'])

        end = block_length - footer_length
        block = datain[start:end]

        data = TData.read_block(block)

        var_label = head['DEPENDENT VARIABLE LABEL']

        label = var_label[:label_length].strip()
        units = var_label[label_length:].strip()

        xaxis = head['INDEPENDENT VARIABLE LABEL'].strip()

        x = data[:len_data]
        y = data[len_data:]

        self.len_data = len_data
        self.header = head
        self.label = label
        self.units = units
        self.xaxis = xaxis
        self.times = x
        self.data = y

        if time:
            # find time index with time closest to target time
            tid = (np.abs(x - time)).argmin()
            self.data = self.data[tid]
            self.time = self.times[tid]


class TData2D(TData):
    def __init__(self, datain, block_length, time=None, header_length=9, footer_length=3, label_length=15):

        start = header_length
        end = block_length - footer_length

        header = datain[:start]
        block = datain[start:end]

        head = TData.read_header(header)
        data = TData.read_block(block)

        xlen_data = int(head['# OF X PTS'])
        ylen_data = int(head['# OF Y PTS'])
        var_label = head['DEPENDENT VARIABLE LABEL']

        label = var_label[:label_length].strip()
        units = var_label[label_length:].strip()

        xaxis = head['INDEPENDENT VARIABLE (X) LABEL'].strip()
        yaxis = head['INDEPENDENT VARIABLE (Y) LABEL'].strip()

        x = data[:xlen_data]
        y = data[xlen_data:xlen_data + ylen_data]
        f = data[xlen_data+ylen_data:]

        self.xlen_data = xlen_data
        self.ylen_data = ylen_data
        self.header = head
        self.label = label
        self.units = units
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.rho = x
        self.times = y
        self.data = np.reshape(f, (ylen_data, xlen_data))

        if time:
            # find time index with time closest to target time
            tid = (np.abs(y - time)).argmin()
            self.data = self.data[tid, :]
            self.time = self.times[tid]
