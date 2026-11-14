import numpy as np

from t3d.FluxModels import FluxModel


class Import_FluxModel(FluxModel):
    """Insert prescribed physical throughputs, not a gyro-Bohm closure."""

    def __init__(self, pars, grid, time):
        super().__init__(pars, grid, time)
        self.label = pars.get('label', 'Import')

    def initialize(self, species, imported):
        shape = (species.N_species, self.N_fluxtubes)
        self.particles = np.zeros(shape)
        self.power = np.zeros(shape)
        configured = False
        for i, s in enumerate(species.get_species_list()):
            for field, active, values, getter in (
                ('particle_flux', species.n_evolve_keys, self.particles, imported.get_particle_flux),
                ('heat_flux', species.T_evolve_keys, self.power, imported.get_heat_flux),
            ):
                table = s.flux_import[field]
                if not table.get('import', False):
                    continue
                if s.type not in active:
                    raise ValueError(f"Imported {field} for {s.type} requires an independently evolved equation.")
                values[i] = getter(keys=table.get('keys'), tag=s.type_tag)
                configured = True
        if not configured:
            raise ValueError("Import flux model requires at least one species particle_flux or heat_flux import.")

    def compute_fluxes(self, species, geo, norms):
        # Inverse of the dimensional Gam_SI20 and Q_MW diagnostics.
        a3 = norms.a_meters**3
        species.add_prescribed_flux(
            self.particles/(norms.Sn_ref_SI20*a3),
            (2./3.)*self.power/(norms.P_ref_MWm3*a3), self.label)
