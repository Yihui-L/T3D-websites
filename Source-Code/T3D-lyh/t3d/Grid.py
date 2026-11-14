import numpy as np
from t3d.Logbook import info, warn


class Grid():

    def __init__(self, inputs):

        # read grid parameters from input file
        grid_parameters = inputs.get('grid', {})
        self.N_radial = grid_parameters.get('N_radial', 10)
        self.rho_edge = grid_parameters.get('rho_edge', 0.8)
        if not isinstance(self.N_radial, int) or self.N_radial < 3:
            raise ValueError("N_radial must be an integer of at least 3")
        if not np.isfinite(self.rho_edge) or not 0 < self.rho_edge <= 1:
            raise ValueError("rho_edge must be finite and in (0, 1]")
        # BD: We should not allow users to set rho_inner until we have non-constant drho coded
        # self.rho_inner = grid_parameters.get('rho_inner', self.rho_edge / (2*self.N_radial - 1))
        if 'rho_inner' in grid_parameters:
            warn('The rho_inner input is currently not available, using default of rho_edge / (2*N - 1)')
        self.rho_inner = self.rho_edge / (2*self.N_radial - 1)
        self.flux_label = grid_parameters.get('flux_label', 'torflux')
        if self.flux_label not in ("rminor", "torflux"):
            raise ValueError(f"Unsupported radial coordinate {self.flux_label!r}")

        # compute additional parameters
        self.rho = np.linspace(self.rho_inner, self.rho_edge, self.N_radial)  # radial axis, N points
        self.midpoints = (self.rho[1:] + self.rho[:-1])/2  # midpoints, (N-1) points

        # TODO: consider case where this is non-constant
        self.drho = (self.rho_edge - self.rho_inner) / (self.N_radial - 1)

        info("\n  Grid Information")
        np.set_printoptions(precision=3)
        if self.flux_label == 'rminor':
            info("    Radial coordinate is rho = r/a (normalized minor radius)")
        elif self.flux_label == 'torflux':
            info("    Radial coordinate is rho = sqrt(toroidal_flux/toroidal_flux_LCFS)")
        info(f"    N_radial: {self.N_radial}")
        info(f"    rho grid:             {self.rho}")
        info(f"    flux (midpoint) grid:   {self.midpoints}")
