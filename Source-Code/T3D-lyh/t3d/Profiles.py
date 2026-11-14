import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded


class Profile():
    '''
    This class is called within the Trinity-Engine library

    It handles general profiles (n, p, F, gamma, Q, etc)
    with options to evaluate half-steps and gradients when initializing new Profile objects
    '''

    @staticmethod
    def _values_on_grid(arr, size):
        values = np.asarray(arr, dtype=float)
        if values.ndim == 0:
            return np.full(size, values, dtype=float)
        if values.shape != (size,):
            raise ValueError(f"Expected {size} radial values, got shape {values.shape}")
        return values

    def __init__(self, arr, grid):

        # take a 1D array to be density, for example
        self.profile = np.array(arr)
        self.length = len(arr)

        try:
            self.rho = grid.rho
        except:
            # backwards compatible for older t3d output 5/2024
            self.rho = grid.rho_axis

    def eval_at(self, rho_eval):
        interp = interp1d(self.rho, self.profile)
        return interp(rho_eval)

    def plot(self, show=False, new_fig=False, label=None):

        if new_fig:
            plt.figure(figsize=(4, 4))

        plt.plot(self.rho, self.profile, '.-', label=label)

        if show:
            plt.show()

    __array_ufunc__ = None

    # operator overloads that automatically dereference the profiles
    def __add__(A, B):
        if isinstance(B, A.__class__):
            return A.__class__(A.profile + B.profile, A.grid)
        elif isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile) or not hasattr(B, '__len__'):
            return A.__class__(A.profile + B, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__add__")

    def __radd__(A, B):
        return A.__add__(B)

    def __sub__(A, B):
        if isinstance(B, A.__class__):
            return A.__class__(A.profile - B.profile, A.grid)
        elif isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile) or not hasattr(B, '__len__'):
            return A.__class__(A.profile - B, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__sub__")

    def __rsub__(A, B):
        return -1*(A.__sub__(B))

    def __mul__(A, B):
        if isinstance(B, A.__class__):
            return A.__class__(A.profile * B.profile, A.grid)
        elif (hasattr(B, 'size') and B.size == 1) or (isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile)) or not hasattr(B, '__len__'):
            return A.__class__(A.profile * B, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__mul__")

    def __rmul__(A, B):
        return A.__mul__(B)

    def __truediv__(A, B):
        if isinstance(B, A.__class__):
            return A.__class__(A.profile / B.profile, A.grid)
        elif isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile) or not hasattr(B, '__len__'):
            return A.__class__(A.profile / B, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__truediv__")

    def __rtruediv__(A, B):
        if isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile) or not hasattr(B, '__len__'):
            return A.__class__(B / A.profile, A.grid)
        else:
            raise Exception("Type mismatch in Profile.__truediv__")

    def __neg__(A):
        return -1*A

    def __pow__(A, b):
        return A.__class__(A.profile**b, A.grid)

    def __eq__(A, B):
        if not isinstance(B, A.__class__):
            return NotImplemented
        return np.array_equal(A.rho, B.rho) and np.array_equal(A.profile, B.profile)

    def __getitem__(A, i):
        return A.profile[i]

    def __setitem__(A, i, val):
        A.profile[i] = val

    def __len__(A):
        return len(A.profile)

    def __repr__(A):
        return A.profile.__repr__()

    def __set__(A, B):
        if isinstance(B, A.__class__):
            A = B
        elif isinstance(B, (list, tuple, np.ndarray)) and len(B) == len(A.profile):
            A.profile = B
        else:
            raise Exception("Type mismatch in Profile.__set__")


class GridProfile(Profile):
    '''
    Profile class with values on the N_radial rho grid points
    '''

    def __init__(self, arr, grid):

        self.grid = grid
        try:
            self.rho = grid.rho
        except:
            # backwards compatible for older t3d output 5/2024
            self.rho = grid.rho_axis

        self.length = len(self.rho)
        self.profile = self._values_on_grid(arr, self.length)

    def gradient_4pt(self):
        f = self.profile
        dr = self.grid.drho
        grad_f = _gradient_4pt(f, dr)

        return GridProfile(grad_f, self.grid)

    def gradient_5pt(self):
        f = self.profile
        dr = self.grid.drho
        grad_f = _gradient_5pt(f, dr)

        return GridProfile(grad_f, self.grid)

    def log_gradient(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.gradient_5pt()/self

    def gradient_as_FluxProfile(self):
        '''
        Returns a FluxProfile with gradient values evaluated at midpoints
        '''

        flux_length = self.length - 1
        grad_prof = np.zeros(flux_length)
        prof = self.profile
        dr = self.grid.drho

        if self.length == 3:
            # The two secants are exact for a quadratic at the midpoints.
            return FluxProfile(np.diff(prof)/dr, self.grid)

        # inner boundary (uncentered)
        ix = 0
        grad_prof[ix] = (-prof[ix+3] + 3.*prof[ix+2] + 21.*prof[ix+1] - 23.*prof[ix]) / (24.*dr)

        # outer boundary (uncentered)
        ix = flux_length - 1
        grad_prof[ix] = (prof[ix-2] - 3*prof[ix-1] - 21.*prof[ix] + 23.*prof[ix+1]) / (24.*dr)

        # interior (centered)
        for ix in np.arange(1, flux_length-1):
            grad_prof[ix] = (prof[ix-1] - 27.*prof[ix] + 27.*prof[ix+1] - prof[ix+2]) / (24.*dr)

        return FluxProfile(grad_prof, self.grid)

    def log_gradient_as_FluxProfile(self):
        '''
        Returns a FluxProfile with log gradient values
        '''
        with np.errstate(divide='ignore', invalid='ignore'):
            return self.log_gradient().toFluxProfile()

    def toFluxProfile(self):

        flux_length = self.length - 1
        flux_prof = np.zeros(flux_length)
        prof = self.profile

        # inner boundary (third order)
        ix = 0
        flux_prof[ix] = 0.125*(3.0*prof[ix] + 6.0*prof[ix+1] - prof[ix+2])
        # if third order interpolated value is negative when it shouldn't be, use second order interpolation
        if flux_prof[ix] <= 0 and prof[ix] > 0 and prof[ix+1] > 0:
            flux_prof[ix] = 0.5*(prof[ix+1] + prof[ix])

        # outer boundary (third order)
        ix = flux_length-1
        flux_prof[ix] = 0.125*(3.0*prof[ix+1] + 6.0*prof[ix] - prof[ix-1])
        # if third order interpolated value is negative when it shouldn't be, use second order interpolation
        if flux_prof[ix] <= 0 and prof[ix] > 0 and prof[ix+1] > 0:
            flux_prof[ix] = 0.5*(prof[ix+1] + prof[ix])

        # interior (fourth order)
        for ix in np.arange(1, flux_length-1):
            flux_prof[ix] = 0.0625*(-prof[ix-1] + 9.0*prof[ix] + 9.0*prof[ix+1] - prof[ix+2])
            # if fourth order interpolated value is negative when it shouldn't be, use second order interpolation
            if flux_prof[ix] <= 0 and prof[ix] > 0 and prof[ix+1] > 0:
                flux_prof[ix] = 0.5*(prof[ix+1] + prof[ix])

        return FluxProfile(flux_prof, self.grid)

    # evaluate at j+1/2
    def plus(self):
        return self.toFluxProfile().plus()

    # evaluate at j-1/2
    def minus(self):
        return self.toFluxProfile().minus()

    def plus1(self):
        arr = np.roll(self.profile, -1)
        arr[-1] = 0.0
        return GridProfile(arr, self.grid)

    def minus1(self):
        arr = np.roll(self.profile, 1)
        arr[0] = 0.0
        return GridProfile(arr, self.grid)


class FluxProfile(Profile):
    '''
    Profile class with values on the N_radial-1 midpoints
    '''

    def __init__(self, arr, grid):

        self.grid = grid
        self.rho = grid.midpoints
        self.length = len(self.rho)
        self.profile = self._values_on_grid(arr, self.length)

    def toGridProfile(self, axis_val=0.0):

        grid_length = self.length + 1
        grid_prof = np.zeros(grid_length)
        prof = self.profile

        # inner boundary (third order)
        ix = 0
        grid_prof[ix] = 0.125*(3.0*axis_val + 6.0*prof[ix] - prof[ix+1])
        # if third order interpolated value is negative when it shouldn't be, use second order interpolation
        if grid_prof[ix] < 0 and axis_val > 0 and prof[ix] > 0:
            grid_prof[ix] = 0.5*(axis_val + prof[ix])

        # outer boundary (third order)
        ix = grid_length-2
        outer_left = axis_val if self.length == 2 else prof[ix-2]
        grid_prof[ix] = 0.125*(3.0*prof[ix] + 6.0*prof[ix-1] - outer_left)
        # if third order interpolated value is negative when it shouldn't be, use second order interpolation
        if grid_prof[ix] < 0 and prof[ix-1] > 0 and prof[ix] > 0:
            grid_prof[ix] = 0.5*(prof[ix-1] + prof[ix])

        # this is only second order accurate, but these values are never really needed (outer boundary is fixed)
        ix = grid_length-1
        grid_prof[ix] = 1.5*prof[ix-1] - 0.5*prof[ix-2]

        # interior (fourth order)
        if self.length == 2:
            return GridProfile(grid_prof, self.grid)

        ix = 1
        grid_prof[ix] = 0.0625*(-axis_val + 9.0*prof[ix-1] + 9.0*prof[ix] - prof[ix+1])
        # if fourth order interpolated value is negative when it shouldn't be, use second order interpolation
        if grid_prof[ix] < 0 and prof[ix-1] > 0 and prof[ix] > 0:
            grid_prof[ix] = 0.5*(prof[ix-1] + prof[ix])

        for ix in np.arange(2, grid_length-2):
            grid_prof[ix] = 0.0625*(-prof[ix-2] + 9.0*prof[ix-1] + 9.0*prof[ix] - prof[ix+1])
            # if fourth order interpolated value is negative when it shouldn't be, use second order interpolation
            if grid_prof[ix] < 0 and prof[ix-1] > 0 and prof[ix] > 0:
                grid_prof[ix] = 0.5*(prof[ix-1] + prof[ix])

        return GridProfile(grid_prof, self.grid)

    # FluxProfile already evaluated at j+/-1/2 grid, so just need to shift for minus
    def plus(self):
        return self

    def minus(self):
        arr = np.roll(self.profile, 1)
        arr[0] = 0.
        return FluxProfile(arr, self.grid)

    # def plus(self):
    #    return self.toFluxProfile()

    # def minus(self):
    #    return self.toFluxProfile().minus1()

    # def plus1(self):
    #    arr = np.roll(self.profile, -1)
    #    arr[-1] = 0.0
    #    return GridProfile(arr, self.grid)

    # def minus1(self):
    #    arr = np.roll(self.profile, 1)
    #    arr[0] = 0.0
    #    return GridProfile(arr, self.grid)


def _gradient_4pt(f, dx):
    """
    Four-point finite-difference gradients; quadratic fallback on three nodes.

    Args
    ----
    f: 1d array_like
    dx: float. spacing of the array.

    Returns
    -------
    array_like
    """
    n = len(f)
    grad_f = np.zeros(n)
    if n == 3:
        return np.gradient(np.asarray(f), dx, edge_order=2)

    # inner boundary
    grad_f[0] = (2.*f[3] - 9.*f[2] + 18.*f[1] - 11.*f[0]) / 6.
    grad_f[1] = (-f[3] + 6.*f[2] - 3.*f[1] - 2.*f[0]) / 6.

    # interior (4 pt centered)
    for ix in np.arange(2, n-2):
        grad_f[ix] = (f[ix-2] - 8.*f[ix-1] + 8.*f[ix+1] - f[ix+2]) / 12.

    # outer boundary
    grad_f[n-2] = (f[n-4] - 6.*f[n-3] + 3.*f[n-2] + 2.*f[n-1]) / 6.
    grad_f[n-1] = (-2.*f[n-4] + 9.*f[n-3] - 18.*f[n-2] + 11.*f[n-1]) / 6.

    return grad_f / dx


def _gradient_5pt(f, dx):
    """
    Up-to-6th-order-accurate gradients along an evenly-spaced 1d array.

    Args
    ----
    f: 1d array_like
    dx: float. spacing of the array.

    Returns
    -------
    array_like

    Notes
    -----
    Boundary points are 2nd-order accurate (2-pt compact difference).
    Next to boundary points are 4th-order accurate (2-pt centered compact difference).
    Interior points are 6th-order accurate (4-pt centered compact difference).
    See fd5pt_real in interp.f90 in fortran trinity.
    """
    f = np.asarray(f)
    n = len(f)

    grad_f = np.zeros(n)

    # The unchanged compact matrix in LAPACK banded storage:
    # ab[1+i-j, j] = A[i, j].
    ab = np.ones((3, n))
    ab[0, 0] = ab[2, -1] = 0.0
    ab[0, 1] = ab[2, -2] = 0.5
    ab[1] = 3.0
    ab[1, (0, -1)] = 0.5
    ab[1, (1, -2)] = 4.0

    # inner boundary
    grad_f[0] = (f[1] - f[0])/dx
    grad_f[1] = 3*(f[2] - f[0])/dx

    # interior
    grad_f[2:-2] = (0.25*(f[4:]-f[:-4]) + 7.*(f[3:-1]-f[1:-3])) / (3.*dx)

    # outer boundary
    grad_f[n-2] = 3*(f[n-1] - f[n-3])/dx
    grad_f[n-1] = (f[n-1] - f[n-2])/dx

    # The coefficients are finite and fixed; preserve the original propagation
    # of a nonfinite input profile rather than changing its meaning here.
    return solve_banded((1, 1), ab, grad_f, check_finite=False)
