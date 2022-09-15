from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve

import pyqhe.utility.constant as const
from pyqhe.utility.utils import tensor


class PoissonSolver(ABC):
    """Meta-class for Poisson equation solver."""

    def __init__(self) -> None:
        # properties
        self.grid = None
        self.charge_density = None
        self.eps = None
        # Cache parameters
        self.e_field = None
        self.v_potential = None

    @abstractmethod
    def calc_poisson(self):
        """Solve Poisson equation and get related electric field and potential.
        """


class PoissonODE(PoissonSolver):
    """ODE integration solver for 1d Poisson equation.

    Args:
        charge_density: The net charge density. Typically, use the dopants
            density minus the electron density.
    """

    def __init__(self, grid: np.ndarray, charge_density: np.ndarray,
                 eps: np.ndarray) -> None:
        super().__init__()

        self.grid = grid
        self.charge_density = charge_density
        self.eps = eps

    def calc_poisson(self, **kwargs):
        """Calculate electric field."""

        d_z = cumulative_trapezoid(const.q * self.charge_density,
                                   self.grid,
                                   initial=0)
        self.e_field = d_z / self.eps
        # integral the potential
        # note here we put a electron, dV/dz = E
        self.v_potential = cumulative_trapezoid(self.e_field,
                                                self.grid,
                                                initial=0)

        return self.v_potential


class PoissonFDM(PoissonSolver):
    """ODE integration solver for 1d Poisson equation.

    Args:
        charge_density: The net charge density. Typically, use the dopants
            density minus the electron density.
    """

    def __init__(self,
                 grid: np.ndarray,
                 charge_density: np.ndarray,
                 eps: np.ndarray,
                 bound_dirichlet: np.ndarray = None,
                 bound_period: list = None) -> None:
        super().__init__()

        if not isinstance(grid, list):  # 1d grid
            grid = [grid]
        self.grid = grid
        self.dim = [grid_axis.shape[0] for grid_axis in self.grid]
        # check matrix dim
        if charge_density.shape == tuple(self.dim) or len(self.dim) == 1:
            self.charge_density = charge_density
        else:
            raise ValueError('The dimension of charge_density is not match')
        if eps.shape == tuple(self.dim) or len(self.dim) == 1:
            self.eps = eps
        else:
            raise ValueError('The dimension of eps is not match.')

        if bound_dirichlet is None:
            self.bound_dirichlet = None
        elif bound_dirichlet.shape == tuple(self.dim) or len(self.dim) == 1:
            self.bound_dirichlet = bound_dirichlet
        else:
            raise ValueError('The dimension of bound_dirichlet is not match.')
        if bound_period is None:
            self.bound_period = [None] * len(self.dim)
        elif len(bound_period) == len(self.dim):
            self.bound_period = bound_period
        else:
            raise ValueError('The dimension of bound_period is not match.')

    def build_d_matrix(self, loc):
        """Build 1D time independent Schrodinger equation kinetic operator.

        Args:
            dim: dimension of kinetic operator.
        """
        mat_d = -2 * np.eye(self.dim[loc]) + np.eye(
            self.dim[loc], k=-1) + np.eye(self.dim[loc], k=1)
        if self.bound_period[loc]:  # add period boundary condition
            mat_d[0, -1] = 1
            mat_d[-1, 0] = 1
        return mat_d

    def calc_poisson(self, **kwargs):
        """Calculate electric field."""

        # discrete laplacian
        a_mat_list = []

        for loc, _ in enumerate(self.dim):
            mat = self.build_d_matrix(loc)
            kron_list = [np.eye(idim) for idim in self.dim[:loc]] + [mat] + [
                np.eye(idim) for idim in self.dim[loc + 1:]
            ] + [1]  # auxiliary element for 1d solver
            delta = self.grid[loc][1] - self.grid[loc][0]
            # construct n-d kinetic operator by tensor product
            d_opt = tensor(*kron_list)
            # tensor contraction
            d_opt = np.einsum(d_opt.reshape(self.dim * 2),
                              np.arange(len(self.dim * 2)), self.eps / delta**2,
                              np.arange(len(self.dim)),
                              np.arange(len(self.dim * 2)))
            a_mat_list.append(
                d_opt.reshape(np.prod(self.dim), np.prod(self.dim)))
        a_mat = np.sum(a_mat_list, axis=0)
        b_vec = const.q * self.charge_density.flatten()

        if self.bound_dirichlet is not None:
            delta = self.grid[0][1] - self.grid[0][0]
            # adjust coefficient let matrix looks good
            bound_b = self.bound_dirichlet * self.eps / delta**2
            bound_a = np.zeros_like(b_vec)
            bound_loc = np.flatnonzero(~np.isnan(self.bound_dirichlet))
            bound_a[bound_loc] = 1 * self.eps.flatten()[bound_loc] / delta**2
            # tensor contraction
            bound_mat = np.diag(bound_a)
            a_mat[bound_loc] = bound_mat[bound_loc]
            b_vec[bound_loc] = bound_b.flatten()[bound_loc]

        self.v_potential = solve(a_mat, b_vec).reshape(self.dim)
        # calculate gradient of potential
        self.e_field = np.gradient(self.v_potential)
        return self.v_potential


# %%
# # QuickTest
if __name__ == '__main__':
    from matplotlib import pyplot as plt

    grid = np.linspace(0, 10, 80)
    eps = np.ones(grid.shape) * const.eps0
    sigma = np.zeros(grid.shape)
    sigma[30:41] = -1
    sigma[60:71] = 1
    sol = PoissonFDM(grid, sigma, eps)
    sol.calc_poisson()

    plt.plot(grid, sol.v_potential)
    plt.show()
    plt.plot(grid, sol.e_field)
    # %%
    delta = 0.02
    x = np.arange(-1, 1, delta)
    y = np.arange(-0.5, 0.5, delta)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    top_plate = (yv <= 0.1 + delta / 2) * (yv >= 0.1 - delta / 2)
    bottom_plate = (yv <= -0.1 + delta /2) * (yv >= -0.1 -delta / 2)
    length = (xv <= 0.5) * (xv >= -0.5)
    bound = np.empty_like(xv)
    bound[:] = np.nan
    bound[top_plate * length] = 1
    bound[bottom_plate * length] = -1
    sol = PoissonFDM([x, y], np.zeros_like(xv),
                     np.ones_like(xv) * const.eps0, bound)
    # sol = PoissonFDM([x, y], bound * 10,
    #                   np.ones_like(bound) * const.eps0)
    v_p = sol.calc_poisson()
    # v potential
    plt.pcolormesh(xv, yv, v_p)
    plt.show()
    # e field
    plt.pcolormesh(xv, yv, np.sqrt(sol.e_field[0]**2 + sol.e_field[1]**2))
# %%
