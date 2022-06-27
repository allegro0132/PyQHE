# %%
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid

import pyqhe.utility.constant as const


class PoissonODE:
    """ODE integration solver for 1d Poisson equation.

    Args:
        charge_density: The net charge density. Typically, use the dopants
            density minus the electron density.
    """

    def __init__(self, grid: np.ndarray, charge_density: np.ndarray,
                 eps: np.ndarray) -> None:
        self.grid = grid
        self.charge_density = charge_density
        self.eps = eps
        # Cache parameters
        self.e_field = None
        self.v_potential = None

    def calc_poisson(self, **kwargs):
        """Calculate electric field."""

        def righthand(z, y):
            # interpolate
            sigma = np.interp(z, self.grid, self.charge_density)
            return sigma * -const.q

        sol = solve_ivp(righthand, (self.grid[0], self.grid[-1]), [0],
                        t_eval=self.grid,
                        **kwargs)
        # divide dielectric `eps`
        self.e_field = sol.y.flatten() / self.eps
        # integral the potential
        self.v_potential = cumulative_trapezoid(self.e_field,
                                                self.grid,
                                                initial=0)

        return self.v_potential


class PoissonFDM:
    """ODE integration solver for 1d Poisson equation.

    Args:
        charge_density: The net charge density. Typically, use the dopants
            density minus the electron density.
    """

    def __init__(self, grid: np.ndarray, charge_density: np.ndarray,
                 eps: np.ndarray) -> None:
        self.grid = grid
        self.charge_density = charge_density
        self.eps = eps
        # Cache parameters
        self.e_field = None
        self.v_potential = None

    def calc_poisson(self, **kwargs):
        """Calculate electric field."""

        F0 = -np.sum(const.q * self.charge_density) / (2.0)
        # is the above necessary since the total field due to the structure should be zero.
        # Do running integral
        tmp = (np.hstack(
            ([0.0], self.charge_density[:-1])) + self.charge_density
              )  # using trapezium rule for integration (?).
        tmp *= (
            const.q / 2.0
        )  # Note: sigma is a number density per unit area, needs to be converted to Couloumb per unit area
        tmp[0] = F0
        f = np.cumsum(tmp) / self.eps

        self.e_field = f
        # integral the potential
        self.v_potential = cumulative_trapezoid(self.e_field,
                                                self.grid,
                                                initial=0)

        return self.v_potential * const.q


# %%
# # QuickTest
# from matplotlib import pyplot as plt

# grid = np.linspace(0, 10, 80)
# eps = np.ones(grid.shape)
# sigma = np.zeros(grid.shape)
# sigma[30:41] = -1
# sigma[60:71] = 1
# solver = PoissonFDM(grid, sigma, eps)
# solver.calc_poisson()

# plt.plot(grid, solver.v_potential)
# # %%
# plt.plot(grid, solver.e_field)
# %%
