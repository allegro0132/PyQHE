# %%
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid

import pyqhe.utility.constant as const


class Poisson1D:
    """ODE integration solver for 1d Poisson equation.

    Args:
        charge_density: The net charge density. Typically, use the dopants
            density minus the electron density.
    """

    def __init__(self,
                 grid: np.ndarray,
                 charge_density: np.ndarray,
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
                        t_eval=self.grid, method='DOP853',
                        **kwargs)
        # divide dielectric `eps`
        self.e_field = sol.y.flatten() / self.eps
        # integral the potential
        self.v_potential = cumulative_trapezoid(self.e_field, self.grid, initial=0)

        return self.v_potential


# %%
# # QuickTest
# from matplotlib import pyplot as plt

# grid = np.linspace(0, 10, 80)
# eps = np.ones(grid.shape)
# sigma = np.zeros(grid.shape)
# sigma[30:41] = -1
# sigma[60:71] = 1
# solver = Poisson1D(grid, sigma, eps)
# solver.calc_poisson()

# plt.plot(grid, solver.v_potential)
# # %%
# plt.plot(grid, solver.e_field)

# # %%
# def calc_field(sigma, eps):
#     """calculate electric field as a function of z-
#     sigma is a number density per unit area
#     eps is dielectric constant"""
#     # i index over z co-ordinates
#     # j index over z' co-ordinates
#     # Note:
#     F0 = -np.sum(const.q * sigma) / (
#         2.0)  # CMP'deki i ve j yer değişebilir - de + olabilir
#     # is the above necessary since the total field due to the structure should be zero.
#     # Do running integral
#     tmp = (np.hstack(([0.0], sigma[:-1])) + sigma
#           )  # using trapezium rule for integration (?).
#     tmp *= (
#         const.q / 2.0
#     )  # Note: sigma is a number density per unit area, needs to be converted to Couloumb per unit area
#     tmp[0] = F0
#     F = np.cumsum(tmp) / eps

#     tmp = const.q * F * (grid[1] - grid[0])
#     V = np.cumsum(tmp)
#     return F, V  # electric field

# e_field, v = calc_field(sigma, eps)
# plt.plot(grid, e_field)
# # %%
# plt.plot(grid, v)
# # %%
