# %%
from abc import ABC, abstractmethod
import numpy as np
from scipy import optimize
import qutip as qt

import pyqhe.utility.constant as const


class SchrodingerSolver(ABC):
    """Meta class for Schrodinger equation solver."""

    def __init__(self) -> None:
        # properties
        self.grid = None
        self.v_potential = None
        self.cb_meff = None
        # Cache parameters
        self.psi = None

    @abstractmethod
    def calc_evals(self):
        """Calculate eigenenergy of any bound states in the chosen potential."""

    @abstractmethod
    def calc_esys(self):
        """Calculate wave function and eigenenergy."""


class SchrodingerShooting(SchrodingerSolver):
    """Shooting method solver for calculation Schrodinger equation."""

    def __init__(self, grid: np.ndarray, v_potential, cb_meff) -> None:

        # Schrodinger equation's parameters
        self.v_potential = v_potential
        self.cb_meff = cb_meff
        # parse grid configuration
        self.grid = grid
        self.delta_z = grid[1] - grid[0]
        # Shooting method parameters for Schrödinger Equation solution
        # Energy step (eV) for initial search. Initial delta_E is 1 meV.
        self.delta_e = 0.5 / 1e3

    def _psi_iteration(self, energy_x0):
        """Use `numpy.nditer` to get iteration solution.

        Args:
            energy_x0: energy to start wavefunction iteration.
        Returns:
            Diverge of psi at infinite x.
        """

        psi = np.zeros(self.grid.shape)
        psi[0] = 0.0
        psi[1] = 1.0
        const_0 = 2 * (self.delta_z / const.hbar)**2
        with np.nditer(psi, flags=['c_index'], op_flags=['writeonly']) as it:
            for x in it:
                if it.index <= 1:
                    continue
                const_1 = 2.0 / (self.cb_meff[it.index - 1] +
                                 self.cb_meff[it.index - 2])
                const_2 = 2.0 / (self.cb_meff[it.index - 1] +
                                 self.cb_meff[it.index])
                x[...] = ((const_0 *
                           (self.v_potential[it.index - 1] - energy_x0) +
                           const_2 + const_1) * psi[it.index - 1] -
                          const_1 * psi[it.index - 2]) / const_2
        self.psi = psi

        return psi[-1]  # psi at inf

    def calc_evals(self,
                   energy_x0=None,
                   max_energy=None,
                   num_band=None,
                   **kwargs):
        """Calculate eigenenergy of any bound states in the chosen potential.

        Args:
            max_energy: shooting eigenenergy that smaller than `max_energy`
            num_band: number of band to shoot eigenenergy.
            energy_x0: minimum energy to start subband search. (Unit in Joules)
            kwargs: argument for `scipy.optimize.root_scalar`
        """

        # find brackets contain eigenvalue
        if energy_x0 is None:
            energy_x0 = np.min(self.v_potential) * 0.9
        if max_energy is None:
            max_energy = np.max(self.v_potential) + 0.1 * (
                np.max(self.v_potential) - np.min(self.v_potential))
        # shooting energy list
        num_shooting = round((max_energy - energy_x0) / self.delta_e)
        energy_list = np.linspace(energy_x0, max_energy, num_shooting)
        psi_list = [self._psi_iteration(energy) for energy in energy_list]
        # check sign change
        shooting_index = np.argwhere(np.diff(np.sign(psi_list)))
        # truncated eigenenergy
        if num_band is not None:
            shooting_index = shooting_index[:num_band]
        # find root in brackets
        shooting_bracket = [
            [energy_list[idx], energy_list[idx + 1]] for idx in shooting_index
        ]
        result_sol = []
        for bracket in shooting_bracket:
            sol = optimize.root_scalar(self._psi_iteration,
                                       bracket=bracket,
                                       **kwargs)
            result_sol.append(sol)
        # acquire eigenenergy
        eig_val = [sol.root for sol in result_sol]

        return eig_val

    def calc_esys(self, **kwargs):
        """Calculate wave function and eigenenergy.

        Args:
            max_energy: shooting eigenenergy that smaller than `max_energy`
            num_band: number of band to shoot eigenenergy.
            energy_x0: minimum energy to start subband search. (Unit in Joules)
            kwargs: argument for `scipy.optimize.root_scalar`
        """

        eig_val = self.calc_evals(**kwargs)
        wave_function = []
        for energy in eig_val:
            self._psi_iteration(energy)
            norm = np.sqrt(np.trapz(self.psi * np.conj(self.psi),
                                    x=self.grid))  # l2-norm
            wave_function.append(self.psi / norm)

        return eig_val, np.asarray(wave_function)


# %%
# # QuickTest
# from matplotlib import pyplot as plt

# grid = np.linspace(0, 10, 100)
# psi = np.zeros(grid.shape)
# v_potential = np.ones(grid.shape) * 10
# # Quantum well
# v_potential[40:61] = 0
# cb_meff = np.ones(grid.shape)
# energy = 1
# solver = SchrodingerShooting(grid, psi, energy, v_potential, cb_meff)
# eig_v = solver.calc_evals(0)
# # %%
# plt.plot(solver.grid, solver.v_potential / 10)
# plt.plot(solver.grid, np.asarray([solver.calc_wavefunction(eig_v[i]) for i in range(3)]).T)
