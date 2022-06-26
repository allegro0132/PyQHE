import numpy as np
from scipy import optimize

import pyqhe.utility.constant as const


def calc_meff_state(wave_function, cb_meff):
    """Calculate subband effective mass."""
    wave_function = np.asarray(wave_function)
    return 1.0 / np.sum(wave_function**2 / cb_meff, axis=1)


class FermiStatistic:
    """Class for Fermi-Dirac statistic and Fermi level.

    Args:
        e_state:
        wavefunction:
    """

    def __init__(
        self,
        grid: np.ndarray,
        cb_meff,
        doping: np.ndarray,
    ) -> None:
        self.doping = doping
        self.grid = grid
        self.cb_meff = cb_meff
        # Calculate 2d doping density
        # integrate 3-d density along axis z
        # in discrete method, just sum over r'n_3d[i] * (grid[i + 1] - grid[i])'
        self.doping_2d = np.trapz(doping, grid)
        # Cache parameters
        self.fermi_energy = None
        self.n_states = None
        self.meff_state = None

    def integral_fermi_dirac(self, energy, fermi_energy, temp):
        """integral of Fermi Dirac Equation for energy independent density of states.
        Ei [meV], Ef [meV], T [K]"""

        return np.log(
            np.exp((fermi_energy - energy) /
                   (const.kb * temp)) + 1) * const.kb * temp

    def fermilevel_0k(self, eig_val, wave_function):
        """Calculate Fermi level at 0 K."""

        self.meff_state = calc_meff_state(wave_function, self.cb_meff)
        # list all fermi energy candidate
        candidate_fermi_energy = []
        for i, _ in enumerate(eig_val):
            accumulate_energy = np.sum(eig_val[:i + 1])
            candidate_fermi_energy.append(
                (self.doping_2d * const.hbar**2 * np.pi / self.meff_state[i] + accumulate_energy) / (i + 1))
        # check true Fermi energy
        fermi_idx = np.argwhere(
            (np.asarray(candidate_fermi_energy) - eig_val) < 0)
        if fermi_idx.size == 0:
            raise ValueError(
                'All energy levels are processed, but no Fermi energy exists.')
        candidate_fermi_energy = np.asarray(candidate_fermi_energy)
        fermi_energy = candidate_fermi_energy[fermi_idx[0] - 1]
        # Calculate populations of energy levels.
        n_state = []
        for i, eig_v in enumerate(eig_val):
            n_sqrt = (fermi_energy - eig_v) * self.meff_state[i] / (
                const.hbar**2 * np.pi)
            if n_sqrt < 0:
                n_sqrt = 0
            n_state.append(n_sqrt**2)

        return fermi_energy, n_state

    def fermilevel(self, eig_val, wave_function, temp, **kwargs):
        """Find Fermi level at selected temperature."""

        # root of the function related to fermi energy at finite temperature.
        def func(f_energy):
            dist = [
                csb_meff * self.integral_fermi_dirac(eig_v, f_energy, temp)
                for eig_v, csb_meff in zip(eig_val, self.meff_state)
            ]
            return self.doping_2d - np.sum(dist) / (const.hbar**2 * np.pi)

        f_energy_0k, _ = self.fermilevel_0k(eig_val, wave_function)
        # sol = optimize.root_scalar(func, x0=f_energy_0k, method='halley', **kwargs)

        # self.fermi_energy = sol.root
        self.fermi_energy = f_energy_0k  # it's hard to converge, just use f_0k now.
        # Calculate populations of energy levels

        self.n_states = [
            self.integral_fermi_dirac(energy, self.fermi_energy, temp) *
            csb_meff / (const.hbar**2 * np.pi)
            for energy, csb_meff in zip(eig_val, self.meff_state)
        ]

        return self.fermi_energy, self.n_states
