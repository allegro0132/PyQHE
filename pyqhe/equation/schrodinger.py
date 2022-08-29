# %%
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np
import scipy.linalg as sciLA
from scipy import optimize

import pyqhe.utility.constant as const
from pyqhe.utility.utils import tensor


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


class SchrodingerMatrix(SchrodingerSolver):
    """N-D Schrodinger equation solver based on discrete Laplacian and FDM.

    Args:
        grid: ndarray or list of ndarray
            when solving n-d schrodinger equation, just pass n grid array
    """

    def __init__(self, grid: Union[List[np.ndarray], np.ndarray],
                 v_potential: np.ndarray, cb_meff: np.ndarray) -> None:
        super().__init__()

        if not isinstance(grid, list):  # 1d grid
            grid = [grid]
        self.grid = grid
        self.dim = [grid_axis.shape[0] for grid_axis in self.grid]
        # check matrix dim
        if v_potential.shape == tuple(self.dim) or len(self.dim) == 1:
            self.v_potential = v_potential
        else:
            raise ValueError('The dimension of v_potential is not match')
        if cb_meff.shape == tuple(self.dim) or len(self.dim) == 1:
            self.cb_meff = cb_meff
        else:
            raise ValueError('The dimension of cb_meff is not match.')
        self.beta = 1e31

    def build_kinetic_operator(self, loc):
        """Build 1D time independent Schrodinger equation kinetic operator.

        Args:
            dim: dimension of kinetic operator.
        """
        mat_d = -2 * np.eye(self.dim[loc]) + np.eye(
            self.dim[loc], k=-1) + np.eye(self.dim[loc], k=1)
        return mat_d

    def build_potential_operator(self, loc):
        """Build 1D time independent Schrodinger equation potential operator.

        Args:
            dim: dimension of potential operator.
        """
        return np.diag(self.v_potential.flatten())

    def hamiltonian(self):
        """Construct time independent Schrodinger equation."""
        # construct V and cb_meff matrix
        # discrete laplacian
        k_mat_list = []
        for loc, dim in enumerate(self.dim):
            mat = self.build_kinetic_operator(loc)
            kron_list = [np.eye(idim) for idim in self.dim[:loc]] + [mat] + [
                np.eye(idim) for idim in self.dim[loc + 1:]
            ] + [1]  # auxiliary element for 1d solver
            delta = self.grid[loc][1] - self.grid[loc][0]
            # coeff = -0.5 * const.hbar**2 * self.cb_meff * self.beta**2 / delta**2
            coeff = -0.5 * const.hbar**2 / self.cb_meff / delta**2
            # construct n-d kinetic operator by tensor product
            k_opt = tensor(*kron_list)
            # tensor contraction
            k_opt = np.einsum(k_opt.reshape(self.dim * 2),
                              np.arange(len(self.dim * 2)), coeff,
                              np.arange(len(self.dim)),
                              np.arange(len(self.dim * 2)))
            k_mat_list.append(k_opt.reshape(np.prod(self.dim), np.prod(self.dim)))
        k_mat = np.sum(k_mat_list, axis=0)
        v_mat = np.diag(self.v_potential.flatten())

        return k_mat + v_mat

    def calc_evals(self):
        ham = self.hamiltonian()
        return sciLA.eigh(ham, eigvals_only=True)

    def calc_esys(self):
        ham = self.hamiltonian()
        eig_val, eig_vec = sciLA.eigh(ham)
        # convert psi(phi) to psi(z)
        # coeff = 1 / self.cb_meff / self.beta
        # eig_vec = np.einsum(eig_vec.reshape(self.dim * 2),
        #                     np.arange(len(self.dim * 2)), coeff,
        #                     np.arange(len(self.dim)),
        #                     np.arange(len(self.dim * 2)))
        eig_vec = eig_vec.reshape(np.prod(self.dim), np.prod(self.dim))
        # eig_vec = np.einsum('ij,i->ij', eig_vec, 1 / self.cb_meff / self.beta)
        wave_func = []
        for vec in eig_vec.T:
            # reshape eigenvector to discrete wave function
            vec = vec.reshape(self.dim)
            # normalize
            norm = vec * np.conj(vec)
            for grid in self.grid[::-1]:
                norm = np.trapz(norm, grid)
            wave_func.append(vec / np.sqrt(norm))

        return eig_val, np.array(wave_func)


# %%
# QuickTest
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from matplotlib import cm

    x = np.linspace(-1, 1, 50)
    y = np.linspace(-1, 1, 55)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    x_barrier = (xv <= -0.5) + (xv >= 0.5)
    y_barrier = (yv <= -0.5) + (yv >= 0.5)
    v_potential = np.zeros([50, 55])
    v_potential[x_barrier + y_barrier] = 1  # set barrier
    sol = SchrodingerMatrix([x, y], v_potential, np.ones_like(v_potential) * const.m_e)
    eig_val, wf = sol.calc_esys()
    # %%
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xv,
                           yv,
                           wf[3],
                           cmap=cm.coolwarm,
                           linewidth=0,
                           antialiased=False)

    plt.show()
    # %%
    grid = np.linspace(0, 10, 100)
    psi = np.zeros(grid.shape)
    v_potential = np.ones(grid.shape)
    # Quantum well
    v_potential[40:61] = 0
    cb_meff = np.ones(grid.shape) * const.m_e
    solver = SchrodingerMatrix(grid, v_potential, cb_meff)
    val, vec = solver.calc_esys()
    plt.plot(solver.grid[0], solver.v_potential)
    plt.plot(solver.grid[0], vec[:3].T)
    plt.show()
# %%
