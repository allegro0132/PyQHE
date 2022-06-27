import numpy as np
from matplotlib import pyplot as plt

from pyqhe.equation.schrodinger import SchrodingerShooting
from pyqhe.equation.poisson import PoissonODE, PoissonFDM
from pyqhe.utility.fermi import FermiStatistic
from pyqhe.core.structure import Structure1D


class OptimizeResult:
    """Optimize result about self-consistent iteration."""

    def __init__(self) -> None:
        # Storage of grid configure
        self.grid = None
        # Optimizer result
        self.params = None
        # Fermi Statistic
        self.fermi_energy = None
        self.n_states = None
        self.sigma = None
        # electron properties
        self.eig_val = None
        self.wave_function = None
        # electric field properties
        self.v_potential = None
        self.e_field = None

    def plot_quantum_well(self):
        """Plot dressed conduction band of quantum well, and electrons'
        eigenenergy and wave function.
        """

        wave_func_rescale = 0.2
        ax = plt.subplot(1, 1, 1)
        ax.plot(self.grid, self.v_potential, "k")
        # just plot the three lowest eigenenergy
        colors = ['y', 'c', 'm']
        for i, (energy, state) in enumerate(
                zip(self.eig_val[:3], self.wave_function[:3])):
            ax.axhline(energy,
                       0.1,
                       0.9,
                       ls="--",
                       color=colors[i],
                       label=f'E_{i}: {energy:3f}')  # eigenenergy
            # plot rescaled wave function
            ax.plot(self.grid, state * wave_func_rescale + energy, color='b')
        ax.axhline(self.fermi_energy, 0.1, 0.9, color="r", ls="--")
        ax.set_xlabel("Position (nm)")
        ax.set_ylabel("Energy (eV)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.grid(True)

        return ax


class SchrodingerPoisson:
    """Self-consistent Schrodinger Poisson solver
    Args:
        sch_solver: Schrodinger equation solver.
        poi_solver: Poisson equation solver.
        fermi_util: Use Fermi statistic for Fermi level and energy bands' distribution.
    """

    def __init__(self, model: Structure1D, learning_rate=0.5, **kwargs) -> None:
        self.model = model
        # load material's properties
        self.temp = model.temp  # temperature
        self.fi = model.fi  # Band structure's potential
        self.cb_meff = model.cb_meff  # Conduction band effective mass
        self.eps = model.eps  # dielectric constant(multiply eps_0)
        self.doping = model.doping  # doping profile
        # load grid configure
        self.grid = model.universal_grid
        # adjust optimizer
        self.learning_rate = learning_rate
        # load solver
        self.sch_solver = SchrodingerShooting(self.grid, self.fi, self.cb_meff)
        self.poi_solver = PoissonFDM(self.grid, self.doping, self.eps)
        self.fermi_util = FermiStatistic(self.grid, self.cb_meff, self.doping)
        # Cache parameters
        self.eig_val = self.sch_solver.calc_evals()
        self.params = None

    def _calc_net_density(self, n_states, wave_func):
        """Calculate the net charge density."""

        # firstly, convert to areal charge density
        grid_dist = np.diff(self.grid)
        grid_dist = np.append(grid_dist, grid_dist[-1])  # adjust shape
        doping_2d = self.doping * grid_dist
        # Accumulate electron areal density in the subbands
        elec_density = np.zeros_like(doping_2d)
        for i, distri in enumerate(n_states):
            elec_density += distri * wave_func[i] * np.conj(wave_func[i])
        # Let dopants density minus electron density
        return doping_2d - elec_density

    def _iteration(self, params):
        """Perform a single iteration of self-consistent Schrodinger-Poisson
        calculation.

        Args:
            v_potential: optimizer parameters.
        """

        # perform schrodinger solver
        v_potential = self.fi + params
        self.sch_solver.v_potential = v_potential
        eig_val, wave_func = self.sch_solver.calc_esys()
        # calculate energy band distribution
        _, n_states = self.fermi_util.fermilevel(eig_val, wave_func, self.temp)
        # calculate the net charge density
        sigma = self._calc_net_density(n_states, wave_func)
        # perform poisson solver
        self.poi_solver.charge_density = sigma
        self.poi_solver.calc_poisson()
        # return eigenenergy loss
        loss = np.abs(self.eig_val[0] - eig_val[0])
        params = self.poi_solver.v_potential
        self.eig_val = eig_val

        return loss, params

    def self_consistent_minimize(self,
                                 num_iter=10,
                                 learning_rate=0.5,
                                 logging=True):
        """Self consistent optimize parameters `v_potential` to get solution.

        Args:
            learning_rate: learning rate between adjacent iteration.
        """
        if self.params is None:
            self.params = 0  # v_potential
        for _ in range(num_iter):
            # perform a iteration
            loss, temp_params = self._iteration(self.params)
            if logging:
                print(
                    f'Loss: {loss}, energy_0: {self.eig_val[0]}, '
                    f'energy_1: {self.eig_val[1]}, energy_2: {self.eig_val[2]}')
            # self-consistent update params
            # params += learning_rate * temp_params
            self.params = temp_params
        # save optimize result
        res = OptimizeResult()
        res.params = self.params
        res.grid = self.grid
        res.v_potential = self.params + self.fi
        # reclaim convergence result
        self.sch_solver.v_potential = res.v_potential
        res.eig_val, res.wave_function = self.sch_solver.calc_esys()
        res.fermi_energy, res.n_states = self.fermi_util.fermilevel(
            res.eig_val, res.wave_function, self.temp)
        res.sigma = self._calc_net_density(res.n_states, res.wave_function)
        self.poi_solver.charge_density = res.sigma
        self.poi_solver.calc_poisson()
        res.e_field = self.poi_solver.e_field

        return res
