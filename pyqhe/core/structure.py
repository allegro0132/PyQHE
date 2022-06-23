# %%
from abc import abstractmethod, ABC
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import qutip

import pyqhe.utility.constant as const


class SplineStorage:
    """Flexible storage for physical parameters, interpolate since equation solver
    using arbitrary grid.

    Args:

    """

    def __init__(self, params: np.ndarray, grid: np.ndarray,
                 method: str = 'intep1d', **kwargs) -> None:

        self.params = params
        self.grid = grid
        self.method = method
        if method == 'interp1d':
            kind = kwargs.get(kind, 'linear')
            self.func = interp1d(self.grid,
                                 self.params,
                                 kind=kind,
                                 bounds_error=False,
                                 fill_value=0.)
        else:
            raise ValueError('Please choose method in `interp1d`.')

    def __call__(self, grid_point):
        if not isinstance(grid_point, (np.ndarray, list)):
            raise ValueError('SplineStorage does not support the grid point.')
        return self.func(grid_point)


class Layer:
    """Class for AlAs/GaAs heterostructure layer.

    Args:
        thickness: The thickness of layer
        alloy_ratio: a ratio in [0, 1]
        name: name of the layer
    """

    def __init__(self, thickness: float, alloy_ratio: float, name: str = 'layer') -> None:

        self.name = name
        self.loc = None  # The layer locate at [a, b)
        # define parameters
        self.thickness = thickness
        self.alloy_ratio = alloy_ratio
        # Physical parameters
        self.fi = None
        self.eps = None
        self.cb_meff = None
        # Physical properties database
        self.GaAs = {
            'm_e':
                0.067,  #conduction band effective mass (relative to electron mass)
            'm_hh':
                0.45,  #heavy hole band effective mass (used by aestimo_numpy_h)
            'm_lh':
                0.087,  #light hole band effective mass (used by aetsimo_numpy_h)
            'epsilonStatic': 12.90,  #dielectric constant
            'Eg': 1.4223,  #1.42 # (ev) band gap
            'Ep':
                28.8,  # (eV) k.p matrix element (used for non-parabolicity calculation (Vurgaftman2001)
            'F':
                -1.94,  # Kane parameter (used for non-parabolicity calculation (Vurgaftman2001)
            'Band_offset':
                0.65,  # conduction band/valence band offset ratio for GaAs - AlGaAs heterojunctions
            'm_e_alpha':
                5.3782e18,  # conduction band non-parabolicity variable for linear relation (Nelson approach)
            # Valence band constants
            'delta': 0.28,  # (eV) Spin split-off energy gap
            # below used by aestimo_numpy_h
            'GA1': 6.8,  #luttinger parameter
            'GA2': 1.9,  #luttinger parameter
            'GA3': 2.73,  #luttinger parameter
            'C11': 11.879,  # (GPa) Elastic Constants
            'C12': 5.376,  # (GPa) Elastic Constants
            'a0': 5.6533,  # (A)Lattice constant
            'Ac': -7.17,  # (eV) deformation potentials (Van de Walle formalism)
            'Av': 1.16,  # (eV) deformation potentials (Van de Walle formalism)
            'B':
                -1.7,  # (eV) shear deformation potential (Van de Walle formalism)
            'TAUN0': 0.1E-7,  # Electron SRH life time
            'TAUP0': 0.1E-7,  # Hole SRH life time
            'mun0': 0.1,  # Electron Mobility in m2/V-s
            'mup0': 0.02,  # Electron Mobility in m2/V-s
            'Cn0':
                2.8e-31,  # generation recombination model parameters [cm**6/s]
            'Cp0':
                2.8e-32,  # generation recombination model parameters [cm**6/s]
            'BETAN':
                2.0,  # Parameter in calculatation of the Field Dependant Mobility
            'BETAP':
                1.0,  # Parameter in calculatation of the Field Dependant Mobility
            'VSATN': 3e5,  # Saturation Velocity of Electrons
            'VSATP': 6e5,  # Saturation Velocity of Holes
            'AVb_E':
                -6.92  #Average Valence Band Energy or the absolute energy level
        }

        self.AlAs = {
            'm_e': 0.15,
            'm_hh': 0.51,
            'm_lh': 0.18,
            'epsilonStatic': 10.06,
            'Eg': 3.0,  #2.980,
            'Ep': 21.1,
            'F': -0.48,
            'Band_offset': 0.53,
            'm_e_alpha': 0.0,
            'GA1': 3.45,
            'GA2': 0.68,
            'GA3': 1.29,
            'C11': 11.879,
            'C12': 5.376,
            'a0': 5.66,
            'Ac': -5.64,
            'Av': 2.47,
            'B': -1.5,
            'delta': 0.28,
            'TAUN0': 0.1E-6,
            'TAUP0': 0.1E-6,
            'mun0': 0.15,
            'mup0': 0.1,
            'Cn0':
                2.8e-31,  # generation recombination model parameters [cm**6/s]
            'Cp0':
                2.8e-32,  # generation recombination model parameters [cm**6/s]
            'BETAN': 2.0,
            'BETAP':
                1.0,  # Parameter in calculatation of the Field Dependant Mobility
            'VSATN': 3e5,  # Saturation Velocity of Electrons
            'VSATP': 6e5,  # Saturation Velocity of Holes
            'AVb_E':
                -7.49  #Average Valence Band Energy or the absolute energy level
        }

        # Alloying properties
        self.alloy = {
            'Bowing_param': 0.37,
            'Band_offset': 0.65,
            'm_e_alpha': 5.3782e18,
            'delta_bowing_param': 0.0,
            'a0_sub': 5.6533,
            'Material1': 'AlAs',
            'Material2': 'GaAs',
            'TAUN0': 0.1E-6,
            'TAUP0': 0.1E-6,
            'mun0': 0.15,
            'mup0': 0.1,
            'Cn0': 2.8e-31,  # generation recombination model parameters [cm**6/s]
            'Cp0': 2.8e-32,  # generation recombination model parameters [cm**6/s]
            'BETAN': 2.0,
            'BETAP': 1.0,  # Parameter in calculatation of the Field Dependant Mobility
            'VSATN': 3e5,  # Saturation Velocity of Electrons
            'VSATP': 6e5,  # Saturation Velocity of Holes
            'AVb_E': -2.1  #Average Valence Band Energy or the absolute energy level
        }
        # Check alloying radio, and use alloying function
        if self.alloy_ratio < 0 or self.alloy_ratio > 1:
            raise ValueError('Incorrect allot ratio.')
        else:
            self._alloying()

    def _alloying(self):
        """Calculate ternary material's properties."""

        # Band structure potential, for electron unit in Joule
        self.fi = self.alloy["Band_offset"] * (
            self.alloy_ratio * self.AlAs["Eg"] +
            (1 - self.alloy_ratio) * self.GaAs["Eg"] -
            self.alloy["Bowing_param"] * self.alloy_ratio *
            (1 - self.alloy_ratio)) * const.q
        # dielectric constant
        self.eps = (
            self.alloy_ratio * self.AlAs["epsilonStatic"] +
            (1 - self.alloy_ratio) * self.GaAs["epsilonStatic"]) * const.eps0
        # conduction band effective mass
        self.cb_meff = (self.alloy_ratio * self.AlAs['m_e'] +
                        (1 - self.alloy_ratio) * self.GaAs['m_e']) * const.m_e
        # non-parabolicity constant.


class Structure1D:
    """Class for modeling 1d material structure.
    """

    def __init__(self, spline_storage=False) -> None:
        # Initialize Layers
        layer0 = Layer(10, 0, name='barrier')
        layer1 = Layer(5, 1, name='quantum_wall')
        layer2 = Layer(10, 0, name='barrier')
        self.layers = [layer0, layer1, layer2]
        # Structure's parameter
        self.stack = None
        self.stack_thick = None
        self.bound_locs = None
        self.layer_arrange()
        # Generate a 'universal grid'. Cache data in ndarray with the same
        # dimension as 'universal grid'.
        self.universal_grid = None
        # Structure's parameters
        self.eps = None
        self.doping = None
        # Schrodinger equation's parameters
        self.psi = None
        # Poisson equation's parameters
        self.potential = None
        self.e_field = None
        self._prepare_structure_stroage(spline_storage=spline_storage)

    def layer_arrange(self):
        """Arrange every layer in `self.layers` and build the stack.

        Args:

        Returns:
            bound_locs: Highlight the boundary's location.
        """
        loc = 0
        bound_locs = [0.]
        for layer in self.layers:
            layer.loc = [loc, loc + layer.thickness]
            loc += layer.thickness
            bound_locs.append(loc)
        self.stack_thick = loc
        self.stack = [layer.loc for layer in self.layers]
        self.bound_locs = bound_locs

        return bound_locs

    def generate_grid(self, num_gridpoint, type='fdm'):
        """Generate a set of grid for solving differential equation numerically.
        """

        return np.linspace(self.bound_locs[0], self.bound_locs[-1], num_gridpoint)

    def _prepare_structure_stroage(self, dx=1, spline_storage=False):
        """Initialize structure's parameters in `SplineStorage`.

        Args:
            dx: the minimum gap between grid point , unit in nanometer.
        """

        num_gridpoint = round(self.stack_thick / dx)
        # Generate a identity grid
        self.universal_grid = self.generate_grid(num_gridpoint)
        eps = np.zeros(self.universal_grid.shape)
        for layer in self.layers:
            layer_mask = (self.universal_grid >= layer.loc[0]) * (self.universal_grid < layer.loc[1])
            eps[layer_mask] = layer.eps
        if spline_storage:
            self.eps = SplineStorage(eps, self.universal_grid)
        else:
            self.eps = eps
# %%
# QuickTest
stack0 = Structure1D()

# %%