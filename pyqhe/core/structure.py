from abc import abstractmethod, ABC
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import qutip


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
        alloy_x: ternary alloying
        name: name of the layer
    """

    def __init__(self, thickness, alloy_ratio, name: str = 'layer') -> None:

        self.name = name
        self.loc = None  # The layer locate at [a, b)
        # define parameters
        self.thickness = thickness
        self.alloy_ratio = alloy_ratio
        # Physical database
        self.GaAs = {
            'dielectric': 13.1,
            'conduction_bands': {
                'Gamma': {
                    'bandgap': 1.422333,
                    'mass': 0.067
                },
                'L': {
                    'bandgap': 1.707,    # nextnano3 tutorial
                    'mass_l': 1.9,
                    'mass_t': 0.0754
                },
                'X': {
                    'bandgap': 1.899,    # nextnano3 tutorial
                    'mass_l': 1.3,
                    'mass_t': 0.23
                }
            },
            'valence_bands': {
                'bandoffset': 1.346 + -2.882,
                'HH': {'mass': 0.480},   # Greg Snider
                'LH': {'mass': 0.082},   # Greg Snider
                'SO': {'mass': 0.172}
            }
        }
        self.AlAs = {
            'dielectric': 10.1,
            'conduction_bands': {
                'Gamma': {
                    'bandgap': 2.972222,
                    'mass': 0.150333
                },
                'L': {
                    'bandgap': 2.352,    # nextnano3 tutorial
                    'mass_l': 1.32,
                    'mass_t': 0.15
                },
                'X': {
                    'bandgap': 2.164,    # nextnano3 tutorial
                    'mass_l': 0.97,
                    'mass_t': 0.22
                }
            },
            'valence_bands': {
                'bandoffset': 0.8874444 + -2.882,
                'HH': {'mass': 0.51},   # Greg Snider
                'LH': {'mass': 0.088666},   # Greg Snider
                'SO': {'mass': 0.28}
            }
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
        # Material's properties
        self.dielectric = None
        self.effect_mass = None

    def alloying(self):
        """Calculate ternary material's properties."""


class Structure1D:
    """Class for modeling 1d material structure.
    """

    def __init__(self, spline_storage=False) -> None:
        # Initialize Layers
        layer0 = Layer(100, 0, name='barrier')
        layer1 = Layer(50, 1, name='quantum_wall')
        layer2 = Layer(100, 0, name='barrier')
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
            bound_locs.append(loc + layer.thickness)
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
        eps = np.zeros(self.universal_grid)
        for layer in self.layers:
            layer_mask = (self.universal_grid >= layer.loc[0]) * (self.universal_grid < layer.loc[1])
            eps[layer_mask] = layer.dielectric
        if spline_storage:
            self.eps = SplineStorage(eps, self.universal_grid)
        else:
            self.eps = eps
