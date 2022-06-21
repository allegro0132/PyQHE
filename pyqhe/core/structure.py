from abc import abstractmethod, ABC
import numpy as np
import qutip


class Layer:
    """Class for AlAs/GaAs heterostructure layer.
    """

    def __init__(self, thickness, alloy_x, name: str = 'layer') -> None:

        self.name = name
        # define parameters
        self.thickness = thickness
        self.alloy_x = alloy_x
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
        # Material's properties
        self.dielectric = None
        self.effect_mass = None

    def alloying(self):
        """Calculate ternary material's properties."""


class Structure1D:
    """Class for modeling 1d material structure.
    """

    def __init__(self) -> None:
        # Initialize Layers
        layer0 = Layer(100, 0, name='barrier')
        layer1 = Layer(50, 1, name='quantum_wall')
        layer2 = Layer(100, 0, name='barrier')
        self.layers = [layer0, layer1, layer2]
        # Structure's parameter
        self.stack = None
        # Generate a 'universal grid'. Cache data in ndarray with the same
        # dimension as 'universal grid'.
        self.universal_grid = None
        # Schrodinger equation's parameters
        self.psi = None
        # Poisson equation's parameters
        self.potential = None
        self.e_field = None

    def arrange_stack(self):


    def generate_grid(self):
        """Generate a set of grid for solving differential equation numerically.
        """


