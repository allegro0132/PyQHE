from typing import Callable
import qutip as qt
import numpy as np
from scipy.constants import hbar


class TightBinding:
    """Second quantization form of the tight-binding method.
    """

    def __init__(self) -> None:
        self.dim = []

    def _gen_reciprocal_lattice(self):
        self.k_vec = [1 / self.dim[0], 1 / self.dim[1]]

    def hopping(self, a: list, b: list, alpha, q, vpotential: Callable = None, hopping_1d=False):
        """Hopping from point A to B. Calculate phase in finite different mothod.
        """
        if not vpotential:
            phase = 0
        else:
            phase = (alpha * (b[0] - a[0]) *  # hopping at y axis
                     (vpotential(alpha * np.flip(b))[1] +
                      vpotential(alpha * np.flip(a))[1]) / 2 + alpha *
                     (b[1] - a[1]) * (vpotential(alpha * np.flip(b))[0] +
                                      vpotential(alpha * np.flip(a))[0]) / 2)
        # Peierls substitution
        hbar = 1
        add_phase = np.exp(q / hbar * phase * 2j * np.pi)
        if hopping_1d:
            return add_phase * (qt.basis(self.dim[1], b[1]) *
                                qt.basis(self.dim[1], a[1]).dag())
        return add_phase * (qt.basis(self.dim, b) * qt.basis(self.dim, a).dag())
