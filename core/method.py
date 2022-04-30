from typing import Callable
import qutip as qt
import numpy as np
from scipy.constants import hbar


class TightBinding:
    """Second quantization form of the tight-binding method.
    """

    def __init__(self) -> None:
        pass

    def hopping(self, a: list, b: list, alpha, q, vpotential: Callable = None):
        """Hopping from point A to B. Calculate phase in finite different mothod.
        """
        if not vpotential:
            phase = 0
        else:
            phase = ((b[0] - a[0]) * (vpotential(alpha * np.flip(b))[1] +
                                      vpotential(alpha * np.flip(a))[1]) / 2 +
                     (b[1] - a[1]) * (vpotential(alpha * np.flip(b))[0] +
                                      vpotential(alpha * np.flip(a))[0]) / 2)
        # Peierls substitution
        add_phase = np.exp(q / hbar * phase * 1j)
        return add_phase * (qt.basis(self.dim, b) * qt.basis(self.dim, a).dag())
