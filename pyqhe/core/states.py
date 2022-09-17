from itertools import combinations
import numbers
import numpy as np
import scipy.special as scipy_spec


def basis(dimensions, n=None):
    """Generates the vector representation of a Fock state.

    Args:
        dimensions : int or list of ints
            Number of Fock states in Hilbert space.  If a list, then the resultant
            object will be a tensor product over spaces with those dimensions.

        n : int or list of ints, optional (default 0 for all dimensions)
            Integer corresponding to desired number state, defaults to 0 for all
            dimensions if omitted.  The shape must match ``dimensions``, e.g. if
            ``dimensions`` is a list, then ``n`` must either be omitted or a list
            of equal length.

    Returns:
        state : ndarray representing the requested number state ``|n>``.

    """
    if isinstance(dimensions, numbers.Integral):
        dimensions = [dimensions]
    if isinstance(n, numbers.Integral):
        n = [n]

    location, size = 0, 1

    for m, dimension in zip(reversed(n), reversed(dimensions)):
        location += m * size
        size *= dimension
    psi = np.zeros(size, dtype=complex)
    psi[location] = 1
    return psi.reshape(-1, 1)


def create_basis(num_orbit):
    """Create arbitrary orthogonal complete basis"""
    return np.eye(int(num_orbit))[:, :, np.newaxis]


class FQHBasis:
    """Class for construct basis for Fraction Quantum Hall system.

    Attributes:
        array_repr: states representation in ndarray
        occupation_repr: states representation in occupation number,
            for fermion, only 0 or 1 at each orbit.
    """

    def __init__(self, num_orbit, num_occupation) -> None:
        self.num_orbit = num_orbit
        self.num_occupation = num_occupation
        occupation_repr = []
        comb = combinations(range(num_orbit), num_occupation)
        for occu_loc in comb:
            occu_repr = np.zeros(num_orbit, dtype=int)
            occu_repr[np.array(occu_loc)] = 1
            occupation_repr.append(occu_repr)
        # convert to numpy ndarray
        self.occupation_repr = np.array(occupation_repr)
        # construct the map between occupation and index
        self.occupation_map = {
            np.array2string(occu_repr): idx
            for idx, occu_repr in enumerate(self.occupation_repr)
        }

    @property
    def dim(self):
        return int(scipy_spec.comb(self.num_orbit, self.num_occupation))

    @property
    def array_repr(self):
        return create_basis(self.dim)

    def lookup_occu_repr(self, occu_repr):
        """Return the indices of occu_repr"""
        occu_repr = np.asarray(occu_repr, dtype=int)
        str_repr = np.array2string(occu_repr)
        idx = self.occupation_map[str_repr]
        # idx = np.argwhere((self.occupation_repr == occu_repr).all(axis=-1))
        return idx


# %%
# Quick test
if __name__ == '__main__':
    basis(5, 2)
    fbasis = FQHBasis(6, 2)
# %%

