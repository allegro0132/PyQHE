# %%
from typing import Callable
import numpy as np

from pyqhe.core.states import basis, FQHBasis


#%%
def create_hamiltonian_truncated(basis: FQHBasis, potential: Callable = None):
    """Create Laughlin FQH Hamiltonian"""
    ham = []
    # construct hamiltonian row by row
    for loc, occu_repr in enumerate(basis.occupation_repr):
        mat_row = np.zeros(basis.dim)
        target_idx = np.argwhere(occu_repr).flatten()
        # find |...1001...> scheme
        diff_idx = np.diff(target_idx)
        hopping_idx = target_idx[np.nonzero(diff_idx == 3)]
        # deal with the hopping terms
        for h_idx in hopping_idx:
            new_occu_repr = occu_repr.copy()
            new_occu_repr[h_idx:h_idx + 4] = [0, 1, 1, 0]
            # lookup correspond basis
            new_loc = basis.lookup_occu_repr(new_occu_repr)
            mat_row[new_loc] = potential(2, 1)
        # find |...11...> scheme
        nj_nj1_idx = target_idx[np.nonzero(diff_idx == 1)]
        for n1_idx in nj_nj1_idx:
            mat_row[loc] += potential(1, 0)
        # find |...1x1...> scheme
        n2_part0 = np.argwhere(diff_idx == 2).flatten()  # |...101...>
        diff_n1_idx = np.diff(nj_nj1_idx)  # diff through continuous 11 scheme
        n2_part1 = np.argwhere(diff_n1_idx == 1).flatten()  # |...111...>
        nj_nj2_idx = np.concatenate([n2_part0, n2_part1])
        for n2_idx in nj_nj2_idx:
            mat_row[loc] += potential(2, 0)
        # find |...1xx1...> scheme
        nj_nj3_idx = []
        for idx in target_idx:
            if idx + 3 in target_idx:
                nj_nj3_idx.append(idx)
        for n3_idx in nj_nj3_idx:
            mat_row[loc] += potential(3, 0)

        ham.append(mat_row)
    ham = np.stack(ham)
    # add hermitian conjugate
    return ham + np.conj(ham).T


def create_hamiltonian(basis: FQHBasis, potential: Callable = None):
    """Create Laughlin FQH Hamiltonian"""
    ham = []
    num_orbit = basis.num_orbit
    # construct hamiltonian row by row
    for loc, occu_repr in enumerate(basis.occupation_repr):
        mat_row = np.zeros(basis.dim)
        target_idx = np.argwhere(occu_repr).flatten()
        for j_idx in target_idx:  # nonzero terms correspond to c_j
            # get the range of m
            m_idx_list = range(-j_idx, num_orbit - j_idx)
            for m_idx in m_idx_list:  # iteration through index m
                # note here, j+k and j+m+k should in range [0, num_orbit)
                k_idx_list = range(
                    abs(m_idx) + 1,
                    min(num_orbit - j_idx, num_orbit - j_idx - m_idx))
                for k_idx in k_idx_list:  # iteration through index k
                    j2 = j_idx + m_idx + k_idx  # c_{j+k+m}
                    j3 = j_idx + k_idx  # c^+_{j+k}
                    j4 = j_idx + m_idx  # c^+_{j+m}
                    # check scheme
                    if (j2 in target_idx) and (j3 not in target_idx) and (
                            j4 not in target_idx):
                        # create scheme after hopping
                        new_occu_repr = occu_repr.copy()
                        new_occu_repr[np.array([j_idx, j2, j3,
                                                j4])] = [0, 0, 1, 1]
                        # lookup correspond basis
                        new_loc = basis.lookup_occu_repr(new_occu_repr)
                        mat_row[new_loc] = potential(k_idx, m_idx)
                    elif (m_idx == 0) and (j2 in target_idx):
                        # number operator
                        mat_row[loc] = potential(k_idx, m_idx)
        ham.append(mat_row)
    ham = np.stack(ham)
    return ham


# %%
# Quick test
if __name__ == '__main__':
    fbasis = FQHBasis(6, 2)
    ham = create_hamiltonian(fbasis, lambda k, m: 1)
# %%
