# %%
from typing import Callable
import numpy as np

from pyqhe.core.states import basis, FQHBasis
#%%
def create_hamiltonian(basis: FQHBasis, potential: Callable = None):
    """Create Laughlin FQH Hamiltonian"""
    # temporary coefficient
    v_10 = 1
    v_20 = 2
    v_30 = 3
    v_21 = 21
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
            new_occu_repr[h_idx:h_idx+4] = [0, 1, 1, 0]
            # lookup correspond basis
            new_loc = basis.lookup_occu_repr(new_occu_repr)
            mat_row[new_loc] = v_21
        # find |...11...> scheme
        nj_nj1_idx = target_idx[np.nonzero(diff_idx == 1)]
        for n1_idx in nj_nj1_idx:
            mat_row[loc] += v_10
        # find |...1x1...> scheme
        n2_part0 = np.argwhere(diff_idx == 2).flatten()  # |...101...>
        diff_n1_idx = np.diff(nj_nj1_idx)  # diff through continuous 11 scheme
        n2_part1 = np.argwhere(diff_n1_idx == 1).flatten()  # |...111...>
        nj_nj2_idx = np.concatenate([n2_part0, n2_part1])
        for n2_idx in nj_nj2_idx:
            mat_row[loc] += v_20
        # find |...1xx1...> scheme
        nj_nj3_idx = []
        for idx in target_idx:
            if idx + 3 in target_idx:
                nj_nj3_idx.append(idx)
        for n3_idx in nj_nj3_idx:
            mat_row[loc] += v_30

        ham.append(mat_row)
    ham = np.stack(ham)
    # add hermitian conjugate
    return ham + np.conj(ham).T
# %%
# Quick test
if __name__ == '__main__':
    fbasis = FQHBasis(24, 8)
    ham = create_hamiltonian(fbasis)
    print(ham)
# %%
