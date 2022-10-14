# %%
import numpy as np
import scipy.linalg as scipyLA
import matplotlib.pyplot as plt

from pyqhe.core.states import FQHBasis
from pyqhe.fqhe.hamiltonian import create_hamiltonian, create_hamiltonian_truncated
from pyqhe.fqhe.pseudo import gm_tensor, potential_interaction, test_potential

np.set_printoptions(threshold=np.inf)

# %%
# test FQHBasis
fbasis = FQHBasis(10, 4)
print(fbasis.occupation_repr)

# %%
# Construct Hamiltonian
l_2 = 5.77

g_m = gm_tensor(0.2, 0.0)
pot = lambda idx_k, idx_m: potential_interaction(idx_k, idx_m, g_m, l_2)
ham = create_hamiltonian_truncated(fbasis, pot)
print(ham)

# %%
# Print wave function
# diagonalize by LAPACK
eig_val, eig_vec = scipyLA.eigh(ham)
# ground states
wave_function = eig_vec[:, 0]
print(wave_function)
# tidy up small value
tidy_up = np.abs(wave_function) < 1e-5
wf_norm = wave_function.copy()
wf_norm[tidy_up] = 0
# normalize
idx = np.nonzero(~tidy_up)
pr_idx = np.argsort(-np.abs(wf_norm[idx]))
minimal_val = wf_norm[idx][pr_idx[-1]]
wf_norm = wf_norm / minimal_val
print(pr_idx.shape)
np.array(np.round(wf_norm[idx][pr_idx]), dtype=int)
# %%
fbasis.occupation_repr[idx][pr_idx]
# %%
eig_val
# %%
