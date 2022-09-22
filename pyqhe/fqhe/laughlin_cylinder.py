# %%
import os
import numpy as np
import scipy.linalg as scipyLA
import matplotlib.pyplot as plt

from pyqhe.core.states import FQHBasis
from pyqhe.fqhe.hamiltonian import create_hamiltonian, create_hamiltonian_truncated
from pyqhe.fqhe.pseudo import gm_tensor, potential_interaction

# os.chdir(r'c:\Users\wangz\Documents\Sources\PyQHE\pyqhe\fqhe')

fbasis = FQHBasis(15, 5)  # N=5

# Generate set of trivial ground state
l_2 = 6.245
list_q = np.linspace(0, 0.6, 20)
list_gm = [gm_tensor(q, 0) for q in list_q]
# %%
trivial_states = []
for gm in list_gm:
    # gm = list_gm[0]
    pot = lambda idx_k, idx_m: potential_interaction(idx_k, idx_m, gm, l_2)
    # construct Hamiltonian
    ham = create_hamiltonian(fbasis, pot)
    # diagonalize by LAPACK
    eig_val, eig_vec = scipyLA.eigh(ham)
    # find unique ground state with zero energy
    idx = np.argmin(np.abs(eig_val))
    # storage of ground states
    trivial_states.append(eig_vec[:, 0])

trivial_states = np.stack(trivial_states)
np.save('trivial_states', trivial_states)
# %%
trivial_states = np.load('trivial_states.npy')
# chosen initial state Q=0
initial_state = trivial_states[0]
# construct quench Hamiltonian for Q=0.26
pot = lambda idx_k, idx_m: potential_interaction(idx_k, idx_m, gm_tensor(0.26, 0), l_2)
ham_quench = create_hamiltonian(fbasis, pot)
# time evolution
# obtain all eigenenergies and eigenvectors
eig_val, eig_vec = scipyLA.eigh(ham_quench)
# %%
def find_max_overlap(psi, trivial_states):
    overlap = np.array([np.abs(np.conj(state) @ psi) for state in trivial_states])
    idx = np.argmax(overlap)
    return idx, trivial_states[idx]

psi_t = []
t_list = np.linspace(0, 10, 50)
for t in t_list:
    psi = np.zeros_like(eig_vec[:, 0])
    for idx, val in enumerate(eig_val):
        vec = eig_vec[:, idx]
        psi += np.exp(-1j * val * t) * (np.conj(vec) @ initial_state) * vec
    psi_t.append(psi)

# %%
quench_idx = np.array([find_max_overlap(psi, trivial_states)[0] for psi in psi_t])
quench_q = list_q[quench_idx]

# %%
