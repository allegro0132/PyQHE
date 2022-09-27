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
l_2 = 5.477
list_q = np.linspace(0, 0.5, 30)
list_phi = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 30)
# list_gm = [gm_tensor(q, 0) for q in list_q]
list_gm = [gm_tensor(0.18, phi) for phi in list_phi]
# %%
trivial_states = []
for gm in list_gm:
    # gm = list_gm[0]
    pot = lambda idx_k, idx_m: potential_interaction(idx_k, idx_m, gm, l_2)
    # construct Hamiltonian
    ham = create_hamiltonian(fbasis, pot)
    # diagonalize by LAPACK
    eig_val, eig_vec = scipyLA.eigh(ham)
    # storage of ground states
    trivial_states.append(eig_vec[:, 0])

trivial_states = np.stack(trivial_states)
np.save('trivial_states_phi_5.477', trivial_states)
# %%
trivial_states = np.load('trivial_states_q_5.477.npy')
# chosen initial state Q=0
initial_state = trivial_states[0]
# construct quench Hamiltonian for Q=0.18
pot = lambda idx_k, idx_m: potential_interaction(idx_k, idx_m, gm_tensor(0.18, 0), l_2)
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
t_list = np.linspace(0, 20, 200)
for t in t_list:
    psi = np.zeros_like(eig_vec[:, 0])
    for idx, val in enumerate(eig_val):
        vec = eig_vec[:, idx]
        psi += np.exp(-1j * val * t) * (np.conj(vec) @ initial_state) * vec
    psi_t.append(psi)

# %%
quench_idx = np.array([find_max_overlap(psi, trivial_states)[0] for psi in psi_t])
quench_q = list_q[quench_idx]
# quench_params = list_phi[quench_idx]
plt.plot(t_list, quench_q)
# %%
from scipy.optimize import curve_fit
# fit curve
def fit_q(t, a, eg, eps, phase):
    return 2.0 * a * np.sin(0.5 * eg * t + phase) + eps
popt, pcov = curve_fit(fit_q, t_list[:100], quench_q[:100])
plt.plot(t_list, fit_q(t_list, *popt), color='tab:orange', linestyle='--')
plt.plot(t_list, quench_q)
# %%
def fit_phi(t, eg, eps):
    return 0.5 * eg * t + eps
popt_1, pcov_1 = curve_fit(fit_phi, t_list[:10], -quench_params[:10] / np.pi)
plt.plot(t_list[:30], fit_phi(t_list[:30], *popt_1), color='tab:orange', linestyle='--')
plt.plot(t_list, -quench_params / np.pi)
# %%
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(t_list, quench_q, '.-', label='N = 5')
ax1.plot(t_list, fit_q(t_list, *popt), color='tab:orange', linestyle='--', label='Bimetric fit')
ax1.set_ylabel(r'$\tildeQ$')
ax1.legend(loc=1)

ax2.plot(t_list, -quench_params / np.pi, '.-')
ax2.set_xlabel('time t')
ax2.set_ylabel(r'$\tilde{\phi}/\pi$')

plt.show()
# %%
