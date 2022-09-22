# %%
import numpy as np
import scipy

from pyqhe.core.states import basis, FQHBasis


# %%
def gm_tensor(q, phi):
    """The electron band mass tensor in terms of real number Q and phi.
    """
    mat = np.array(
        [[np.cosh(q) + np.cos(phi) * np.sinh(q),
          np.sin(phi) * np.sinh(q)],
         [np.sin(phi) * np.sinh(q),
          np.cosh(q) - np.cos(phi) * np.sinh(q)]])
    return mat


def potential_interaction(idx_k, idx_m, tensor_g, l_2):
    """The interaction potential of matrix elements"""
    g_11 = tensor_g[0, 0]
    g_12 = tensor_g[0, 1]
    return (idx_k**2 + idx_m**2) * np.exp(
        -2.0 * np.pi**2 *
        (idx_k**2 + idx_m**2 - 2j * idx_k * idx_m * g_12) / g_11 / l_2**2)
