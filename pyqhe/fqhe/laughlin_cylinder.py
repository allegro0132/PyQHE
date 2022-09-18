# %%
import numpy as np
import scipy.linalg as scipyLA

from pyqhe.core.states import FQHBasis
from pyqhe.fqhe.hamiltonian import create_hamiltonian

# %%
fbasis = FQHBasis(12, 4)
ham = create_hamiltonian(fbasis)
# %%
eig_val, eig_vec = scipyLA.eigh(ham)
# %%
