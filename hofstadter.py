# %%
import qutip as qt
import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import fft2, fftshift, fftfreq

from core.method import TightBinding


# %%
# create magnetic class with gauge
class Magnetic:
    """The demo codes only support magnetic field B along the z-axis.
    """

    def __init__(self, b) -> None:
        self.magnetic_field = np.asarray(b)
        self.vpotential = 0

    def landau_gauge(self, axis='y'):
        """Landau gauge.
        Args:
            axis(str): ``x`` or ``y``.
        """

        def func_y(position: np.ndarray):
            return np.array([0, self.magnetic_field[2] * position[0], 0])

        def func_x(position: np.ndarray):
            return np.array([-self.magnetic_field[2] * position[1], 0, 0])

        if axis == 'x':
            self.vpotential = func_x
        else:
            self.vpotential = func_y

    def symmetric_gauge(self):
        """Symmetric gauge."""

        def func_a(position: np.ndarray):
            return 0.5 * np.array([
                -self.magnetic_field[2] * position[1],
                self.magnetic_field[2] * position[0], 0
            ])

        self.vpotential = func_a
# %%
mag = Magnetic([0,0,1])
mag.landau_gauge('x')
lattice = np.reshape(np.arange(30), (10,3))
[mag.vpotential(point) for point in lattice]

# %%
class Hofstadter(TightBinding):
    """Calculate a lattice model.
    """

    def __init__(self, num_x, num_y, eps, t, alpha, q, magnetic=None) -> None:
        super().__init__()
        self.dim = [num_y, num_x]
        self.t = t
        self.alpha = alpha
        self.q = q
        self.vpotential = magnetic.vpotential

        self.eig_k = 0
        self.hamiltonian_k = 0
        # initialize a Hilbert Space in real space
        self.hamiltonian = eps / 2 * qt.qeye(self.dim)
        # present lattice in fock basis
        # forward hopping
        for n in range(num_y):
            for m in range(num_x):
                if m + n == num_x + num_y - 2:
                    continue
                elif m == num_x - 1 and n < num_y - 1:
                    self.hamiltonian += -t * self.hopping(
                        [n, m], [n + 1, m], alpha, q, magnetic.vpotential)
                elif n == num_y - 1 and m < num_x - 1:
                    self.hamiltonian += -t * self.hopping(
                        [n, m], [n, m + 1], alpha, q, magnetic.vpotential)
                else:
                    self.hamiltonian += -t * self.hopping(
                        [n, m], [n, m + 1], alpha, q, magnetic.vpotential)
                    self.hamiltonian += -t * self.hopping(
                        [n, m], [n + 1, m], alpha, q, magnetic.vpotential)
        # conjugate hopping
        self.hamiltonian += self.hamiltonian.dag()

    def add_periodic_bound(self, mode='torus'):
        """Add periodic boundary condition.
        Args:
            mode(str): torus, cylinder
        """
        for n in range(self.dim[0]):
            periodic_term = -self.t * self.hopping([n, self.dim[1] - 1], [n, 0],
                                                   self.alpha, self.q,
                                                   self.vpotential)
            self.hamiltonian += periodic_term + periodic_term.dag()
        if mode == 'torus':
            for m in range(self.dim[1]):
                periodic_term = -self.t * self.hopping(
                    [self.dim[0] - 1, m], [0, m], self.alpha, self.q,
                    self.vpotential)
                self.hamiltonian += periodic_term + periodic_term.dag()

    def transform_to_momentum_space(self):
        """2-d Discrete Fourier Transfrom."""
        # related k points after DFT
        k_y = fftfreq(self.dim[0], 1)
        k_x = fftfreq(self.dim[1], 1)
        # define unitary matrix for Fourier Transform
        dft_matrix = np.zeros(self.hamiltonian.shape, complex)
        for n in range(self.dim[0]):
            for m in range(self.dim[1]):
                dft_matrix[n * m + m][:] = np.exp(
                    -2j * np.pi * m * np.tile(k_x, self.dim[0])) * np.exp(
                        2j * np.pi * n * np.repeat(k_y, self.dim[1]))
        dft_matrix = dft_matrix / np.sqrt(len(dft_matrix))  # normalize
        self.hamiltonian_k = dft_matrix.conjugate().T @ self.hamiltonian.full(
        ) @ dft_matrix

    def calc_band_dispersion(self):
        """Calculate eigenenergy and eigenvector in momentum space.
        """
        eig_k, eigv_k = np.linalg.eigh(self.hamiltonian_k)
        self.eig_k = np.reshape(eig_k, self.dim)

    def calc_eig(self):
        hermitian = self.hamiltonian.full()
        return np.linalg.eigh(hermitian)



mag.landau_gauge()
hof = Hofstadter(3,5,0,1, alpha=0.3, q=0, magnetic=mag)
hof.add_periodic_bound()
hof.hamiltonian
# %%
hof.hamiltonian * qt.basis(hof.dim, [2,1])
# %%
hof.transform_to_momentum_space()
hof.calc_band_dispersion()
# %%
