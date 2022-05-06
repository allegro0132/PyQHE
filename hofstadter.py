# %%
import qutip as qt
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.ticker import LinearLocator
from scipy.fft import fft, fft2, fftshift, fftfreq

from core.method import TightBinding


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

    def transform_to_momentum_space(self, k_point):
        """2-d Discrete Fourier Transfrom."""

        # define unitary matrix for Fourier Transform
        dft_matrix = []
        for n in range(self.dim[0]):
            for m in range(self.dim[1]):
                dft_matrix = np.concatenate([
                    dft_matrix, [np.exp(2j * np.pi * np.dot(k_point, [n, m]))]
                ])
        h_dim = self.dim[0] * self.dim[1]
        dft_matrix = np.reshape(dft_matrix, (h_dim, 1))
        dft_matrix = dft_matrix / np.sqrt(h_dim)  # normalize
        self.fmat = dft_matrix
        return dft_matrix.conjugate().T @ self.hamiltonian.full() @ dft_matrix

    def solve_klist(self, k_list):
        """Solve eigenvalue at selected kpoints in reciprocal lattice
        """

        eig_list = []
        for kpoint in k_list:
            h_k = self.transform_to_momentum_space(kpoint)
            eig_k, eigv_k = np.linalg.eigh(h_k)
            eig_list.append(eig_k)
        return np.asarray(eig_list)

    def calc_band_dispersion_2d(self, grid=30):
        """Calculate eigenenergy and eigenvector in momentum space.
        """

        k_x = np.linspace(-0.5, 0.5, grid)
        k_y = np.linspace(-0.5, 0.5, grid)
        k_list = np.array([[[k_yn, k_xm] for k_xm in k_x] for k_yn in k_y])
        k_y, k_x = np.meshgrid(k_y, k_x)
        eig_list = self.solve_klist(np.reshape(k_list, (grid**2, 2)))
        eig_list = np.reshape(eig_list, k_list.shape[:2])

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(k_y,
                               k_x,
                               eig_list,
                               cmap=cm.coolwarm,
                               linewidth=0,
                               antialiased=False)
        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')
        plt.show()
        return eig_list

    def calc_eig(self):
        eig, eigv = np.linalg.eigh(self.hamiltonian)
        self.eig = np.reshape(eig, self.dim)


# %%
mag = Magnetic([0, 0, 1])
mag.landau_gauge()
hof = Hofstadter(21, 21, 0, 1, alpha=0.1, q=0, magnetic=mag)
hof.add_periodic_bound()
hof.hamiltonian
# %%
hof.calc_band_dispersion_2d()
