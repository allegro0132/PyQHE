# %%
import qutip as qt
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.ticker import LinearLocator

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

    def __init__(self,
                 num_x,
                 num_y,
                 eps,
                 t,
                 alpha,
                 q,
                 magnetic=None,
                 geometry='identity') -> None:
        super().__init__()
        self.dim = [num_y, num_x]
        self.eps = eps
        self.t = t
        self.alpha = alpha
        self.q = q
        self.vpotential = magnetic.vpotential
        self.geometry = geometry

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

    def add_periodic_bound(self, mode='identity'):
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
        """2-d Discrete Fourier Transfrom.

        Args:
            geometry(str): `identity`, `cylinder` or `torus`.
        """

        # define unitary matrix for Fourier Transform
        h_dim = self.dim[0] * self.dim[1]
        real_index = [[y_n, x_m]
                      for y_n in range(self.dim[0])
                      for x_m in range(self.dim[1])]
        # In the presence of the magnetic field, lattice translation symmetry
        # was broken. we should work with the magnetic unit cells, now choose
        # `cylinder` or `torus`to start numerical calculation.
        if self.geometry == 'torus':
            # In torus geometry, just assume self.dim[1] is the length of magnetic
            # unit cell, and build hamiltonian in reciprocal lattice.
            unit_vec = np.array([1, self.dim[1]])
            k_point = [k_point[0], k_point[1] / self.dim[1]]
            # initialize a Hilbert Space in reciprocal space
            self.hamiltonian_k = self.eps / 2 * qt.qeye(self.dim[1])
            # hopping
            for i in range(self.dim[1]):
                self.hamiltonian_k += -self.t * np.exp(
                    -2j * np.pi * k_point[0] * unit_vec[0]) * self.hopping(
                        [0, i], [1, i],
                        self.alpha,
                        self.q,
                        self.vpotential,
                        reciprocal=True)
                if i != self.dim[1] - 1:
                    self.hamiltonian_k += -self.t * self.hopping(
                        [0, i], [0, i + 1],
                        self.alpha,
                        self.q,
                        self.vpotential,
                        reciprocal=True)
            self.hamiltonian_k += -self.t * np.exp(
                -2j * np.pi * k_point[1] * unit_vec[1]) * self.hopping(
                    [0, self.dim[1] - 1], [0, 0],
                    self.alpha,
                    self.q,
                    self.vpotential,
                    reciprocal=True)
            self.hamiltonian_k += self.hamiltonian_k.dag()
            return self.hamiltonian_k
        if self.geometry == 'cylinder':
            dft_matrix = []
            for orbital in range(
                    self.dim[1]):  # let x axis have periodic boundary.
                roll = [[0, x_m - orbital] for x_m in range(self.dim[1])]
                displacement = np.tile(roll, (self.dim[0], 1))
                dft_matrix.append(
                    np.exp(-2j * np.pi * np.dot(
                        (real_index - displacement), k_point)))
            dft_matrix = np.asarray(dft_matrix)
        if self.geometry == 'identity':
            dft_matrix = np.exp(-2j * np.pi * np.dot(real_index, k_point))
        dft_matrix = dft_matrix / np.sqrt(h_dim)  # normalize
        self.fmat = dft_matrix

        return dft_matrix @ self.hamiltonian.full() @ dft_matrix.conjugate().T

    def solve_klist(self, k_list):
        """Solve eigenvalue at selected kpoints in reciprocal lattice
        """
        k_list = np.asarray(k_list)
        if k_list.ndim == 1:
            k_list = [k_list]
        eig_list = []
        for kpoint in k_list:
            h_k = self.transform_to_momentum_space(kpoint)
            if not h_k.shape:
                h_k = np.array([[h_k]])
            eig_k, eigv_k = np.linalg.eigh(h_k)
            eig_list.append(eig_k)
        return np.asarray(eig_list)

    def calc_band_dispersion_2d(self, grid=30):
        """Calculate eigenenergy and eigenvector in momentum space.
        """

        k_x = np.linspace(-0.5, 0.5, grid)
        k_y = np.linspace(-0.5, 0.5, grid)
        k_list = np.array([[k_yn, k_xm] for k_yn in k_y for k_xm in k_x])
        k_y, k_x = np.meshgrid(k_y, k_x)
        eig_list = self.solve_klist(k_list)
        eig_list = np.reshape(eig_list,
                              (k_y.shape[0], k_y.shape[1], eig_list.shape[1]))
        band_0_idx = np.argmax(-eig_list, axis=-1)
        band_0 = np.take_along_axis(eig_list,
                                    np.expand_dims(band_0_idx, axis=-1),
                                    axis=-1).squeeze(axis=-1)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(k_y,
                               k_x,
                               band_0,
                               cmap=cm.coolwarm,
                               linewidth=0,
                               antialiased=False)
        # Customize the z axis.
        # ax.set_zlim(-4.01, 4.01)
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')
        plt.show()
        return eig_list

    def calc_band_dispersion_x(self, k_y=0, grid=50):
        """Calculate eigenenergy and eigenvector in momentum space along x axis.
        """

        k_x = np.linspace(-0.5, 0.5, grid)
        k_list = np.array([[k_y, k_xm] for k_xm in k_x])
        eig_list = self.solve_klist(k_list)
        plt.plot(k_x, eig_list)
        plt.show()

    def calc_eig(self):
        eig, eigv = np.linalg.eigh(self.hamiltonian)
        self.eig = np.reshape(eig, self.dim)


def draw_hofstadter_butterfly(k_point, num_orbital, num_phi):
    """Draw a Hofstadter's butterfly.
    """
    energy_list = []
    phi_list = np.linspace(0, 1, num_phi)
    for phi in phi_list:
        mag = Magnetic([0, 0, phi])
        mag.landau_gauge('y')
        hof = Hofstadter(num_orbital,
                         3,
                         0,
                         1,
                         alpha=1,
                         q=1,
                         magnetic=mag,
                         geometry='torus')
        energy_list.append(hof.solve_klist(k_point).squeeze())
    energy_list = np.asarray(energy_list)
    plt.plot(phi_list, energy_list, '.')
    plt.show()


# %%
mag = Magnetic([0, 0, 1 / 3])
mag.landau_gauge('y')
hof = Hofstadter(21, 3, 0, 1, alpha=1, q=1, magnetic=mag, geometry='torus')
# hof.add_periodic_bound()
hof.hamiltonian
# hof.hamiltonian * qt.basis(hof.dim, [0,3])
# %%
# hof.transform_to_momentum_space([0, 0])
# %%
hof.calc_band_dispersion_2d()
# %%
hof.calc_band_dispersion_x()
# %%
draw_hofstadter_butterfly([0, 0], 30, 100)
# %%
