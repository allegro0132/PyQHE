# %%
import qutip as qt
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.ticker import LinearLocator

from pyqhe.core.method import TightBinding

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
                 num_y,
                 num_x,
                 eps,
                 t,
                 alpha,
                 q,
                 magnetic=None,
                 geometry='identity') -> None:
        super().__init__()
        self.dim = [num_y, num_x]
        self.r_vec = self.dim  # Just work with 2-d orthogonal lattice now.
        self._gen_reciprocal_lattice()

        self.eps = eps
        self.t = t
        self.alpha = alpha
        self.q = q
        self.vpotential = magnetic.vpotential
        self.geometry = geometry
        self.hamiltonian_k = 0
        if self.geometry in ['torus', 'cylinder', 'identity']:

            # initialize a Hilbert Space in momentum space
            self.hamiltonian_k = self.eps / 2 * qt.qeye(self.dim)
            # present lattice in fock basis
            # forward hopping
            for n in range(self.dim[0]):
                for m in range(self.dim[1]):
                    if m + n == sum(self.dim) - 2:
                        continue
                    elif m == self.dim[1] - 1 and n < self.dim[0] - 1:
                        self.hamiltonian_k += -self.t * self.hopping(
                            [n, m], [n + 1, m], self.alpha, self.q, self.vpotential)
                    elif n == self.dim[0] - 1 and m < self.dim[1] - 1:
                        self.hamiltonian_k += -self.t * self.hopping(
                            [n, m], [n, m + 1], self.alpha, self.q, self.vpotential)
                    else:
                        self.hamiltonian_k += -self.t * self.hopping(
                            [n, m], [n, m + 1], self.alpha, self.q, self.vpotential)
                        self.hamiltonian_k += -self.t * self.hopping(
                            [n, m], [n + 1, m], self.alpha, self.q, self.vpotential)
            # conjugate hopping
            self.hamiltonian_k += self.hamiltonian_k.dag()

    def transform_to_momentum_space(self, k_point):
        """2-d Discrete Fourier Transfrom.

        Args:
            geometry(str): `identity`, `cylinder` or `torus`.
        """

        k_point = [k_point[0] * self.k_vec[0], k_point[1] * self.k_vec[1]]
        # In the presence of the magnetic field, lattice translation symmetry
        # was broken. we should work with the magnetic unit cells, now choose
        # `cylinder` or `torus`to start numerical calculation.
        if self.geometry in ['cylinder', 'torus']:
            h_k = self.hamiltonian_k.copy()
            for n in range(self.dim[0]):  # Periodic boundary at x axis
                periodic_term = -self.t * np.exp(
                    -2j * np.pi * k_point[1] * self.r_vec[1]) * self.hopping(
                        [n, self.dim[1] - 1], [n, 0], self.alpha, self.q,
                        self.vpotential)
                h_k += periodic_term + periodic_term.dag()
            if self.geometry == 'torus':
                for m in range(self.dim[1]):  # Periodic boundary at y axis
                    periodic_term = -self.t * np.exp(
                        -2j * np.pi * k_point[0] *
                        self.r_vec[0]) * self.hopping(
                            [self.dim[0] - 1, m], [0, m], self.alpha, self.q,
                            self.vpotential)
                    h_k += periodic_term + periodic_term.dag()

        if self.geometry == 'torus_1d':
            # In torus geometry, just assume self.dim[1] is the length of magnetic
            # unit cell, and build hamiltonian in reciprocal lattice.

            # dft_matrix = []
            # for orbital in range(
            #         self.dim[1]):  # let x axis have periodic boundary.
            #     roll = [[0, x_m - orbital] for x_m in range(self.dim[1])]
            #     displacement = np.tile(roll, (self.dim[0], 1))
            #     dft_matrix.append(
            #         np.exp(-2j * np.pi * np.dot(
            #             (real_index - displacement), k_point)))
            # dft_matrix = np.asarray(dft_matrix)

            # initialize a Hilbert Space in reciprocal space
            h_k = self.eps / 2 * qt.qeye(self.dim[1])
            # hopping
            for i in range(self.dim[1]):
                h_k += -self.t * np.exp(
                    -2j * np.pi * k_point[0] *
                    self.r_vec[0]) * self.hopping([0, i], [1, i],
                                                self.alpha,
                                                self.q,
                                                self.vpotential,
                                                hopping_1d=True)
                if i != self.dim[1] - 1:
                    h_k += -self.t * np.exp(
                        -2j * np.pi * k_point[1] * self.r_vec[1]) * self.hopping(
                            [0, i], [0, i + 1],
                            self.alpha,
                            self.q,
                            self.vpotential,
                            hopping_1d=True)
            h_k += -self.t * np.exp(
                -2j * np.pi * k_point[1] * self.r_vec[1]) * self.hopping(
                    [0, self.dim[1] - 1], [0, 0],
                    self.alpha,
                    self.q,
                    self.vpotential,
                    hopping_1d=True)
            h_k += h_k.dag()

        if self.geometry == 'cylinder_1d':
            unit_vec = np.array([1, 3])
            # initialize a Hilbert Space in reciprocal space
            h_k = self.eps / 2 * qt.qeye(self.dim[1])
            # hopping
            for i in range(self.dim[1]):
                h_k += -self.t * np.exp(
                    -2j * np.pi * k_point[0] *
                    unit_vec[0]) * self.hopping([0, i], [1, i],
                                                self.alpha,
                                                self.q,
                                                self.vpotential,
                                                hopping_1d=True)
                if i != self.dim[1] - 1:
                    h_k += -self.t * self.hopping(
                            [0, i], [0, i + 1],
                            self.alpha,
                            self.q,
                            self.vpotential,
                            hopping_1d=True)
            h_k += h_k.dag()

        if self.geometry == 'identity':
            real_point = [[y_n, x_m] for y_n in range(self.dim[0]) for x_m in range(self.dim[1])]
            k_point_normal = [
                k_point[0] / self.k_vec[0], k_point[1] / self.k_vec[1]
            ]
            dft_matrix = np.exp(-2j * np.pi * np.dot(real_point, k_point_normal))
            dft_matrix = dft_matrix / np.sqrt(self.dim[0] * self.dim[1])  # normalize
            h_k = dft_matrix @ self.hamiltonian_k.full() @ dft_matrix.conjugate().T

        return h_k

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
        band_0_idx = np.argmin(eig_list, axis=-1)
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
        for eig in eig_list.T:
            plt.scatter(k_x, eig, edgecolors='none', s=0.6, c='k')
        plt.xlabel('$k_x$')
        plt.xticks([-0.5, -1 / 3, -1 / 6, 0, 1 / 6, 1 / 3, 0.5],
                   ['$-\pi$', '$-2\pi/3$', '$-\pi/3$', 0, '$\pi/3$', '$2\pi/3$', '$\pi$'])
        plt.ylabel('Energy E')
        plt.yticks([-3, -2, -1, 0, 1, 2, 3], ['-3t', '-2t', '-1t', 0, '1t', '2t', '3t'])

    def calc_band_dispersion_y(self, k_x=0, grid=50):
        """Calculate eigenenergy and eigenvector in momentum space along x axis.
        """

        k_y = np.linspace(-0.5, 0.5, grid)
        k_list = np.array([[k_yn, k_x] for k_yn in k_y])
        eig_list = self.solve_klist(k_list)
        plt.scatter(k_list, eig_list, edgecolors='none', s=0.6, c='k')
        plt.show()


def draw_hofstadter_butterfly(k_point, num_phi,  max_num_orbital=1000):
    """Draw a Hofstadter's butterfly.
    """

    from fractions import Fraction

    energy_list = []
    phi_list = np.linspace(0, 1, num_phi)
    phi_denominator = [Fraction(phi).limit_denominator(max_num_orbital).denominator for phi in phi_list]
    for i, phi in enumerate(phi_list):
        mag = Magnetic([0, 0, phi])
        mag.landau_gauge('y')
        hof = Hofstadter(1,
                         phi_denominator[i],
                         0,
                         1,
                         alpha=1,
                         q=1,
                         magnetic=mag,
                         geometry='torus_1d')
        energy_list.append(hof.solve_klist(k_point).squeeze())
    energy_list = np.asarray(energy_list)
    for i, phi in enumerate(phi_list):
        if energy_list[i].ndim == 0:
            plt.scatter(energy_list[i], phi, edgecolors='none', s=0.6, c='k')
        else:
            for energy in energy_list[i]:
                plt.scatter(energy, phi, edgecolors='none', s=0.6, c='k')
    y_lines = [0, 1 / 4, 2 / 6, 1 / 2, 4 / 6, 3 / 4, 1]
    x_bound = [[-4, -3, -3, -3, -3, -3 , -4],
               [4, 3, 3, 3, 3, 3, 4] ]
    plt.hlines(y_lines, x_bound[0], x_bound[1], ['b', 'c', 'm', 'y', 'm', 'c', 'b'])
    plt.xlabel('Energy E')
    plt.xticks([-4,0,4], ['-4t', 0, '4t'])
    plt.ylabel('Flux $\phi$')
    plt.yticks(y_lines,[ 0, '$\pi/2$', '$2\pi/3$',
                '$\pi$', '$4\pi/3$', '$3\pi/2$', '$2\pi$'])
    plt.show()


# %%
mag = Magnetic([0, 0, 1 / 3])
mag.landau_gauge('y')
hof = Hofstadter(1, 3, 0, 1, alpha=1, q=1, magnetic=mag, geometry='torus_1d')
# hof.add_periodic_bound()
hof.hamiltonian_k
# %%
hof.hamiltonian_k * qt.basis(hof.dim, [0, 0])
# %%
hof.transform_to_momentum_space([0, 0.1])
# %%
hof.calc_band_dispersion_2d()
# %%
for x in np.linspace(-0.5, 0.5, 20):
    hof.calc_band_dispersion_x(x)
plt.show()
# %%
draw_hofstadter_butterfly([0, 0], 100)
# %%
