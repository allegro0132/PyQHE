# %%
import numpy as np
from scipy import optimize
from scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib import cm

from pyqhe.core.structure import Layer, Structure2D, Structure3D
from pyqhe.schrodinger_poisson import SchrodingerPoisson
from pyqhe.equation.poisson import PoissonFDM, PoissonODE
from pyqhe.equation.schrodinger import SchrodingerMatrix

# map between effective layer thickness and quantum well width
stick_nm = [20, 30, 40, 45, 50, 60, 70, 80]
stick_list = [
    0.66575952, 0.97971301, 1.36046512, 1.64324592, 1.88372093, 2.59401286,
    3.25977239, 3.98119743
]  # unit in l_B


def factor_q_fh(thickness, q):
    """Using the Fang-Howard variational wave function to describe the electron
    wave function in the perpendicular direction.

    Args:
        thickness: physical meaning of electron layer thickness.
    """
    b = 1 / thickness
    return (1 + 9 / 8 * q / b + 3 / 8 * (q / b)**2) * (1 + q / b)**(-3)


def factor_q(grid, wave_func, q):
    """Calculate `F(q)` in reduced Coulomb interaction."""
    # make 2-d coordinate matrix
    z_1, z_2 = np.meshgrid(grid, grid)
    exp_term = np.exp(-q * np.abs(z_1 - z_2))
    wf2_z1, wf2_z2 = np.meshgrid(wave_func**2, wave_func**2)
    factor_matrix = wf2_z1 * wf2_z2 * exp_term
    # integrate using the composite trapezoidal rule
    return np.trapz(np.trapz(factor_matrix, grid), grid)


def calc_omega(thickness=10, tol=5e-5):
    layer_list = []
    layer_list.append(Layer(20, 0.24, 0.0, name='barrier'))
    layer_list.append(Layer(2, 0.24, 5e17, name='n-type'))
    layer_list.append(Layer(5, 0.24, 0.0, name='spacer'))
    layer_list.append(Layer(thickness, 0, 0, name='quantum_well'))
    layer_list.append(Layer(5, 0.24, 0.0, name='spacer'))
    layer_list.append(Layer(2, 0.24, 5e17, name='n-type'))
    layer_list.append(Layer(20, 0.24, 0.0, name='barrier'))

    model = Structure2D(layer_list, width=50, temp=10, delta=1)
    # instance of class SchrodingerPoisson
    schpois = SchrodingerPoisson(
        model,
        schsolver=SchrodingerMatrix,
        poisolver=PoissonFDM,
        # quantum_region=(255 - 20, 255 + thickness + 30),
    )
    # test = schpois.sch_solver.calc_evals()
    # perform self consistent optimization
    res, loss = schpois.self_consistent_minimize(tol=tol)
    if loss > tol:
        res, loss = schpois.self_consistent_minimize(tol=tol)
    # plot 2DES areal electron density
    xv, yv = np.meshgrid(*schpois.grid, indexing='ij')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(xv,
                           yv,
                           res.sigma * thickness * 1e14,
                           cmap=cm.coolwarm,
                           linewidth=0,
                           antialiased=False)
    plt.show()

    # # fit a normal distribution to the data
    # def gaussian(x, amplitude, mean, stddev):
    #     return amplitude * np.exp(-(x - mean)**2 / 2 / stddev**2)

    # popt, _ = optimize.curve_fit(gaussian,
    #                              res.grid,
    #                              res.electron_density,
    #                              p0=[1, np.mean(res.grid), 10])
    # wf2 = res.wave_function[0] * np.conj(
    #     res.wave_function[0])  # only ground state
    # symmetry_axis = popt[1]
    # # calculate standard deviation
    # # the standard deviation of the charge distribution from its center, from PRL, 127, 056801 (2021)
    # charge_distribution = res.electron_density / np.trapz(
    #     res.electron_density, res.grid)
    # sigma = np.sqrt(
    #     np.trapz(charge_distribution * (res.grid - symmetry_axis)**2, res.grid))
    # plt.plot(res.grid,
    #          wf2,
    #          label=r'$|\Psi(z)|^2$')
    # plt.plot(res.grid,
    #          charge_distribution,
    #          label='Charge distribution',
    #          color='r')
    # plt.axvline(symmetry_axis - sigma, ls='--', color='y')
    # plt.axvline(symmetry_axis + sigma, ls='--', color='y')
    # plt.xlabel('Position (nm)')
    # plt.ylabel('Distribution')
    # plt.legend()
    # plt.show()
    # # plot factor_q verses q
    # q_list = np.linspace(0, 1, 20)
    # f_fh = []
    # f_self = []
    # for q in q_list:
    #     f_fh.append(factor_q_fh(thickness, q))
    #     f_self.append(factor_q(res.grid, res.wave_function[0], q))
    # plt.plot(q_list, f_fh, label='the Fang-Howard wave function')
    # plt.plot(q_list, f_self, label='self-consistent wave function', color='r')
    # plt.legend()
    # plt.xlabel('wave vector q')
    # plt.ylabel(r'$F(q)$')
    # plt.show()

    return res


# %%
res = calc_omega(20)
# res.plot_quantum_well()
# %%
xv, yv = np.meshgrid(*res.grid, indexing='ij')
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xv,
                       yv,
                       res.electron_density,
                       cmap=cm.coolwarm,
                       linewidth=0,
                       antialiased=False)
# ax.view_init(0, 90)
# %%
shape = np.array([dim / 2 for dim in res.sigma.shape], dtype=int)
plt.plot(res.grid[1], res.sigma[shape[0]] * 20 * 1e14)
plt.show()
plt.plot(res.grid[0], res.sigma[:, shape[1]] * 20 * 1e14)
# %%
thickness_list = np.linspace(10, 80, 30)
res_list = []
omega_list = []
for thick in thickness_list:
    omega, res = calc_omega(thick)
    res_list.append(res)
    omega_list.append(omega)


# %%
def line(x, a, b):
    return a * np.asarray(x) + b


popt1, _ = optimize.curve_fit(line, omega_list[:3], thickness_list[:3])
# %%
plt.plot(np.asarray(omega_list), thickness_list, label='PyQHE')
plt.plot(np.asarray(stick_list) * 7.1, stick_nm, label='Shayegan')
plt.xlabel(r'Layer thickness $\bar{\omega}$ (nm)')
plt.ylabel(r'Geometry thickness $\omega$ (nm)')
plt.legend()
# %%
res_list[-1].plot_quantum_well()
# %%
plt.plot(res.grid, res.params)
# %%
