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
    # add boundary condition
    grid = model.universal_grid
    delta = grid[0][1] - grid[0][0]
    xv, yv = np.meshgrid(*grid, indexing='ij')
    plate_length = (xv < 35) * (xv > 15)
    top_plate = (yv <= 15) * (yv >= 10)
    bottom_plate = (yv <= 65) * (yv >= 60)
    bound = np.empty_like(xv)
    bound[:] = np.nan
    bound[top_plate * plate_length] = -0.02  # meV
    # bound[bottom_plate] = 0
    model.add_dirichlet_boundary(bound)
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
plt.show()
# ax.view_init(0, 90)
# %%
shape = np.array([dim / 2 for dim in res.sigma.shape], dtype=int)
plt.plot(res.grid[1], res.sigma[shape[0]] * 1e21)
plt.show()
plt.plot(res.grid[0], res.sigma[:, 30] * 1e21)
plt.show()
# %%
xv, yv = np.meshgrid(*res.grid, indexing='ij')
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(xv,
                       yv,
                       res.v_potential,
                       cmap=cm.coolwarm,
                       linewidth=0,
                       antialiased=False)
plt.show()
plt.plot(res.grid[1], res.v_potential[shape[0]])
# %%
e_field = np.sqrt(res.e_field[0]**2 + res.e_field[1]**2)
plt.pcolormesh(xv, yv, e_field)
plt.colorbar()
plt.xlabel('Axis X(nm)')
plt.ylabel('Axis Z(nm)')
plt.show()
# %%
