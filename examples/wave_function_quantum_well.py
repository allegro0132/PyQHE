# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import jax
import jax.numpy as jnp

from pyqhe.core.structure import Layer, Structure1D
from pyqhe.schrodinger_poisson import SchrodingerPoisson
from pyqhe.equation.poisson import PoissonFDM
from pyqhe.equation.schrodinger import SchrodingerMatrix


def factor_q_fh(thickness, q):
    """Using the Fang-Howard variational wave function to describe the electron
    wave function in the perpendicular direction.

    Args:
        thickness: physical meaning of electron layer thickness.
    """
    b = 1 / thickness
    return (1 + 9 / 8 * q / b + 3 / 8 * (q / b)**2) * (1 + q / b)**(-3)

@jax.jit
def factor_q(grid, wave_func, q):
    """Calculate `F(q)` in reduced Coulomb interaction."""
    grid = jnp.asarray(grid)
    wave_func = jnp.asarray(wave_func)
    # make 2-d coordinate matrix
    z_1, z_2 = jnp.meshgrid(grid, grid)
    exp_term = jnp.exp(-q * jnp.abs(z_1 - z_2))
    wf2_z1, wf2_z2 = jnp.meshgrid(wave_func**2, wave_func**2)
    # wf2_z1, wf2_z2 = np.meshgrid(wave_func, wave_func)
    factor_matrix = wf2_z1 * wf2_z2 * exp_term
    # integrate using the composite trapezoidal rule
    return jnp.trapz(jnp.trapz(factor_matrix, grid), grid)


def calc_wave_function(thickness, tol=5e-5):
    # construct model
    layer_list = []
    layer_list.append(Layer(20, 0.24, 0.0, name='barrier'))
    layer_list.append(Layer(2, 0.24, 5e17, name='n-type'))
    layer_list.append(Layer(5, 0.24, 0.0, name='spacer'))
    layer_list.append(Layer(thickness, 0, 0, name='quantum_well'))
    layer_list.append(Layer(5, 0.24, 0.0, name='spacer'))
    layer_list.append(Layer(2, 0.24, 5e17, name='n-type'))
    layer_list.append(Layer(20, 0.24, 0.0, name='barrier'))

    model = Structure1D(layer_list, temp=10, dz=0.01)
    # instance of class SchrodingerPoisson
    schpois = SchrodingerPoisson(
        model,
        schsolver=SchrodingerMatrix,
        poisolver=PoissonFDM,
    )
    # perform self consistent optimization
    res, _ = schpois.self_consistent_minimize(tol=tol)
    return res


# %%
thickness = 14.82
res = calc_wave_function(thickness)
res.plot_quantum_well()
# Ground state wave function
wf = res.wave_function[0]
# wf = res.electron_density
# %%
# load Yihan's data template
with open('../template/Ns_15.txt', 'rb') as file:
    df = pd.read_csv(file, sep=' ', names=['index_x', 'index_y', 'q_x', 'q_y', 'f'])
# %%
q_x = df['q_x'].to_numpy()
q_y = df['q_y'].to_numpy()
q_list = np.sqrt(q_x**2 + q_y**2)
# calculate `F(q)` in reduced Coulomb interaction
form_factor = [factor_q(res.grid[0], wf, q) for q in q_list]
# insert the form factor to DataFrame
df['form_factor'] = form_factor
# %%
# insert the form factor to DataFrame
# import pyqhe.utility.constant as const
# l_b = np.sqrt(const.hbar/(const.q * 21))
# df['form_factor_fh'] = factor_q_fh(thickness, q_list)
# %%
# df.to_excel('../output/output.xlsx', sheet_name='partical_5')
df.to_csv('../output/Ns_15_beta_0_5.txt', sep=' ')
# %%
plt.show()
plt.plot(q_list, np.array([df['f'], df['form_factor']]).T, label=['f_fang-howard', 'wave_function'])
plt.legend()
plt.xlabel('|q|')
plt.show()
# %%
