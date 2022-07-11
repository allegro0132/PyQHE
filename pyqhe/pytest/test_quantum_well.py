# %%
import numpy as np
from scipy import optimize
from scipy.stats import norm
from matplotlib import pyplot as plt

from pyqhe.core.structure import Layer, Structure1D
from pyqhe.schrodinger_poisson import SchrodingerPoisson
from pyqhe.equation.poisson import PoissonFDM, PoissonODE

# map between effective layer thickness and quantum well width
stick_nm = [20, 30, 40, 45, 50, 60, 70, 80]
stick_list = [
    0.66575952, 0.97971301, 1.36046512, 1.64324592, 1.88372093, 2.59401286,
    3.25977239, 3.98119743]  # unit in l_B


def calc_omega(thickness=10, tol=5e-5):
    layer_list = []
    layer_list.append(Layer(20, 0.24, 0.0, name='barrier'))
    layer_list.append(Layer(2, 0.24, 5e17, name='n-type'))
    layer_list.append(Layer(5, 0.24, 0.0, name='spacer'))
    layer_list.append(Layer(thickness, 0, 0, name='quantum_well'))
    layer_list.append(Layer(5, 0.24, 0.0, name='spacer'))
    layer_list.append(Layer(2, 0.24, 5e17, name='n-type'))
    layer_list.append(Layer(20, 0.24, 0.0, name='barrier'))

    model = Structure1D(layer_list, temp=10, dz=0.2)
    # instance of class SchrodingerPoisson
    schpois = SchrodingerPoisson(model,
                                 poisolver=PoissonFDM,
                                 # quantum_region=(255 - 20, 255 + thickness + 30),
                                )
    # perform self consistent optimization
    res, loss = schpois.self_consistent_minimize(tol=tol)
    if loss > tol:
        res, loss = schpois.self_consistent_minimize(tol=tol)
    # fit a normal distribution to the data
    def gaussian(x, amplitude, mean, stddev):
        return amplitude * np.exp(-(x - mean)**2 / 2 / stddev**2)

    popt, _ = optimize.curve_fit(gaussian,
                                 res.grid[schpois.quantum_mask],
                                 res.wave_function[0][schpois.quantum_mask],
                                 p0=[1, np.mean(res.grid), 10])
    plt.plot(res.grid[schpois.quantum_mask], res.wave_function[0][schpois.quantum_mask], label='Self consistent')
    plt.plot(res.grid[schpois.quantum_mask],
             gaussian(res.grid[schpois.quantum_mask], *popt),
             label='Norm fit')
    plt.xlabel('Position (nm)')
    plt.ylabel('Charge distribution')
    plt.legend()
    plt.show()

    return popt[2], res
# %%
thickness_list = np.linspace(10, 65, 20)
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
# plt.plot(omega_list, line(omega_list, *popt1))
plt.plot(np.asarray(stick_list) * 7.1, stick_nm, label='Shayegan')
plt.xlabel(r'Layer thickness $\bar{\omega}$ (nm)')
plt.ylabel(r'Geometry thickness $\omega$ (nm)')
plt.legend()
# %%
res_list[15].plot_quantum_well()
# %%
plt.plot(res_list[15].grid, res_list[15].wave_function[0])
# %%
