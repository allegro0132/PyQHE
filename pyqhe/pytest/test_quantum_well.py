# %%
import numpy as np
from matplotlib import pyplot as plt

from pyqhe.core.structure import Layer, Structure1D
from pyqhe.schrodinger_poisson import SchrodingerPoisson

# %%
layer_list = []
layer_list.append(Layer(10, 0.3, 0.0, name='barrier'))
layer_list.append(Layer(5, 0.3, 5e17, name='n-type'))
layer_list.append(Layer(5, 0.3, 0.0, name='cladding'))
layer_list.append(Layer(11, 0, 0, name='quantum_well'))
layer_list.append(Layer(5, 0.3, 0.0, name='cladding'))
layer_list.append(Layer(5, 0.3, 5e17, name='n-type'))
layer_list.append(Layer(10, 0.3, 0.0, name='barrier'))

model = Structure1D(layer_list, temp=60)
# %%
schpois = SchrodingerPoisson(model)
# drawing initial v_potential and electrons' eigenenergy and wave function
plt.plot(schpois.grid, schpois.fi)
plt.hlines(schpois.eig_val, 10, 40)
# %%
res = schpois.self_consistent_minimize()
res.params
# %%
plt.plot(res.grid, res.v_potential)
plt.hlines(res.eig_val, 10, 40)
# %%
plt.plot(res.grid, res.e_field)
# plt.plot(res.grid, res.wave_function.T + 1)
# %%
