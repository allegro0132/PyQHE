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

model = Structure1D(layer_list, temp=10, dx=0.1)
# instance of class SchrodingerPoisson
schpois = SchrodingerPoisson(model)
# %%
# perform self consistent optimization
res = schpois.self_consistent_minimize()
# %%
# plot result
res.plot_quantum_well()
# %%
res.eig_val
# %%
