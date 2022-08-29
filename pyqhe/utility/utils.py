import numpy as np

def tensor(*opt):
    if len(opt) == 1:
        opt = opt[0]
    kron_product = opt[0]
    for element in opt[1:]:
        kron_product = np.kron(kron_product, element)
    return kron_product
