import numpy as np


def initiate_wb_dict(layer_dims):
    """
    :param layer_dims: an array of the dimensions of each layer in the network
    :return: a dictionary containing the initialized W and b parameters of each layer
    """
    params = {f'W{i + 1}': np.random.randn(layer_dims[i + 1], layer_dims[i]) * np.sqrt(2 / layer_dims[i]) for i in
              range(len(layer_dims) - 1)}
    params.update({f'b{i + 1}': np.zeros((layer_dims[i + 1], 1)) for i in range(len(layer_dims) - 1)})

    return params
