import numpy as np


def forward_pass(X, wb_dict):
    """
    compute forward pass of the network
    :param X: Input Data - dimension: n (num features) x m (num samples)
    :param wb_dict: dictionary with Weights and biases for each layer of the network
    :param C: the true labels (or the class which it actually belongs)  dimension: l (num labels) x m (num samples)
    :return:
    """
    num_layers = len(wb_dict) // 2
    A_i = X
    AZ_dict = {f'A{0}': A_i}
    for layer_i in range(1, num_layers):
        W_i = wb_dict[f'W{layer_i}']
        Z_i = np.matmul(W_i, A_i) + wb_dict[f'b{layer_i}']
        A_i = tanh_func(Z_i)
        AZ_dict.update({f'A{layer_i}': A_i, f'Z{layer_i}': Z_i})

    W_L = wb_dict[f'W{num_layers}']
    Z_L = np.matmul(W_L, A_i)  # TODO: add bias here??
    A_L = softmax(Z_L)
    AZ_dict.update({f'A{num_layers}': A_L, f'Z{num_layers}': Z_L})
    return A_L, AZ_dict


def tanh_func(Z_i):
    return np.tanh(Z_i)


def relu_func(Z):
    def relu(zi):
        return np.maximum(0., zi)

    relu_func = np.vectorize(relu)
    A = relu_func(Z)
    return A


def softmax(Z):
    """
    :param Z: the linear component of the activation function
    :return A: the activations of the layer
    :return activation_cache: returns Z, which will be useful for the backpropagation
    """
    exp = np.exp(Z)
    A = exp / np.sum(exp, axis=0)[None, :]
    return A


def safe_softmax(Z):
    Z_safe = Z - Z.max(axis=0)
    return softmax(Z_safe)


def cross_entropy_softmax_lost(A_prev, W_L, C, with_eta=False):
    """
    computation of lost function
    :param with_eta: use eta for safety
    :param A_prev: Z values of last layer - dimension: n (num features) x m (num samples)
    :param W_L: weights matrix - dimension: num_neurons_cur x num_neurons_prev
    :param C: classes vector matrix - dimension: l (num labels) x m (num samples)
    :return loss: dimensions: scalar
    """
    l = C.shape[0]
    m = A_prev.shape[1]
    X_t = np.transpose(A_prev)
    eta_lst = [max([np.matmul(X_t[i], W_L[k]) for k in range(0, l)]) for i in range(0, m)] if with_eta else [0] * m
    softmax_denominator = sum([np.exp(np.matmul(X_t, W_L[k]) - eta_lst) for k in range(0, l)])
    loss = 0
    for k in range(0, l):
        softmax_prob = np.log(np.exp(np.matmul(X_t, W_L[k]) - eta_lst) / softmax_denominator)
        loss += np.matmul(C[k], softmax_prob)
    loss *= (-1 / m)

    # loss = (-1 / m) * sum(
    #     np.matmul(C[k], np.log(np.exp(np.matmul(X_t, W[:, k]) - eta_lst) / softmax_denominator)) for k
    #     in range(0, l))

    return loss
