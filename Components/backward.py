import numpy as np


def backward_pass(A_L, W_dict, A_dict, C):
    num_layers = len(W_dict.keys())
    grad_w, grad_x = backward_softmax(C, W_dict[f'W{num_layers}'], A_dict[f'A{num_layers - 1}'])
    for i in range(num_layers - 1, 0, -1):
        grad_w, grad_x = backward_linear(W_dict[f'W{i}'], A_dict[f'A{i - 1}'], grad_x, W_dict[f'b{i}'])

    # TODO: update the network...


def backward_softmax(C, W, A_prev):
    grad_w = compute_softmax_gradient_vector_respect_to_weights(A_prev, W, C)
    grad_x = compute_softmax_gradient_vector_respect_to_data(A_prev, W, C)
    return grad_w, grad_x


def backward_linear(W, A_prev, dx, b=None):
    grad_x = jacobianTMV_grad_x(A_prev, W, dx, b)
    grad_w = jacobianTMV_grad_w(A_prev, W, dx, b)
    return grad_w, grad_x


def compute_softmax_gradient_vector_respect_to_weights(A_prev, W_L, C):
    """
    :param A_prev: activation of last layer - dimension: n (num features) x m (num samples)
    :param W_L: weights matrix - dimension: n (neuron current layer) x l (neuron prev layer)
    :param C: classes vector matrix - dimension: l (num labels) x m (num samples)
    :return grads:  gradient with respect to 'W' (same shape as W)
    """
    m = A_prev.shape[1]
    num_labels = C.shape[0]
    X_t = np.transpose(A_prev)
    denominator = sum([np.exp(np.matmul(X_t, W_L[k])) for k in range(0, num_labels)])
    grads = np.array([(1 / m) * np.matmul(A_prev, (np.exp(np.matmul(X_t, W_L[p])) / denominator) - C[p]) for p in
                      range(0, num_labels)])
    return grads

    # grads = []
    # for p in range(0, l):
    #     C_p = C[p]
    #     nominator = np.exp(np.matmul(X_t, W[:, p]))
    #     grads.append((1 / m) * np.matmul(X, (nominator / denominator) - C_p))
    # grads = np.array(grads).transpose()
    # return grads


def compute_softmax_gradient_vector_respect_to_data(A_pev, W, C):
    """
    :param A_pev: nXm
    :param W: current_layer_features X prev_layer_features(n)
    :param C: l (num labels) x m (num samples)
    :return:
    """
    nominator = np.exp(np.matmul(W, A_pev))
    denominator = np.sum(np.exp(np.matmul(W[k], A_pev)) for k in range(0, W.shape[0]))
    return 1 / A_pev.shape[1] * (np.matmul(W.T, nominator / denominator - C))


# p 16 (w.r.t w)
def jacobianTMV_grad_w(X, W, v, b):
    """

    :param X:
    :param W:
    :param v:  we think it is dx
    :param b:
    :return:
    """
    wx_b = np.matmul(W, X) + b
    tanh_derv = derv_tanh(wx_b)
    pair_wise_mult = np.multiply(tanh_derv, v)
    jac_res = np.matmul(pair_wise_mult, X.T)
    return jac_res


# p 16 (w.r.t x)
def jacobianTMV_grad_x(X, W, v, b):
    """
    :param X:
    :param W:
    :param v: we think it is dx
    :param b:
    :return:
    """
    wx_b = np.matmul(W, X) + b
    tanh_derv = derv_tanh(wx_b)
    pair_wise_mult = np.multiply(tanh_derv, v)
    jac_res = np.matmul(W.T, pair_wise_mult)
    return jac_res


def derv_tanh(x):
    ones_matrix = np.ones(x.shape)
    tanh_x_power_2 = np.multiply(np.tanh(x), np.tanh(x))
    return np.subtract(ones_matrix, tanh_x_power_2)
