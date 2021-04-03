import numpy as np

from Components import sgd


def backward_pass(A_L, WB_dict, AZ_dict, C):
    num_layers = len(WB_dict.keys()) // 2
    grads_dict = {}
    grad_w, grad_x, grad_b = backward_softmax(C, WB_dict[f'W{num_layers}'], AZ_dict[f'A{num_layers - 1}'])
    grads_dict.update({f'grads{num_layers}': {"grad_w": grad_w, "grad_x": grad_x, "grad_b": grad_b}})
    for i in range(num_layers - 1, 0, -1):
        grad_w, grad_x, grad_b = backward_linear(WB_dict[f'W{i}'], AZ_dict[f'A{i - 1}'], AZ_dict[f'Z{i}'], grad_x, WB_dict[f'b{i}'])
        grads_dict.update({f'grads{i}': {"grad_w": grad_w, "grad_x": grad_x, "grad_b": grad_b}})

    return grads_dict


def backward_softmax(C, W, A_prev):
    grad_w = compute_softmax_gradient_vector_respect_to_weights(A_prev, W, C)
    grad_x = compute_softmax_gradient_vector_respect_to_data(A_prev, W, C)
    return grad_w, grad_x, np.zeros((W.shape[0], 1))



def backward_linear(WB, A_prev, Z_cur, dx, b=None):
    m = A_prev.shape[1]
    grad_x = jacobianTMV_grad_x(Z_cur, WB, dx)
    grad_w = jacobianTMV_grad_w(A_prev, Z_cur, dx)
    grad_b = jacobianTMV_grad_b(Z_cur, dx)
    grad_w = (1 / m) * grad_w
    grad_b = (1 / m) * np.sum(grad_b, axis=1)
    grad_b = grad_b.reshape(-1, 1)

    return grad_w, grad_x, grad_b


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
def jacobianTMV_grad_b(z, v):
    """
    :param x:
    :param W:
    :param v:  we think it is dx
    :param b:
    :return:
    """
    # wx_b = np.matmul(W, x) + b  # w * x + b
    # tanh_derv = derv_tanh(wx_b)  # tanh'(w*x +b)
    tanh_derv = derv_tanh(z)  # tanh'(w*x +b)
    pair_wise_mult = np.multiply(tanh_derv, v)  # tanh_derv * v
    return pair_wise_mult


# p 16 (w.r.t w)
def jacobianTMV_grad_w(a_prev, z, v):
    """

    :param a_prev:
    :param W:
    :param v:  we think it is dx
    :param b:
    :return:
    """
    # wx_b = np.matmul(W, x) + b  # w * X + b
    # tanh_derv = derv_tanh(wx_b)  # tanh'(w*x +b)
    tanh_derv = derv_tanh(z)
    pair_wise_mult = np.multiply(tanh_derv, v)  # tanh_derv * v
    jac_res = np.matmul(pair_wise_mult, a_prev.T)
    return jac_res


# p 16 (w.r.t x)
def jacobianTMV_grad_x(z, W, v):
    """
    :param a_prev:
    :param W:
    :param v: we think it is dx
    :param b:
    :return:
    """
    tanh_derv = derv_tanh(z)
    pair_wise_mult = np.multiply(tanh_derv, v)
    jac_res = np.matmul(W.T, pair_wise_mult)
    return jac_res


def derv_tanh(x):
    ones_matrix = np.ones(x.shape)
    tanh_x_power_2 = np.multiply(np.tanh(x), np.tanh(x))
    return np.subtract(ones_matrix, tanh_x_power_2)
