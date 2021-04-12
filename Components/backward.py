import numpy as np


def backward_pass(WB_dict, AZ_dict, C):
    """
    Computing the backward pass of the network
    :param WB_dict: dictionary of the weight and biases of all layers in the network
    :param AZ_dict: dictionary of the activations and Z values of all layers in the network
    :param C: the ground true labels
    :return grads_dict: dictionary with the gradients (w.r.t x,w,b) for each layer in the network
    """
    num_layers = len(WB_dict.keys()) // 2
    grads_dict = {}
    grad_w, grad_x, grad_b = backward_softmax(C, WB_dict[f'W{num_layers}'], AZ_dict[f'A{num_layers - 1}'])
    grads_dict.update({f'grads{num_layers}': {"grad_w": grad_w, "grad_x": grad_x, "grad_b": grad_b}})
    for i in range(num_layers - 1, 0, -1):
        grad_w, grad_x, grad_b = backward_linear(WB_dict[f'W{i}'], AZ_dict[f'A{i - 1}'], AZ_dict[f'Z{i}'], grad_x)
        grads_dict.update({f'grads{i}': {"grad_w": grad_w, "grad_x": grad_x, "grad_b": grad_b}})

    return grads_dict


def backward_softmax(C, W, A_prev):
    """
    computing gradients of the loss function “soft-max regression”, w.rt x,w,b
    :param C: ground truth labels
    :param W: weights of the softmax layer
    :param A_prev: activations prior to the soft-max layer
    :return grad_w, grad_x, grad_b:
    """
    grad_w = softmax_grad_wrt_weights(A_prev, W, C)
    grad_x = compute_softmax_gradient_vector_respect_to_data(A_prev, W, C)
    return grad_w, grad_x, np.zeros((W.shape[0], 1))


def backward_linear(WB, A_prev, Z_cur, dx):
    """
    Computing the gradient w.r.t x,w,b of the remaining layers of the network
    :param WB:
    :param A_prev: Activations of the previous layer
    :param Z_cur: Z values of the current layer (W*X + b)
    :param dx: grad_x of the previous layer
    :return grad_w, grad_x, grad_b:
    """
    grad_x = jacT_wrt_x(Z_cur, WB, dx)
    grad_w = jacT_wrt_w(A_prev, Z_cur, dx)
    grad_b = jacT_wrt_b(Z_cur, dx)
    return grad_w, grad_x, grad_b


def softmax_grad_wrt_weights(A_prev, W_L, C):
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


def compute_softmax_gradient_vector_respect_to_data(A_pev, W, C):
    """
    :param A_pev: nXm
    :param W: current_layer_features X prev_layer_features(n)
    :param C: l (num labels) x m (num samples)
    :return:
    """
    nominator = np.exp(np.matmul(W, A_pev))
    denominator = sum(np.exp(np.matmul(W[k], A_pev)) for k in range(0, W.shape[0]))
    return 1 / A_pev.shape[1] * (np.matmul(W.T, nominator / denominator - C))


# p 16 (w.r.t b)
def jacT_wrt_b(Z, v):
    """
    Computing the Jacobian matrix (transposed) w.r.t 'b' as depicted in 'DeepLearningFromScratchV2' p16
    :param Z:  z values of current layer (W*X + b)
    :param v:  dx passed from the next layer
    :return jac_res:
    """
    m = Z.shape[1]
    tanh_derv = derv_tanh(Z)  # tanh'(w*x +b)
    pair_wise_mult = np.multiply(tanh_derv, v)
    grad_b = (1 / m) * np.sum(pair_wise_mult, axis=1)
    grad_b = grad_b.reshape(-1, 1)
    return grad_b


# p 16 (w.r.t w)
def jacT_wrt_w(a_prev, Z, v):
    """
    Computing the Jacobian matrix (transposed) w.r.t 'W' as depicted in 'DeepLearningFromScratchV2' p16
    :param a_prev: Activations of the previous layer
    :param Z: z values of current layer (W*X + b)
    :param v: dx passed from the next layer
    :return jac_res:
    """
    tanh_derv = derv_tanh(Z)
    pair_wise_mult = np.multiply(tanh_derv, v)  # tanh_derv * v
    jac_res = np.matmul(pair_wise_mult, a_prev.T)
    return jac_res


# p 16 (w.r.t x)
def jacT_wrt_x(Z, W, v):
    """
    Computing the Jacobian matrix (transposed) w.r.t 'x' as depicted in 'DeepLearningFromScratchV2' p16
    :param Z: z values of current layer (W*X + b)
    :param W:
    :param v: dx passed from the next layer
    :return jac_res:
    """
    tanh_derv = derv_tanh(Z)
    pair_wise_mult = np.multiply(tanh_derv, v)
    jac_res = np.matmul(W.T, pair_wise_mult)
    return jac_res


def derv_tanh(x):
    """
    computing tanh Derivative
    :param x:
    :return:
    """
    ones_matrix = np.ones(x.shape)
    tanh_x_power_2 = np.multiply(np.tanh(x), np.tanh(x))
    return np.subtract(ones_matrix, tanh_x_power_2)
