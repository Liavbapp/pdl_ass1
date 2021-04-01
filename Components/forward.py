import numpy as np


def forward_pass(X, W_dict):
    """
    compute forward pass of the network
    :param X: Input Data - dimension: n (num features) x m (num samples)
    :param W_dict: dictionary with Weights for each layer of the network
    :param C: the true labels (or the class which it actually belongs)  dimension: l (num labels) x m (num samples)
    :return:
    """
    num_layers = len(W_dict)
    A_i = X
    for layer_i in range(1, num_layers - 1):
        W_i = W_dict[layer_i]
        Z_i = np.matmul(W_i, A_i)
        A_i = tanh_func(Z_i)

    W_L = W_dict[f'W{num_layers}']
    Z_L = np.matmul(W_L, A_i)
    A_L = softmax(Z_L)
    return A_L


def tanh_func(Z_i):
    return np.tanh(Z_i)


def sgd_step(grads, W_old, lr):
    W_new = W_old - lr * grads
    return W_new


def compute_softmax_gradient_vector_respect_to_weights(A_L, W_L, C):
    """
    :param A_L: activation of last layer - dimension: n (num features) x m (num samples)
    :param W_L: weights matrix - dimension: n (neuron current layer) x l (neuron prev layer)
    :param C: classes vector matrix - dimension: l (num labels) x m (num samples)
    :return grads:  gradient with respect to 'W' (same shape as W)
    """
    m = A_L.shape[1]
    num_labels = C.shape[0]
    X_t = np.transpose(A_L)
    denominator = sum([np.exp(np.matmul(X_t, W_L[k])) for k in range(0, num_labels)])
    grads = np.array([(1 / m) * np.matmul(A_L, (np.exp(np.matmul(X_t, W_L[p])) / denominator) - C[p]) for p in
                      range(0, num_labels)])
    return grads

    # grads = []
    # for p in range(0, l):
    #     C_p = C[p]
    #     nominator = np.exp(np.matmul(X_t, W[:, p]))
    #     grads.append((1 / m) * np.matmul(X, (nominator / denominator) - C_p))
    # grads = np.array(grads).transpose()
    # return grads


def softmax_old(A_prev, W_L, with_eta=False):
    """
     :param A_prev: Activation of previous layer - dimension: num_features_prev_layer x m
     :param W_L: weights matrix - dimension: n (neuron current layer) x n-1 (neuron prev layer)
     :return softmax_res: dimensions: n (num features) x m (num samples)
     """
    Z_L_t = A_prev.T
    m = A_prev.shape[1]
    num_labels = W_L.shape[0]
    eta_lst = [max([np.matmul(Z_L_t[i], W_L[k]) for k in range(0, num_labels)]) for i in range(0, m)] if with_eta else [0] * m
    softmax_denominator = sum([np.exp(np.matmul(Z_L_t, W_L[k]) - eta_lst) for k in range(0, num_labels)])
    softmax_res = np.array([np.exp(np.matmul(Z_L_t, W_L[k]) - eta_lst) / softmax_denominator for k in range(0, num_labels)])
    return softmax_res


def softmax(Z):
    """
    :param Z: the linear component of the activation function
    :return A: the activations of the layer
    :return activation_cache: returns Z, which will be useful for the backpropagation
    """
    exp = np.exp(Z)
    A = exp / np.sum(exp, axis=0)[None, :]
    return A, {'Z': Z}


def safe_softmax(Z):
    Z_safe = Z - Z.max(axis=0)
    return softmax(Z_safe)[0], {'Z': Z}


def cross_entropy_softmax_lost(Z_L, W_L, C, with_eta=False):
    """
    computation of lost function
    :param with_eta: use eta for safety
    :param Z_L: Z values of last layer - dimension: n (num features) x m (num samples)
    :param W_L: weights matrix - dimension: n (neuron current layer) x l (neuron prev layer)
    :param C: classes vector matrix - dimension: l (num labels) x m (num samples)
    :return loss: dimensions: scalar
    """
    l = C.shape[0]
    m = Z_L.shape[1]
    X_t = np.transpose(Z_L)
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
