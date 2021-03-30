import numpy as np


def sgd_step(grads, W_old, lr):
    W_new = W_old - lr * grads
    return W_new


def compute_softmax_gradient_vector_respect_to_weights(X, W, C):
    """
    :param X: data matrix - dimension: n (num features) x m (num samples)
    :param W: weights matrix - dimension: n (neuron current layer) x l (neuron prev layer)
    :param C: classes vector matrix - dimension: l (num labels) x m (num samples)
    :return grads:  gradient with respect to 'W' (same shape as W)
    """
    m = X.shape[1]
    num_labels = C.shape[0]
    X_t = np.transpose(X)
    denominator = sum([np.exp(np.matmul(X_t, W[k])) for k in range(0, num_labels)])
    grads = np.array([(1 / m) * np.matmul(X, (np.exp(np.matmul(X_t, W[p])) / denominator) - C[p]) for p in
                      range(0, num_labels)])
    return grads

    # grads = []
    # for p in range(0, l):
    #     C_p = C[p]
    #     nominator = np.exp(np.matmul(X_t, W[:, p]))
    #     grads.append((1 / m) * np.matmul(X, (nominator / denominator) - C_p))
    # grads = np.array(grads).transpose()
    # return grads


def softmax(X, W, C, with_eta=False):
    """
     :param X: data matrix - dimension: n (num features) x m (num samples)
     :param W: weights matrix - dimension: n (neuron current layer) x l (neuron prev layer)
     :param C: classes vector matrix - dimension: l (num labels) x m (num samples)
     :return softmax_res: dimensions: n (num features) x m (num samples)
     """
    X_t = X.T
    m = X.shape[1]
    l = C.shape[0]
    eta_lst = [max([np.matmul(X_t[i], W[k]) for k in range(0, l)]) for i in range(0, m)] if with_eta else [0] * m
    softmax_denominator = sum([np.exp(np.matmul(X_t, W[k]) - eta_lst) for k in range(0, l)])
    softmax_res = np.array([np.exp(np.matmul(X_t, W[k]) - eta_lst) / softmax_denominator for k in range(0, l)])
    return softmax_res


def cross_entropy_softmax_lost(X, W, C, with_eta=False):
    """
    computation of lost function
    :param with_eta: use eta for safety
    :param X: data matrix - dimension: n (num features) x m (num samples)
    :param W: weights matrix - dimension: n (neuron current layer) x l (neuron prev layer)
    :param C: classes vector matrix - dimension: l (num labels) x m (num samples)
    :return loss: dimensions: scalar
    """
    l = C.shape[0]
    m = X.shape[1]
    X_t = np.transpose(X)
    eta_lst = [max([np.matmul(X_t[i], W[k]) for k in range(0, l)]) for i in range(0, m)] if with_eta else [0] * m
    softmax_denominator = sum([np.exp(np.matmul(X_t, W[k]) - eta_lst) for k in range(0, l)])
    loss = 0
    for k in range(0, l):
        softmax_prob = np.log(np.exp(np.matmul(X_t, W[k]) - eta_lst) / softmax_denominator)
        loss += np.matmul(C[k], softmax_prob)
    loss *= (-1 / m)

    # loss = (-1 / m) * sum(
    #     np.matmul(C[k], np.log(np.exp(np.matmul(X_t, W[:, k]) - eta_lst) / softmax_denominator)) for k
    #     in range(0, l))

    return loss
