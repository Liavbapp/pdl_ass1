import numpy as np


def compute_softmax_gradient_vector_respect_to_data(X, W, C, b):
    """
    :param X: data matrix - dimension: n x m
    :param W: weights matrix - dimension: n x l
    :param C: classes vector matrix - dimension: l x m
    :param b: bias vector - length: l
    :return:
    """
    m = len(X[0])
    l = len(C)
    W_t = np.transpose(W)
    b_t = np.expand_dims(b, 1)
    nominator = np.exp(np.matmul(W_t, X) + b_t)
    dominator = sum(np.exp(np.matmul(np.transpose(W[:, k]), X) + b[k]) for k in range(0, l))
    grads = (1 / m) * np.matmul(W, (nominator / dominator) - C)
    return grads


def compute_softmax_gradient_vector_respect_to_weights(X, W, C, b):
    """
    :param X: data matrix - dimension: n x m
    :param W: weights matrix - dimension: n x l
    :param C: classes vector matrix - dimension: l x m
    :param b: bias vector - length: l
    :return:
    """
    m = len(X[0])
    l = len(C)
    X_t = np.transpose(X)
    denominator = sum([np.exp(np.matmul(X_t, W[:, k]) + b[k]) for k in range(0, l)])
    grads = np.array([(1 / m) * np.matmul(X, (np.exp(np.matmul(X_t, W[:, p]) + b[p]) / denominator) - C[p]) for p in
                      range(0, l)]).transpose()
    # grads = []
    # for p in range(0, l):
    #     C_p = C[p]
    #     nominator = np.exp(np.matmul(X_t, W[:, p]) + b[p])
    #     grads.append((1 / m) * np.matmul(X, (nominator / denominator) - C_p))
    # grads = np.array(grads).transpose()
    # return grads
    return grads


def cross_entropy_softmax_lost(X, C, W, b, with_eta=False):
    """
    computation of lost function
    :param with_eta: use eta for safety
    :param X: data matrix - dimension: n x m
    :param C: classes vector matrix - dimension: m x l
    :param W: weights matrix - dimension: n x l
    :param b: bias vector - length: l
    :param num_classes:
    :return:
    """
    l = len(C)
    m = len(X[0])
    X_t = np.transpose(X)
    eta_lst = [max([np.matmul(X_t[i], W[:, k]) + b[k] for k in range(0, l)]) for i in range(0, m)] if with_eta else [
                                                                                                                        0] * m
    softmax_denominator = sum([np.exp(np.matmul(X_t, W[:, k]) + b[k] - eta_lst) for k in range(0, l)])
    loss = 0
    for k in range(0, l):
        softmax_prob = np.log(np.exp(np.matmul(X_t, W[:, k]) + b[k] - eta_lst) / softmax_denominator)
        loss += np.matmul(C[k], softmax_prob)
    loss *= (-1 / m)

    # loss = (-1 / m) * sum(
    #     np.matmul(C[k], np.log(np.exp(np.matmul(X_t, W[:, k]) + b[k] - eta_lst) / softmax_denominator)) for k
    #     in range(0, l))

    return loss
