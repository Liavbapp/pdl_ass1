import numpy as np

#
# def softmax(X, W, j, num_classes):
#     """
#     :param num_classes: number of classification class
#     :param j: current classification class
#     :param X: data matrix - dimension: n x m
#     :param W: weights matrix : dimension: n x l
#     :return prob: probability that X belongs to class j
#     """
#     X_t = np.transpose(X)
#     numerator = np.exp(np.matmul(X_t, W[:, j]))
#     prob = numerator / denominator
#     return prob
#
#
# def safe_softmax(X, W, j, num_classes, eta):
#     """
#     :param eta: max(X_t*w_i for i in {0,1,... num_classes})
#     :param num_classes: number of classification class
#     :param j: current classification class
#     :param X: matrix of input data - dimension: n x m
#     :param W: matrix of weights vectors : dimension: n x l
#     :return prob: probability that X belongs to class j
#     """
#     X_t = np.transpose(X)
#     numerator = np.exp(np.matmul(X_t, W[:, j]) - eta)
#     prob = numerator / softmax_denominator
#     return prob
#

def cross_entropy_softmax_lost(X, C, W, with_eta=False):
    """
    computation of lost function
    :param X: data matrix - dimension: n x m
    :param C: classes vector matrix - dimension: m x l
    :param W: weights matrix - dimension: n x l
    :param num_classes:
    :return:
    """
    l = len(C)
    m = len(X[0])
    X_t = np.transpose(X)
    eta = max([np.matmul(X_t, W[:, i]) for i in range(0, l)]) if with_eta else 0
    softmax_denominator = np.sum(np.exp(np.matmul(X_t, W[:, j]) - eta) for j in range(0, l))
    loss = (-1 / m) * np.sum(np.matmul(np.transpose(C[:, k]), np.log(np.exp(np.matmul(X_t, W[:, k])))) for k in range(0, l))