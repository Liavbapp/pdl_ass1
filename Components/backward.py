import numpy as np


def backward_pass(A_L):


    pass

def backward(A_i, func):
    if func == "softmax":
        compute_softmax_gradient_vector_respect_to_weights(X,W,C)
        return


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


def compute_softmax_gradient_vector_respect_to_data(A, W, C):
    """
    :param A: nXm
    :param W: current_layer_features X prev_layer_features(n)
    :param C: l (num labels) x m (num samples)
    :return:
    """
    nominator = np.exp(A)
    denominator = np.sum(np.exp(np.matmul(W[k], A)) for k in range(0, W.shape[0]))


    # nominator = np.exp(np.matmul(W, A))
    # denominator = np.sum(np.exp(np.matmul(W[k], A)) for k in range(0, W.shape[0]))
    # return 1/A.shape[1] * (np.matmul(W.T, nominator/denominator - C))
