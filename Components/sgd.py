def sgd_step(grads, W_old, lr):
    W_new = W_old - lr * grads
    return W_new
