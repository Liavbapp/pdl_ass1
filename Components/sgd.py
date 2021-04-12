
def sgd_step(grads, old_param, lr):
    """
    performing the sgd step
    :param grads: gradients
    :param old_param:
    :param lr: learning rate
    :return:
    """
    new_param = old_param - lr * grads
    return new_param


def update_wb(grads_dict, WB_dict, lr):
    """
    Update the dictionary of the weights and biases for all the layers in the network
    :param grads_dict: dictionary with the gradients for each layer in the network
    :param WB_dict: dictionary with the weight and biases for each layer in the network
    :param lr: learning rate
    :return WB_dict: the WB_dict with the new weights and biases
    """
    num_layers = len(WB_dict.keys()) // 2
    for i in range(num_layers, 0, -1):
        WB_dict[f'W{i}'] = sgd_step(grads_dict[f'grads{i}']["grad_w"], WB_dict[f'W{i}'], lr)
        WB_dict[f'b{i}'] = sgd_step(grads_dict[f'grads{i}']["grad_b"], WB_dict[f'b{i}'], lr)

    return WB_dict
