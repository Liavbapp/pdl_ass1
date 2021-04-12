def sgd_step(grads, old_param, lr):
    new_param = old_param - lr * grads
    return new_param


def update_wb(grads_dict, WB_dict, lr):
    num_layers = len(WB_dict.keys()) // 2
    for i in range(num_layers, 0, -1):
        WB_dict[f'W{i}'] = sgd_step(grads_dict[f'grads{i}']["grad_w"], WB_dict[f'W{i}'], lr)
        WB_dict[f'b{i}'] = sgd_step(grads_dict[f'grads{i}']["grad_b"], WB_dict[f'b{i}'], lr)

    return WB_dict