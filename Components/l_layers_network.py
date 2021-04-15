from Components import forward, sgd
from Components.backward import backward_pass
from Utils import data_handler
from Utils.Params import DataSets, HyperParams
from Utils import auxiliary
import pandas as pd

from Utils.auxiliary import plt_acc
from Utils.data_handler import generate_sub_samples


def print_acc(epoch, test_x_samples, train_x_samples, test_y_samples, train_y_samples, wb_dict):
    print("epoch num: ", epoch)
    A_L_test, A_test_dict = forward.forward_pass(test_x_samples, wb_dict)
    A_L_train, A_train_dict = forward.forward_pass(train_x_samples, wb_dict)

    test_accuracy = auxiliary.compute_acc(A_L_test, test_y_samples)
    train_accuracy = auxiliary.compute_acc(A_L_train, train_y_samples)
    print(f'train_accuracy: {train_accuracy}, test accuracy: {test_accuracy}')


def train_complete_network(hidden_layers):
    data_set = DataSets.peaks
    train_x, test_x, train_y, test_y = data_handler.load_data(data_set)  # loading dataset
    train_x_samples, test_x_samples, train_y_samples, test_y_samples = generate_sub_samples(train_x, test_x, train_y,
                                                                                            test_y)  # sub samples

    num_features, num_labels = train_x.shape[0], train_y.shape[0]
    layers_dim = [num_features] + hidden_layers + [num_labels]
    HyperParams.layers_dim = layers_dim
    wb_dict = auxiliary.initiate_wb_dict(layers_dim)  # dict of weights and bias

    for epoch in range(0, HyperParams.num_epochs):
        train_data_batches = data_handler.pre_processing(train_x, train_y, HyperParams.batch_size) # split to batches

        for batch_i in range(0, len(train_data_batches)):
            X_batch, Y_batch = data_handler.initiate_batch(train_data_batches, batch_i, num_features)
            A_L, A_dict = forward.forward_pass(X_batch, wb_dict) #compute forward pass
            grads_dict = backward_pass(wb_dict, A_dict, Y_batch) # update gradients
            wb_dict = sgd.update_wb(grads_dict, wb_dict, HyperParams.learning_rate) # update weights

        if epoch % 5 == 0:
            print_acc(epoch, test_x_samples, train_x_samples, test_y_samples, train_y_samples, wb_dict)


if __name__ == '__main__':
    hidden_layers = [[20, 20, 20, 20, 20], [10, 5, 7], [14, 13, 15], [10, 10, 10, 10], [], [15], [2, 5], [10, 5],
                     [20, 17, 15, 7], [20, 20, 20, 20, 20]]

    for hidden_layer in hidden_layers:
        train_complete_network(hidden_layer)