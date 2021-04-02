from Components import forward, sgd
from Components.backward import backward_pass
from Utils import data_handler
from Utils.Params import DataSets, HyperParams
from Utils import auxiliary


def main():
    train_x, test_x, train_y, test_y = data_handler.load_data(DataSets.swiss_roll)  # loading dataset

    num_features = train_x.shape[0]
    num_labels = train_y.shape[0]
    layers_dim = [num_features, 20, 10, 7, num_labels]
    wb_dict = auxiliary.initiate_wb_dict(layers_dim)  # dict of weights and bias

    for epoch in range(0, HyperParams.num_epochs):
        # TODO: what to do with the bias neuron of last layer / how to derviate the last layer w.e.t bias?
        train_data_batches, test_data_batches = data_handler.pre_processing(train_x, test_x, train_y, test_y,
                                                                            HyperParams.batch_size)
        print("epoch num: ", epoch)
        for batch_i in range(0, len(train_data_batches)):
            X_batch, Y_batch = data_handler.initiate_batch(train_data_batches, batch_i, num_features)
            A_L, A_dict = forward.forward_pass(X_batch, wb_dict)
            grads_dict = backward_pass(A_L, wb_dict, A_dict, Y_batch)
            wb_dict = sgd.update_wb(grads_dict, wb_dict, HyperParams.learning_rate)

        print("accuracy: ", auxiliary.compute_acc(A_L, Y_batch))



            # W_new = update_weights(X_batch, W_old, Y_batch)
            # W_old = W_new


if __name__ == '__main__':
    main()
