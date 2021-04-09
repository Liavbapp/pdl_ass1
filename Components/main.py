from Components import forward, sgd
from Components.backward import backward_pass
from Utils import data_handler
from Utils.Params import DataSets, HyperParams
from Utils import auxiliary


def main():
    train_x, test_x, train_y, test_y = data_handler.load_data(DataSets.swiss_roll)  # loading dataset

    num_features = train_x.shape[0]
    num_labels = train_y.shape[0]
    layers_dim = [num_features, 20, 20, 20, 20, 20, num_labels]
    wb_dict = auxiliary.initiate_wb_dict(layers_dim)  # dict of weights and bias

    for epoch in range(0, HyperParams.num_epochs):
        # TODO: what to do with the bias neuron of last layer / how to derviate the last layer w.e.t bias?
        train_data_batches, test_data_batches = data_handler.pre_processing(train_x, test_x, train_y, test_y,
                                                                            HyperParams.batch_size)
        # print("epoch num: ", epoch)
        for batch_i in range(0, len(train_data_batches)):
            X_batch, Y_batch = data_handler.initiate_batch(train_data_batches, batch_i, num_features, epoch)
            A_L, A_dict = forward.forward_pass(X_batch, wb_dict)
            grads_dict = backward_pass(A_L, wb_dict, A_dict, Y_batch)
            wb_dict = sgd.update_wb(grads_dict, wb_dict, HyperParams.learning_rate)

        if epoch % 1 == 0:
            print("epoch num: ", epoch)
            last_layer = len(layers_dim)
            A_L_test, A_test_dict = forward.forward_pass(test_x, wb_dict)
            A_L_train, A_train_dict = forward.forward_pass(train_x, wb_dict)
            train_loss = forward.cross_entropy_softmax_lost(A_train_dict[f'A{last_layer - 2}'],
                                                            wb_dict[f'W{last_layer - 1}'], train_y)
            validation_loss = forward.cross_entropy_softmax_lost(A_test_dict[f'A{last_layer - 2}'],
                                                                 wb_dict[f'W{last_layer - 1}'], test_y)
            test_accuracy = auxiliary.compute_acc(A_L_test, test_y)
            print(
                f"train loss: {train_loss} \n validation loss: {validation_loss}\n test accuracy: {test_accuracy} lr: {HyperParams.learning_rate}")


if __name__ == '__main__':
    main()
