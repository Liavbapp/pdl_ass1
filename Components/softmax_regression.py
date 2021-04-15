import numpy as np
import pandas as pd
from Utils import data_handler
from Components import backward
from Components import sgd
from Utils.Params import HyperParams, DataSets
from Utils.auxiliary import plt_acc, compute_acc
from Utils.data_handler import generate_sub_samples


def update_weights(X, W, Y):
    """
    Updating the weights using the sgd_step
    :param X:
    :param W:
    :param Y:
    :return:
    """
    grads = backward.softmax_grad_wrt_weights(X, W, Y)
    W_new = sgd.sgd_step(grads, W, HyperParams.learning_rate)
    return W_new


def prepare_performance_dataframe(acc_chunks_train, acc_chunks_test, train_accuracy, test_accuracy, epoch):
    acc_chunks_train.append(pd.DataFrame({'acc': train_accuracy, 'epoch': epoch}, index=[0]))
    acc_chunks_test.append(pd.DataFrame({'acc': test_accuracy, 'epoch': epoch}, index=[0]))
    print(f'train acc: {train_accuracy}')
    print(f'test acc: {test_accuracy}')


def run_test():
    data_set = DataSets.gmm
    train_x, test_x, train_y, test_y = data_handler.load_data(data_set)  # loading dataset
    train_x, test_x = data_handler.add_bias_neuron(train_x, test_x)  # adding bias neuron
    train_x_samples, test_x_samples, train_y_samples, test_y_samples = generate_sub_samples(train_x, test_x, train_y,
                                                                                            test_y)  # generate sub-samples

    num_features = train_x.shape[0]  # number of features (including the bias neuron also)
    num_labels = train_y.shape[0]
    W_old = np.random.randn(num_labels, num_features) * np.sqrt(2 / num_labels)  # initial weights
    acc_chunks_train = []
    acc_chunks_test = []

    for epoch in range(0, HyperParams.num_epochs):
        if epoch > 0:
            train_accuracy = compute_acc(np.matmul(W_old, train_x_samples), train_y_samples)
            test_accuracy = compute_acc(np.matmul(W_old, test_x_samples), test_y_samples)
            print(f'train acc: {train_accuracy}, test acc: {test_accuracy}')

        train_data_batches = data_handler.pre_processing(train_x, train_y, HyperParams.batch_size)  # split to batches
        for batch_i in range(0, len(train_data_batches)):
            X_batch, Y_batch = data_handler.initiate_batch(train_data_batches, batch_i, num_features) # split to features - labels
            W_new = update_weights(X_batch, W_old, Y_batch) # computing the new weights
            W_old = W_new

    plt_acc(pd.concat(acc_chunks_train), pd.concat(acc_chunks_test), data_set) # plt accuracy graph


run_test()
