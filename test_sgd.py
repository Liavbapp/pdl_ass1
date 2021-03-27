import time

from scipy.io import loadmat
import numpy as np

import forward


def load_data(data_set):
    data = loadmat(f'Data/{data_set}.mat')
    train_x = data['Yt']
    test_x = data['Yv']
    train_y = data['Ct']
    test_y = data['Cv']

    return train_x, test_x, train_y, test_y


def add_bias_neuron(train_x, test_x):
    # adding ones vector to data (for bias)
    train_ones_v = np.ones((1, train_x.shape[1]))
    test_one_v = np.ones((1, test_x.shape[1]))
    train_x = np.concatenate([train_x, train_ones_v], axis=0)
    test_x = np.concatenate([test_x, test_one_v], axis=0)
    return train_x, test_x


def shuffle_data(train_x, test_x, train_y, test_y):
    train_data = np.concatenate([train_x, train_y])
    test_data = np.concatenate([test_x, test_y])
    np.random.shuffle(np.transpose(train_data))
    np.random.shuffle(np.transpose(test_data))
    return train_data, test_data


def pre_processing(train_x, test_x, train_y, test_y, batch_size):
    train_x, test_x = add_bias_neuron(train_x, test_x)
    train_data_shf, test_data_shf = shuffle_data(train_x, test_x, train_y, test_y)
    train_data_batches = np.array_split(train_data_shf, indices_or_sections=batch_size, axis=1)
    test_data_batches = np.array_split(test_data_shf, indices_or_sections=batch_size, axis=1)
    # train_x_shf, train_y_shf = tuple(train_data_split)
    return train_data_batches, test_data_batches


def sgd_test():
    data_sets = ['SwissRollData', 'GMMData', 'PeaksData.mat']
    train_x, test_x, train_y, test_y = load_data(data_sets[0])  # loading dataset
    num_features = train_x.shape[0]
    num_labels = train_y.shape[0]


    # Hyper-Params
    batches_sizes = [2 ** i for i in range(0, 7)]
    # lrs = [1 * (1/10)**i for i in range(2, 6)]
    lr = 0.00001
    num_epochs = 100
    costs = []
    train_accuracy = 0

    W_old = np.random.rand(num_features + 1, num_labels)
    for epoch in range(0, num_epochs):
        train_data_batches, test_data_batches = pre_processing(train_x, test_x, train_y, test_y, 32)
        print(train_accuracy)
        for batch in range(0, len(train_data_batches)):
            X_batch = train_data_batches[batch][0:num_features + 1, :]  # all the features rows including bias
            Y_batch = train_data_batches[batch][num_features + 1:, :]  # the label rows
            Y_batch = Y_batch.transpose()
            cost = forward.cross_entropy_softmax_lost(X_batch, Y_batch, W_old)
            costs.append(cost)
            grads = forward.compute_softmax_gradient_vector_respect_to_weights(X_batch, W_old, Y_batch)
            W_new = forward.sgd_step(grads, W_old, lr)
            softmax_output = forward.softmax(X_batch, Y_batch, W_new, with_eta=False)
            soft_max_predictions = np.argmax(softmax_output, axis=1)
            true_labels_predictions = np.argmax(Y_batch, axis=1)
            train_accuracy = np.sum(soft_max_predictions == true_labels_predictions) / X_batch.shape[1]
            W_old = W_new


sgd_test()
