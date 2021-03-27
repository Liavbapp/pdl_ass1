import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import pandas as pd
import forward
from Params import HyperParams, DataSets
import plotly.express as px


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


def initiate_batch(all_batches, batch_num, num_features):
    x_batch = all_batches[batch_num][0:num_features + 1, :]  # all the features rows including bias
    y_batch = all_batches[batch_num][num_features + 1:, :]  # the label rows
    y_batch = y_batch.transpose()
    return x_batch, y_batch


def update_weights(X, Y, W_old):
    grads = forward.compute_softmax_gradient_vector_respect_to_weights(X, W_old, Y)
    W_new = forward.sgd_step(grads, W_old, HyperParams.learning_rate)
    return W_new


def compute_acc(X, Y, W, with_eta=False):
    softmax_output = forward.softmax(X, Y, W, with_eta=with_eta)
    soft_max_predictions = np.argmax(softmax_output, axis=1)
    true_labels_predictions = np.argmax(Y, axis=1)
    train_accuracy = np.sum(soft_max_predictions == true_labels_predictions) / X.shape[1]
    return train_accuracy


def plt_acc(train_acc_lst):
    # TODO: add validation acc
    df_acc = pd.DataFrame({'train_acc': train_acc_lst})
    df_acc = pd.DataFrame({'train_acc': train_acc_lst})
    # acc_val = df_acc.history['val_loss']
    epochs = range(1, 35)
    plt.plot(df_acc.index, df_acc['train_acc'], 'g', label='Training acc')
    # plt.plot(epochs, loss_val, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def sgd_test():
    train_x, test_x, train_y, test_y = load_data(DataSets.swiss_roll)  # loading dataset
    num_features = train_x.shape[0]
    num_labels = train_y.shape[0]
    costs = []
    train_acc_lst = []
    test_acc = []
    W_old = np.random.rand(num_features + 1, num_labels)

    for epoch in range(0, HyperParams.num_epochs):
        print(epoch)
        train_data_batches, test_data_batches = pre_processing(train_x, test_x, train_y, test_y, HyperParams.batch_size)
        train_acc_lst.append(
            train_acc) if epoch > 0 else None  # TODO: it is only the last batch need to figureout what to do here
        for batch_num in range(0, len(train_data_batches)):
            X_batch, Y_batch = initiate_batch(train_data_batches, batch_num, num_features)
            cost = forward.cross_entropy_softmax_lost(X_batch, Y_batch, W_old)
            W_new = update_weights(X_batch, Y_batch, W_old)
            train_acc = compute_acc(X_batch, Y_batch, W_new, with_eta=False)
            costs.append(cost)
            W_old = W_new

    plt_acc(train_acc_lst)


sgd_test()
