import random
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from Utils import data_handler
from Components import forward
from Components import backward
from Components import sgd
from Utils.Params import HyperParams, DataSets




def update_weights(X, W, Y):
    grads = backward.softmax_grad_wrt_weights(X, W, Y)
    W_new = sgd.sgd_step(grads, W, HyperParams.learning_rate)
    return W_new


def compute_acc(X_samples, W, Y_samples):
    Z = np.matmul(W, X_samples)
    softmax_output = forward.softmax(Z)
    softmax_predictions = np.argmax(softmax_output, axis=0)
    true_labels_predictions = np.argmax(Y_samples, axis=0)
    accuracy = np.sum(softmax_predictions == true_labels_predictions) / X_samples.shape[1]
    return accuracy


def plt_acc(df_train_accuracy, df_test_accuracy):
    plt.plot(df_train_accuracy['epoch'], df_train_accuracy['acc'], 'g', label='training accuracy')
    plt.plot(df_test_accuracy['epoch'], df_test_accuracy['acc'], 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def sgd_test():
    train_x, test_x, train_y, test_y = data_handler.load_data(DataSets.swiss_roll)  # loading dataset
    train_indices = random.sample(range(0, train_x.shape[1]), int(0.05 * train_x.shape[1]))
    test_indices = random.sample(range(0, test_x.shape[1]), int(0.05 * test_x.shape[1]))
    train_x_samples = train_x[:, train_indices]
    train_x_samples = np.append(train_x_samples, np.ones((1, train_x_samples.shape[1])), axis=0)
    train_y_samples = train_y[:, train_indices]
    test_x_samples = test_x[:, test_indices]
    test_x_samples = np.append(test_x_samples, np.ones((1, test_x_samples.shape[1])), axis=0)
    test_y_samples = test_y[:, test_indices]

    num_features = train_x.shape[0]
    num_labels = train_y.shape[0]
    costs = []

    W_old = np.random.randn(num_labels, num_features + 1) * np.sqrt(
        2 / num_labels)  # the +1 is for the bias neuron feature
    acc_chunks_train = []
    acc_chunks_test = []

    for epoch in range(0, HyperParams.num_epochs):
        if epoch % 1 == 0 and epoch > 0:
            print(epoch)
            train_accuracy = compute_acc(train_x_samples, W_old, train_y_samples)
            test_accuracy = compute_acc(test_x_samples, W_old, test_y_samples)
            acc_chunks_train.append(pd.DataFrame({'acc': train_accuracy, 'epoch': epoch}, index=[0]))
            acc_chunks_test.append(pd.DataFrame({'acc': test_accuracy, 'epoch': epoch}, index=[0]))
            print(f'train acc: {train_accuracy}')
            print(f'test acc: {test_accuracy}')
            cost = forward.cross_entropy_softmax_lost(A_prev=X_batch, W_L=W_old, C=Y_batch)  # TODO: compute lost of last batch only - fix this
            # costs.append(cost)
            print(f'cost: {cost}')

        train_x, test_x = data_handler.add_bias_neuron(train_x, test_x)
        train_data_batches, test_data_batches = data_handler.pre_processing(train_x, test_x, train_y, test_y, HyperParams.batch_size)
        for batch_i in range(0, len(train_data_batches)):
            X_batch, Y_batch = data_handler.initiate_batch(train_data_batches, batch_i, num_features + 1) # added +1 to num_features because of the bias neuron
            W_new = update_weights(X_batch, W_old, Y_batch)
            W_old = W_new

    plt_acc(pd.concat(acc_chunks_train), pd.concat(acc_chunks_test))


if __name__ == '__main__':
    sgd_test()
