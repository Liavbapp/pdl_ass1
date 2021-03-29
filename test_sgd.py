import random
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
import pandas as pd
import forward
from Params import HyperParams, DataSets



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
    return x_batch, y_batch


def update_weights(X,  W, Y):
    grads = forward.compute_softmax_gradient_vector_respect_to_weights(X, W, Y)
    W_new = forward.sgd_step(grads, W, HyperParams.learning_rate)
    return W_new


def compute_acc(X_samples, W, Y_samples, with_eta=False):
    softmax_output = forward.softmax(X_samples, W,  Y_samples, with_eta=with_eta)
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
    train_x, test_x, train_y, test_y = load_data(DataSets.swiss_roll)  # loading dataset
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

    W_old = np.random.randn(num_labels, num_features + 1) * np.sqrt(2 / num_labels) # the +1 is for the bias neuron feature
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
            cost = forward.cross_entropy_softmax_lost(X_batch, W_old, Y_batch) #TODO: compute lost of last batch only - fix this
            # costs.append(cost)
            print(f'cost: {cost}')

        train_data_batches, test_data_batches = pre_processing(train_x, test_x, train_y, test_y, HyperParams.batch_size)
        for batch_i in range(0, len(train_data_batches)):
            X_batch, Y_batch = initiate_batch(train_data_batches, batch_i, num_features)
            W_new = update_weights(X_batch, W_old, Y_batch)
            W_old = W_new

    plt_acc(pd.concat(acc_chunks_train), pd.concat(acc_chunks_test))


if __name__ == '__main__':
    sgd_test()
