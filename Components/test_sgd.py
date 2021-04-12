import random
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from Utils import data_handler
from Components import forward
from Components import backward
from Components import sgd
from Utils.Params import HyperParams, DataSets


def generate_sub_samples(train_x, test_x, train_y, test_y):
    random.seed(42)
    train_indices = random.sample(range(0, train_x.shape[1]), int(0.1 * train_x.shape[1]))
    test_indices = random.sample(range(0, test_x.shape[1]), int(0.8 * test_x.shape[1]))
    train_x_samples = train_x[:, train_indices]
    train_y_samples = train_y[:, train_indices]
    test_x_samples = test_x[:, test_indices]
    test_y_samples = test_y[:, test_indices]
    return train_x_samples, test_x_samples, train_y_samples, test_y_samples


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


def plt_acc(df_train_accuracy, df_test_accuracy, data_set):
    fig = plt.figure()
    ax = fig.add_subplot()

    plt.plot(df_train_accuracy['epoch'], df_train_accuracy['acc'], 'g', label='training accuracy')
    plt.plot(df_test_accuracy['epoch'], df_test_accuracy['acc'], 'b', label='validation accuracy')
    plt.title(f'{data_set} Dataset,  Learning Rate={HyperParams.learning_rate}, Batch Size={HyperParams.batch_size}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # file_name = f"lr={str(HyperParams.learning_rate).replace('.', '_')}_batch_size={str(HyperParams.batch_size)}.png"
    # path = r'C:\Users\LiavB\OneDrive\Desktop\Msc\Semester_B\Practical Deep Learning\Ass\Ass1\Graphs\sgd_test\GMM'
    # plt.savefig(path + f'\\{file_name}')

    plt.show()




def run_test():
    data_set = DataSets.gmm
    train_x, test_x, train_y, test_y = data_handler.load_data(data_set)  # loading dataset
    train_x, test_x = data_handler.add_bias_neuron(train_x, test_x)
    train_x_samples, test_x_samples, train_y_samples, test_y_samples = generate_sub_samples(train_x, test_x, train_y,
                                                                                            test_y)

    num_features = train_x.shape[0]  # num features include the bias neuron
    num_labels = train_y.shape[0]
    W_old = np.random.randn(num_labels, num_features) * np.sqrt(2 / num_labels)  # +1 for the bias neuron
    acc_chunks_train = []
    acc_chunks_test = []

    for epoch in range(0, HyperParams.num_epochs):
        if epoch > 0:
            print(epoch)
            train_accuracy = compute_acc(train_x_samples, W_old, train_y_samples)
            test_accuracy = compute_acc(test_x_samples, W_old, test_y_samples)
            acc_chunks_train.append(pd.DataFrame({'acc': train_accuracy, 'epoch': epoch}, index=[0]))
            acc_chunks_test.append(pd.DataFrame({'acc': test_accuracy, 'epoch': epoch}, index=[0]))
            print(f'train acc: {train_accuracy}')
            print(f'test acc: {test_accuracy}')

        train_data_batches, test_data_batches = data_handler.pre_processing(train_x, test_x, HyperParams.batch_size)
        for batch_i in range(0, len(train_data_batches)):

            X_batch, Y_batch = data_handler.initiate_batch(train_data_batches, batch_i, num_features)
            W_new = update_weights(X_batch, W_old, Y_batch)
            W_old = W_new

    plt_acc(pd.concat(acc_chunks_train), pd.concat(acc_chunks_test), data_set)


run_test()

# for i in range(0, 4):
#     HyperParams.learning_rate = 0.01 / (10 ** i)
#     for j in range(3, 9):
#         HyperParams.batch_size = 2**j
#         run_test()
