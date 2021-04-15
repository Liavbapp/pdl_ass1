import numpy as np
import matplotlib.pyplot as plt
from Utils.Params import HyperParams


def initiate_wb_dict(layer_dims):
    """
    initiating weights and biases for all layers in the network
    :param layer_dims: an array of the dimensions of each layer in the network
    :return: a dictionary containing the initialized W and b parameters of each layer
    """
    params = {f'W{i + 1}': np.random.randn(layer_dims[i + 1], layer_dims[i]) * np.sqrt(2 / layer_dims[i]) for i in
              range(len(layer_dims) - 1)}
    params.update({f'b{i + 1}': np.random.randn(layer_dims[i + 1], 1) for i in range(len(layer_dims) - 1)})

    return params


def compute_acc(prediction, Y_samples):
    """
    computing the accuracy
    :param prediction: the predicted labels
    :param Y_samples: the ground truth labels
    :return:
    """
    softmax_predictions = np.argmax(prediction, axis=0)
    true_labels_predictions = np.argmax(Y_samples, axis=0)
    accuracy = np.sum(softmax_predictions == true_labels_predictions) / prediction.shape[1]
    return accuracy


def plt_acc(df_train_accuracy, df_test_accuracy, data_set, layers_dim=None):
    """
    The function plots the accuracy of the 1-layer network
    :param df_train_accuracy:
    :param df_test_accuracy:
    :param data_set:
    :return:
    """

    plt.plot(df_train_accuracy['epoch'], df_train_accuracy['acc'], 'g', label='training accuracy')
    plt.plot(df_test_accuracy['epoch'], df_test_accuracy['acc'], 'b', label='validation accuracy')
    title_txt = f'{data_set}, lr={HyperParams.learning_rate}, batch_s={HyperParams.batch_size}'
    if layers_dim is not None:
        title_txt += f' layers dims: {layers_dim}'
    plt.title(title_txt)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

