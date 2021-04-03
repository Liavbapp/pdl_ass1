from scipy.io import loadmat
import numpy as np



def load_data(data_set):
    data = loadmat(f'../Data/{data_set}.mat')
    train_x = data['Yt']
    test_x = data['Yv']
    train_y = data['Ct']
    test_y = data['Cv']

    return train_x, test_x, train_y, test_y


def shuffle_data(train_x, test_x, train_y, test_y):
    train_data = np.concatenate([train_x, train_y])
    test_data = np.concatenate([test_x, test_y])
    # np.random.shuffle(np.transpose(train_data))
    # np.random.shuffle(np.transpose(test_data))
    return train_data, test_data


def initiate_batch(all_batches, batch_num, num_features):
    x_batch = all_batches[batch_num][0:num_features, :]  # all the features rows including bias
    y_batch = all_batches[batch_num][num_features:, :]  # the label rows
    return x_batch, y_batch


def add_bias_neuron(train_x, test_x):
    # adding ones vector to data (for bias)
    train_ones_v = np.ones((1, train_x.shape[1]))
    test_one_v = np.ones((1, test_x.shape[1]))
    train_x = np.concatenate([train_x, train_ones_v], axis=0)
    test_x = np.concatenate([test_x, test_one_v], axis=0)
    return train_x, test_x


def pre_processing(train_x, test_x, train_y, test_y, batch_size):
    train_data_shf, test_data_shf = shuffle_data(train_x, test_x, train_y, test_y)
    train_data_batches = np.array_split(train_data_shf, indices_or_sections=batch_size, axis=1)
    test_data_batches = np.array_split(test_data_shf, indices_or_sections=batch_size, axis=1)
    # train_x_shf, train_y_shf = tuple(train_data_split)
    return train_data_batches, test_data_batches
