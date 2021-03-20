import unittest
import numpy as np

from forward import cross_entropy_softmax_lost


class Tests(unittest.TestCase):

    def test_cross_entropy_softmax_lost(self):
        """
        computation of lost function
        :param X: data matrix - dimension: n x m
        :param C: classes vector matrix - dimension: m x l
        :param W: weights matrix - dimension: n x l
        :param num_classes:
        :return:
        """
        X = np.array([[1, 2, 3],
                      [3, 4, 4]])

        C = np.array([[0, 1],
                      [1, 0],
                      [0, 1]])

        W = np.array([[1, 2],
                      [2, 1]])

        l = len(C[0])  # why l?
        m = len(X[0])
        X_t = np.transpose(X)
        eta = max([np.matmul(X_t[i], W[:, i]) for i in range(0, l)]) if with_eta else 0
        softmax_denominator = np.sum(np.exp(np.matmul(X_t, W[:, j]) - eta) for j in range(0, l))
        loss = (-1 / m) * np.sum(np.matmul(np.transpose(C[:, k]), np.log(np.exp(np.matmul(X_t, W[:, k])))) for k in range(0, l))  # should deviate
        return loss

        return cross_entropy_softmax_lost(X, C, W)
