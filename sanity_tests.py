import unittest
import numpy as np

import forward_old

from forward_old import cross_entropy_softmax_lost


class Tests(unittest.TestCase):

    def test_SGD_softmax(self):
        X = np.array([[10, 5, 8],
                      [11, 3, 2],
                      [6, 1, 7]])  # n=3,  m=3

        C = np.array([[1, 0],
                      [0, 1],
                      [1, 0]])  # l=2, m=3

        W = np.array([[1, 3],
                        [2, 4],
                        [4, 6]])  # n=3, l=2

        b = np.array([0, 0])

        forward_old.SGD_softmax(X, W, C, b, lr=1e-05)

    def test_compute_softmax_gradient_vector_respect_to_weights(self):
        """
        computation of lost function
        :param X: data matrix - dimension: n x m
        :param C: classes vector matrix - dimension: m x l
        :param W: weights matrix - dimension: n x l
        :param b: bias vector - length l
        :param num_classes:
        :return:
        """
        X = np.array([[10, 5, 8],
                      [11, 3, 2],
                      [6, 1, 7]])  # n=3,  m=3

        C = np.array([[1, 0],
                      [0, 1],
                      [1, 0]]) .transpose() # m=3, l=2

        W = np.array([[1, 3],
                      [2, 4],
                      [4, 6]]) # n=3, l=2

        b = np.array([0, 0])

        actual_grads = forward_old.compute_softmax_gradient_vector_respect_to_weights(X, W, C, b)
        print(actual_grads)

    def test_cross_entropy_softmax_lost_no_eta(self):
        """
        computation of lost function
        :param X: data matrix - dimension: n x m
        :param C: classes vector matrix - dimension: l x m
        :param W: weights matrix - dimension: n x l
        :param num_classes:
        :return:
        """
        X = np.array([[1, 2, 3],
                      [3, 4, 4]])

        C = np.array([[0, 1, 0],
                      [1, 0, 1]]).transpose()

        W = np.array([[1, 3],
                      [5, 1]])

        b = np.array([0, 0])

        expected_loss = 6.666698980663971
        actual_loss = forward_old.cross_entropy_softmax_lost(X, C, W, b, with_eta=False)
        self.assertTrue(expected_loss == actual_loss)



    def test_cross_entropy_softmax_lost_with_eta(self):
        """
        computation of lost function
        :param X: data matrix - dimension: n x m
        :param C: classes vector matrix - dimension: l x m
        :param W: weights matrix - dimension: n x l
        :param num_classes:
        :return:
        """
        X = np.array([[1, 2, 3],
                      [3, 4, 4]])

        C = np.array([[0, 1, 0],
                      [1, 0, 1]])

        W = np.array([[1, 3],
                      [5, 1]])

        b = np.array([0, 0])

        expected_loss = 6.666698980663971
        actual_loss = forward_old.cross_entropy_softmax_lost(X, C, W, b, with_eta=True)
        self.assertTrue(expected_loss == actual_loss)

    def test_cross_entropy_softmax_lost_with_bias(self):
        """
        computation of lost function
        :param X: data matrix - dimension: n x m
        :param C: classes vector matrix - dimension: l x m
        :param W: weights matrix - dimension: n x l
        :param num_classes:
        :return:
        """
        X = np.array([[1, 2, 3],
                      [3, 4, 4]])

        C = np.array([[0, 1, 0],
                      [1, 0, 1]])

        W = np.array([[1, 3],
                      [5, 1]])

        b = np.array([1, 4])

        expected_loss = 4.667315445032424
        actual_loss = forward_old.cross_entropy_softmax_lost(X, C, W, b, with_eta=True)
        self.assertTrue(expected_loss == actual_loss)
