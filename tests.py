import unittest
import numpy as np

import forward

from forward import cross_entropy_softmax_lost


class Tests(unittest.TestCase):

    def test_compute_softmax_gradient_vector_respect_to_weights(self):
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

        W = np.array([[2, 3],
                      [5, 1]])

        b = np.array([0, 0])

        expected_grads = np.array([[1.33326871, 2.33321921],
                                   [-1.33326871, - 2.33321921]])
        actual_grads = forward.compute_softmax_gradient_vector_respect_to_weights(X, W, C, b)
        print(forward.compute_softmax_gradient_vector_respect_to_weights(X, W, C, b))
        #self.assertTrue(expected_grads == actual_grads)

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
                      [1, 0, 1]])

        W = np.array([[1, 3],
                      [5, 1]])

        b = np.array([0, 0])

        expected_loss = 6.666698980663971
        actual_loss = forward.cross_entropy_softmax_lost(X, C, W, b, with_eta=False)
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
        actual_loss = forward.cross_entropy_softmax_lost(X, C, W, b, with_eta=True)
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
        actual_loss = forward.cross_entropy_softmax_lost(X, C, W, b, with_eta=True)
        self.assertTrue(expected_loss == actual_loss)
