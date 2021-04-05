from _pytest import unittest
import numpy as np
from numpy.linalg import linalg

from Components import forward


class JacobianTests(unittest.TestCase):


    def test_jacobian(self):
        """
        :param X: data matrix - dimension: n x m
        :param W: weights matrix - dimension: current_layer_features x prev_layer_features
        :param C: classes vector matrix - dimension: l x m
        :param b: bias vector - length: l"""
        X = np.array([[10, 5, 8, 6],
                      [11, 3, 2, 5],
                      [6, 1, 7, 4]])  # n=3,  m=4

        C = np.array([[1, 0, 0, 1],
                      [0, 1, 1, 0]])  # l=2, m=4

        W_0 = np.array([[1, 3, 4],
                        [2, 4, 6]])  # n=2, l=3

        b = np.array([6, 5])

        d = np.random.rand(len(W_0), len(W_0[0]))
        d = d / linalg.norm(d)
        d_f = d.flatten()

        Fw = lambda W: forward.cross_entropy_softmax_lost(X, W, C)
        Fw_delta = lambda W, epsilon: Fw(W) + epsilon * np.matmul(d_f,
                                                                  forward.compute_softmax_gradient_vector_respect_to_weights(
                                                                      X, W, C).flatten()) + epsilon ** 2

        res_sum1 = []
        res_sum2 = []
        eps_0 = 0.001
        for i in range(0, 10):
            epsi = (0.5 ** i) * eps_0
            sum1 = abs(Fw_delta(W_0, epsi) - Fw(W_0))
            sum2 = abs(Fw_delta(W_0, epsi) - Fw(W_0) - epsi * np.matmul(d_f,
                                                                        forward.compute_softmax_gradient_vector_respect_to_weights(
                                                                            X, W_0, C).flatten()))
            res_sum1.append(sum1)
            res_sum2.append(sum2)

        factors_2 = [res_sum1[i] / res_sum1[i + 1] for i in range(0, len(res_sum1) - 1)]
        factors_4 = [res_sum2[i] / res_sum2[i + 1] for i in range(0, len(res_sum2) - 1)]
        print(f'factors 2: {factors_2}')
        print(f'factors 4: {factors_4}')



