import unittest
import numpy as np
import torch
from numpy import linalg

import forward


class GradTests(unittest.TestCase):

    def test_a(self):
        """
        computation of lost function
        :param with_eta: use eta for safety
        :param X: data matrix - dimension: n x m
        :param C: classes vector matrix - dimension: l x m
        :param W: weights matrix - dimension: n x l
        :param b: bias vector - length: l
        :param num_classes:
        :return:
        """
        # v = np.random.rand(2)
        # v_hat = v / linalg.norm(v)

        X = np.array([[1, 0.3, 0.5, 0.4, 0.9]])
        X_T = X.transpose()
        m = len(X)
        c1 = np.array([[1],
                       [0],
                       [1],
                       [0],
                       [1]])
        c1_t = c1.transpose()
        c2 = np.array([[0],
                       [1],
                       [0],
                       [1],
                       [0]])
        c2_t = c2.transpose()
        b = np.array([0, 0])



        Fw = lambda W: (-1 / m) * (np.matmul(c1_t, np.log(self.sigmoid(np.matmul(X_T, W))))
                                   + np.matmul(c2_t, np.log(1 - self.sigmoid(np.matmul(X_T, W)))))

        grad_Fw = lambda W: (1 / m) * np.matmul(X, self.sigmoid(np.matmul(X_T, W)) - c1)

        fw_delta = lambda W, eps: Fw(W) + eps * np.matmul(d_t, grad_Fw(W)) + eps**2

        d = np.array([[0.7336602]])
        d_t = d.transpose()


        W_0 = np.array([[1]])

        res_sum1 = []
        res_sum2 = []
        eps_0 = 0.005
        for i in range(0, 10):
            epsi = (0.5 ** i) * eps_0
            sum1 = abs(fw_delta(W_0, epsi) - Fw(W_0))
            res_sum1.append(sum1)
            sum2 = abs(fw_delta(W_0, epsi) - Fw(W_0) - epsi * np.matmul(d_t, grad_Fw(W_0)))
            res_sum2.append(sum2)
        # print(f'series 1: {res_sum1}')
        # print(f'series 2: {res_sum2}')
        factors_2 = [res_sum1[i] / res_sum1[i + 1] for i in range(0, len(res_sum1) - 1)]
        factors_4 = [res_sum2[i] / res_sum2[i + 1] for i in range(0, len(res_sum2) - 1)]
        print(f'factors 2: {factors_2}')
        print(f'factors 4: {factors_4}')



        stop= 1
        # grad_f = forward.compute_softmax_gradient_vector_respect_to_weights(d, w, c, b)
        # a = 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class JacobianTests(unittest.TestCase):
    pass
