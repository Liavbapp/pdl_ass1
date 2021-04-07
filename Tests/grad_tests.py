import unittest
import numpy as np
from numpy import linalg

import Components.forward as forward
from Components import backward


class GradTests(unittest.TestCase):

    def test_grad(self):
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
                                                                  backward.compute_softmax_gradient_vector_respect_to_weights(
                                                                      X, W, C).flatten()) + epsilon ** 2

        res_sum1 = []
        res_sum2 = []
        eps_0 = 0.001
        for i in range(0, 10):
            epsi = (0.5 ** i) * eps_0
            sum1 = abs(Fw_delta(W_0, epsi) - Fw(W_0))
            sum2 = abs(Fw_delta(W_0, epsi) - Fw(W_0) - epsi * np.matmul(d_f,
                                                                        backward.compute_softmax_gradient_vector_respect_to_weights(
                                                                            X, W_0, C).flatten()))
            res_sum1.append(sum1)
            res_sum2.append(sum2)

        factors_2 = [res_sum1[i] / res_sum1[i + 1] for i in range(0, len(res_sum1) - 1)]
        factors_4 = [res_sum2[i] / res_sum2[i + 1] for i in range(0, len(res_sum2) - 1)]
        print(f'factors 2: {factors_2}')
        print(f'factors 4: {factors_4}')



    def test_grad_for_JTMV_X(self):
        """
        :param X: data matrix - dimension: n x m
        :param W: weights matrix - dimension: current_layer_features x prev_layer_features
        :param C: classes vector matrix - dimension: l x m
        :param b: bias vector - length: l"""

        X = np.random.randn(3, 4)

        W_0 = np.array([[1, 3, 4],
                        [2, 4, 6]])  # n=2, l=3

        b = np.random.randn(2, 1)

        mult = np.matmul(W_0, X)

        check = mult + b

        u = np.random.randn(2, 4)

        d = np.random.rand(len(X), len(X[0]))
        d = d / linalg.norm(d)
        d_f = d.flatten()

        Fw = lambda x: np.inner(np.tanh(x).flatten(), u.flatten())
        Fw_delta = lambda x, epsilon: Fw(x) + np.matmul(epsilon * d_f,
                                                                  backward.jacobianTMV_grad_x(x, W_0, u).flatten()) + epsilon ** 2

        res_sum1 = []
        res_sum2 = []
        eps_0 = 0.001
        input_x = np.matmul(W_0, X) + b
        for i in range(0, 10):
            epsi = (0.5 ** i) * eps_0
            sum1 = abs(Fw_delta(input_x, epsi) - Fw(input_x))
            sum2 = abs(Fw_delta(input_x, epsi) - Fw(input_x) - np.matmul(epsi * d_f,
                                                                  backward.jacobianTMV_grad_x(input_x, W_0, u).flatten()))
            res_sum1.append(sum1)
            res_sum2.append(sum2)

        factors_2 = [res_sum1[i] / res_sum1[i + 1] for i in range(0, len(res_sum1) - 1)]
        factors_4 = [res_sum2[i] / res_sum2[i + 1] for i in range(0, len(res_sum2) - 1)]
        print(f'factors 2: {factors_2}')
        print(f'factors 4: {factors_4}')



    def test_grad_for_JTMV_b(self):
        """
        :param X: data matrix - dimension: n x m
        :param W: weights matrix - dimension: current_layer_features x prev_layer_features
        :param C: classes vector matrix - dimension: l x m
        :param b: bias vector - length: l"""

        X = np.random.randn(3, 4)

        W_0 = np.array([[1, 3, 4],
                        [2, 4, 6]])  # n=2, l=3

        b = np.random.randn(2, 1)

        mult = np.matmul(W_0, X)

        check = mult + b

        u = np.random.randn(2, 4)

        d = np.random.rand(len(b), len(b[0]))
        d = d / linalg.norm(d)
        d_f = d.flatten()

        Fw = lambda x: np.inner(np.tanh(x).flatten(), u.flatten())
        Fw_delta = lambda x, epsilon: Fw(x) + np.matmul(epsilon * d_f,
                                                                  backward.jacobianTMV_grad_b(x, u).flatten()) + epsilon ** 2

        res_sum1 = []
        res_sum2 = []
        eps_0 = 0.001
        input_x = np.matmul(W_0, X) + b
        for i in range(0, 10):
            epsi = (0.5 ** i) * eps_0
            sum1 = abs(Fw_delta(input_x, epsi) - Fw(input_x))
            sum2 = abs(Fw_delta(input_x, epsi) - Fw(input_x) - np.matmul(epsi * d_f,
                                                                  backward.jacobianTMV_grad_b(input_x, u).flatten()))
            res_sum1.append(sum1)
            res_sum2.append(sum2)

        factors_2 = [res_sum1[i] / res_sum1[i + 1] for i in range(0, len(res_sum1) - 1)]
        factors_4 = [res_sum2[i] / res_sum2[i + 1] for i in range(0, len(res_sum2) - 1)]
        print(f'factors 2: {factors_2}')
        print(f'factors 4: {factors_4}')


    def test_grad_for_JTMV_w(self):
        """
        :param X: data matrix - dimension: n x m
        :param W: weights matrix - dimension: current_layer_features x prev_layer_features
        :param C: classes vector matrix - dimension: l x m
        :param b: bias vector - length: l"""

        X = np.random.randn(3, 4)

        W_0 = np.array([[1, 3, 4],
                        [2, 4, 6]])  # n=2, l=3

        b = np.random.randn(2, 1)

        mult = np.matmul(W_0, X)

        check = mult + b

        u = np.random.randn(2, 4)

        d = np.random.rand(len(W_0), len(W_0[0]))
        d = d / linalg.norm(d)
        d_f = d.flatten()

        Fw = lambda x: np.inner(np.tanh(x).flatten(), u.flatten())
        Fw_delta = lambda x, epsilon: Fw(x) + np.matmul(epsilon * d_f,
                                                                  backward.jacobianTMV_grad_w(X, x, u).flatten()) + epsilon ** 2

        res_sum1 = []
        res_sum2 = []
        eps_0 = 0.001
        input_x = np.matmul(W_0, X) + b
        for i in range(0, 10):
            epsi = (0.5 ** i) * eps_0
            sum1 = abs(Fw_delta(input_x, epsi) - Fw(input_x))
            sum2 = abs(Fw_delta(input_x, epsi) - Fw(input_x) - np.matmul(epsi * d_f,
                                                                  backward.jacobianTMV_grad_w(X, input_x, u).flatten()))
            res_sum1.append(sum1)
            res_sum2.append(sum2)

        factors_2 = [res_sum1[i] / res_sum1[i + 1] for i in range(0, len(res_sum1) - 1)]
        factors_4 = [res_sum2[i] / res_sum2[i + 1] for i in range(0, len(res_sum2) - 1)]
        print(f'factors 2: {factors_2}')
        print(f'factors 4: {factors_4}')



    def test_a(self):
        """
        computation of lost function
        :param with_eta: use eta for safety
        :param X: data matrix - dimension: n x m
        :param C: classes vector matrix - dimension: l x m
        :param W: weights matrix - dimension: current_layer_features x prev_layer_features
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

        fw_delta = lambda W, eps: Fw(W) + eps * np.matmul(d_t, grad_Fw(W)) + eps ** 2

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

        stop = 1
        # grad_f = forward.compute_softmax_gradient_vector_respect_to_weights(d, w, c, b)
        # a = 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


