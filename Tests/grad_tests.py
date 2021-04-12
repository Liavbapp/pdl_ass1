import unittest
import numpy as np
from numpy import linalg

import Components.forward as forward
from Components import backward
from Utils import auxiliary


class GradTests(unittest.TestCase):

    def test_whole_network(self):
        X = np.random.randn(2, 8)
        C = np.array([[0., 1., 1., 1., 1., 1., 1., 1.],
                      [1., 0., 0., 0., 0., 0., 0., 0.]])
        num_features = X.shape[0]
        num_labels = C.shape[0]
        layers_dim = [num_features, 5, 3, num_labels]
        wb_dict = auxiliary.initiate_wb_dict(layers_dim)
        A_L, AZ_dict = forward.forward_pass(X, wb_dict)

        num_layers = len(wb_dict.keys()) // 2
        grads_dict = {}

        # Test the softmax layer
        self.test_grad_cross_entropy_wrt_w(AZ_dict[f'A{num_layers - 1}'], C, wb_dict[f'W{num_layers}'])
        # after test pass we compute the actual grads and update the dictionaries
        grad_w, grad_x, grad_b = backward.backward_softmax(C, wb_dict[f'W{num_layers}'], AZ_dict[f'A{num_layers - 1}'])
        grads_dict.update({f'grads{num_layers}': {"grad_w": grad_w, "grad_x": grad_x, "grad_b": grad_b}})

        for i in range(num_layers - 1, 0, -1):
            # Test jac_grad_wrt_x
            self.test_jac_wrt_x(X=AZ_dict[f'A{i - 1}'], W_0=wb_dict[f'W{i}'], b=wb_dict[f'b{i}'])
            # Test jac_grad_wrt_w
            self.test_jac_wrt_w(X=AZ_dict[f'A{i - 1}'], W_0=wb_dict[f'W{i}'], b=wb_dict[f'b{i}'])
            # Test jac_grad_wrt_b
            self.test_jac_wrt_b(X=AZ_dict[f'A{i - 1}'], W_0=wb_dict[f'W{i}'], b=wb_dict[f'b{i}'])

            # after test pass we compute the actual grads and update the dictionaries

            grad_w, grad_x, grad_b = backward.backward_linear(wb_dict[f'W{i}'], AZ_dict[f'A{i - 1}'], AZ_dict[f'Z{i}'], grad_x,
                                                     wb_dict[f'b{i}'])
            grads_dict.update({f'grads{i}': {"grad_w": grad_w, "grad_x": grad_x, "grad_b": grad_b}})


    def test_grad_cross_entropy_wrt_w(self, X=None, C=None, W_0=None):

        # X = np.array([[10, 5, 8, 6],
        #               [11, 3, 2, 5],
        #               [6, 1, 7, 4]])  # n=3,  m=4
        #
        # C = np.array([[1, 0, 0, 1],
        #               [0, 1, 1, 0]])  # l=2, m=4
        #
        # W_0 = np.array([[1, 3, 4],
        #                 [2, 4, 6]])  # n=2, l=3
        #
        # b = np.array([6, 5])

        d = np.random.rand(len(W_0), len(W_0[0]))
        d = d / linalg.norm(d)
        d_f = d.flatten()

        Fw = lambda W: forward.cross_entropy_softmax_lost(X, W, C)
        Fw_delta = lambda W, epsilon: Fw(W) + epsilon * np.matmul(d_f,
                                                                  backward.softmax_grad_wrt_weights(X, W,
                                                                                                    C).flatten()) + epsilon ** 2

        res_sum1 = []
        res_sum2 = []
        eps_0 = 0.001
        for i in range(0, 10):
            epsi = (0.5 ** i) * eps_0
            sum1 = abs(Fw_delta(W_0, epsi) - Fw(W_0))
            sum2 = abs(Fw_delta(W_0, epsi) - Fw(W_0) - epsi * np.matmul(d_f,
                                                                        backward.softmax_grad_wrt_weights(X, W_0,
                                                                                                          C).flatten()))
            res_sum1.append(sum1)
            res_sum2.append(sum2)

        factors_2 = [res_sum1[i] / res_sum1[i + 1] for i in range(0, len(res_sum1) - 1)]
        factors_4 = [res_sum2[i] / res_sum2[i + 1] for i in range(0, len(res_sum2) - 1)]
        np.testing.assert_allclose(factors_2, np.array([2] * len(factors_2)), rtol=0.2)
        np.testing.assert_allclose(factors_4, np.array([4] * len(factors_4)), rtol=0.2)

        # print(f'factors 2: {factors_2}')
        # print(f'factors 4: {factors_4}')

    def test_jac_wrt_x(self, X=None, W_0=None, b=None):
        # X = np.random.randn(3, 4)
        #
        # W_0 = np.array([[1, 3, 4],
        #                 [2, 4, 6]])  # n=2, l=3
        #
        # b = np.random.randn(2, 1)

        u = np.random.randn(W_0.shape[0], X.shape[1])
        d = np.random.rand(len(X), len(X[0]))
        d = d / linalg.norm(d)
        d_f = d.flatten()

        Fw = lambda x: np.inner(np.tanh(x).flatten(), u.flatten())
        Fw_delta = lambda x, epsilon: Fw(x) + np.matmul(epsilon * d_f,
                                                        backward.jacT_wrt_x(x, W_0, u).flatten()) + epsilon ** 2

        res_sum1 = []
        res_sum2 = []
        eps_0 = 0.001
        input_x = np.matmul(W_0, X) + b
        for i in range(0, 10):
            epsi = (0.5 ** i) * eps_0
            sum1 = abs(Fw_delta(input_x, epsi) - Fw(input_x))
            sum2 = abs(Fw_delta(input_x, epsi) - Fw(input_x) - np.matmul(epsi * d_f, backward.jacT_wrt_x(input_x, W_0,
                                                                                                         u).flatten()))
            res_sum1.append(sum1)
            res_sum2.append(sum2)

        factors_2 = [res_sum1[i] / res_sum1[i + 1] for i in range(0, len(res_sum1) - 1)]
        factors_4 = [res_sum2[i] / res_sum2[i + 1] for i in range(0, len(res_sum2) - 1)]
        np.testing.assert_allclose(factors_2, np.array([2] * len(factors_2)), rtol=0.2)
        np.testing.assert_allclose(factors_4, np.array([4] * len(factors_4)), rtol=0.2)
        # print(f'factors 2: {factors_2}')
        # print(f'factors 4: {factors_4}')

    def test_jac_wrt_b(self, X, W_0, b):
        """
        :param X: data matrix - dimension: n x m
        :param W: weights matrix - dimension: current_layer_features x prev_layer_features
        :param C: classes vector matrix - dimension: l x m
        :param b: bias vector - length: l"""

        # X = np.random.randn(3, 4)
        #
        # W_0 = np.random.randn(2, 3)  # n=2, l=3
        #
        # b = np.random.randn(2, 1)
        u = np.random.randn(W_0.shape[0], X.shape[1])
        d = np.random.rand(len(b), len(b[0]))
        d = d / linalg.norm(d)
        d_f = d.flatten()

        Fw = lambda x: np.inner(np.tanh(x).flatten(), u.flatten())
        Fw_delta = lambda x, epsilon: Fw(x) + np.matmul(epsilon * d_f,
                                                        backward.jacT_wrt_b(x, u).flatten()) + epsilon ** 2

        res_sum1 = []
        res_sum2 = []
        eps_0 = 0.001
        input_x = np.matmul(W_0, X) + b
        for i in range(0, 10):
            epsi = (0.5 ** i) * eps_0
            sum1 = abs(Fw_delta(input_x, epsi) - Fw(input_x))
            sum2 = abs(Fw_delta(input_x, epsi) - Fw(input_x) - np.matmul(epsi * d_f,
                                                                         backward.jacT_wrt_b(input_x,
                                                                                             u).flatten()))
            res_sum1.append(sum1)
            res_sum2.append(sum2)

        factors_2 = [res_sum1[i] / res_sum1[i + 1] for i in range(0, len(res_sum1) - 1)]
        factors_4 = [res_sum2[i] / res_sum2[i + 1] for i in range(0, len(res_sum2) - 1)]
        np.testing.assert_allclose(factors_2, np.array([2] * len(factors_2)), rtol=0.2)
        np.testing.assert_allclose(factors_4, np.array([4] * len(factors_4)), rtol=0.2)
        # print(f'factors 2: {factors_2}')
        # print(f'factors 4: {factors_4}')

    def test_jac_wrt_w(self, X=None, W_0=None, b=None):
        #
        # X = np.random.randn(3, 4)
        #
        # W_0 = np.array([[1, 3, 4],
        #                 [2, 4, 6]])  # n=2, l=3
        #
        # b = np.random.randn(2, 1)

        u = np.random.randn(W_0.shape[0], X.shape[1])

        d = np.random.rand(len(W_0), len(W_0[0]))
        d = d / linalg.norm(d)
        d_f = d.flatten()

        Fw = lambda x: np.inner(np.tanh(x).flatten(), u.flatten())
        Fw_delta = lambda x, epsilon: Fw(x) + np.matmul(epsilon * d_f,
                                                        backward.jacT_wrt_w(X, x, u).flatten()) + epsilon ** 2

        res_sum1 = []
        res_sum2 = []
        eps_0 = 0.001
        input_x = np.matmul(W_0, X) + b
        for i in range(0, 10):
            epsi = (0.5 ** i) * eps_0
            sum1 = abs(Fw_delta(input_x, epsi) - Fw(input_x))
            sum2 = abs(Fw_delta(input_x, epsi) - Fw(input_x) - np.matmul(epsi * d_f,
                                                                         backward.jacT_wrt_w(X, input_x,
                                                                                             u).flatten()))
            res_sum1.append(sum1)
            res_sum2.append(sum2)

        factors_2 = [res_sum1[i] / res_sum1[i + 1] for i in range(0, len(res_sum1) - 1)]
        factors_4 = [res_sum2[i] / res_sum2[i + 1] for i in range(0, len(res_sum2) - 1)]
        np.testing.assert_allclose(factors_2, np.array([2] * len(factors_2)), rtol=0.2)
        np.testing.assert_allclose(factors_4, np.array([4] * len(factors_4)), rtol=0.2)
        # print(f'factors 2: {factors_2}')
        # print(f'factors 4: {factors_4}')
