import unittest
import numpy as np
from numpy import linalg
import Components.forward as forward
from Components import backward
from Utils import auxiliary


class GradTests(unittest.TestCase):

    def test_whole_network(self, layers_dim, X, C):
        """
        testing each layer of the network, with variety of length of layers. last layer with grad_test, others with jacb_test
        :param layers_dim: number of neurons in each layer
        :param X: input
        :param C: ground truth labels
        :return:
        """
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

            grad_w, grad_x, grad_b = backward.backward_linear(wb_dict[f'W{i}'], AZ_dict[f'A{i - 1}'], AZ_dict[f'Z{i}'],
                                                              grad_x)
            grads_dict.update({f'grads{i}': {"grad_w": grad_w, "grad_x": grad_x, "grad_b": grad_b}})


    def test_grad_cross_entropy_wrt_w(self, X=None, C=None, W_0=None):
        """
        testing the last layer of the network with cross_entroy w.r.t weights
        :param X: input data
        :param C: ground truth labels
        :param W_0: weights
        :return:
        """

        d = np.random.rand(len(W_0), len(W_0[0]))
        d = d / linalg.norm(d)
        d_f = d.flatten()

        Fw = lambda W: forward.cross_entropy_softmax_lost(X, W, C)
        Fw_delta = lambda W, epsilon: Fw(W) + np.matmul(epsilon * d_f, backward.softmax_grad_wrt_weights(X, W,
                                                                                                         C).flatten()) + epsilon ** 2

        res_sum1 = []
        res_sum2 = []

        eps_0 = 0.001
        for i in range(0, 10):
            epsi = (0.5 ** i) * eps_0
            sum1 = abs(Fw_delta(W_0, epsi) - Fw(W_0))
            sum2 = abs(Fw_delta(W_0, epsi) - Fw(W_0) - np.matmul(epsi * d_f, backward.softmax_grad_wrt_weights(X, W_0,
                                                                                                               C).flatten()))
            res_sum1.append(sum1)
            res_sum2.append(sum2)

        factors_2 = [res_sum1[i] / res_sum1[i + 1] for i in range(0, len(res_sum1) - 1)]
        factors_4 = [res_sum2[i] / res_sum2[i + 1] for i in range(0, len(res_sum2) - 1)]
        np.testing.assert_allclose(factors_2, np.array([2] * len(factors_2)), rtol=0.2)
        np.testing.assert_allclose(factors_4, np.array([4] * len(factors_4)), rtol=0.2)


    def test_jac_wrt_x(self, X=None, W_0=None, b=None):
        """
        the jacobian test w.r.t data
        :param X: input data
        :param W_0: weights
        :param b: bias
        :return:
        """
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


    def test_jac_wrt_b(self, X, W_0, b):
        """
        the jacobian test w.r.t bias
        :param X: data matrix - dimension: n x m
        :param W: weights matrix - dimension: current_layer_features x prev_layer_features
        :param C: classes vector matrix - dimension: l x m
        :param b: bias vector - length: l"""

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


    def test_jac_wrt_w(self, X=None, W_0=None, b=None):
        """
        the jacobian test w.r.t w
        :param X: input data
        :param W_0: weights
        :param b: bias
        :return:
        """
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
                                                                         backward.jacT_wrt_w(X, input_x, u).flatten()))
            res_sum1.append(sum1)
            res_sum2.append(sum2)

        factors_2 = [res_sum1[i] / res_sum1[i + 1] for i in range(0, len(res_sum1) - 1)]
        factors_4 = [res_sum2[i] / res_sum2[i + 1] for i in range(0, len(res_sum2) - 1)]
        np.testing.assert_allclose(factors_2, np.array([2] * len(factors_2)), rtol=0.2)
        np.testing.assert_allclose(factors_4, np.array([4] * len(factors_4)), rtol=0.2)



