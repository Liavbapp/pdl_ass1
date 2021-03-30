import unittest
import numpy as np
from Components import forward


class Tests(unittest.TestCase):

    def test_softmax(self):
        X = np.random.randn(5, 10)  # features=5,  m=10

        W = np.random.randn(2, 5)  # cur_layer=2, prev_layer=5

        softmax_res = forward.softmax(X, W)
        self.assertTrue(softmax_res.shape[0] == 2 and softmax_res.shape[1] == 10)
        np.testing.assert_allclose(softmax_res.sum(axis=0), np.ones(10))

    def test_compute_softmax_gradient_vector_respect_to_weights(self):
        X = np.random.rand(10, 4)  # features=10,  m=4

        W = np.random.rand(3, 10)  # cur=3, prev=10

        C = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 0],
                      [0, 0, 0, 1]])  # l=3, m=4

        actual_grads = forward.compute_softmax_gradient_vector_respect_to_weights(X, W, C)
        self.assertTrue(actual_grads.shape[0] == W.shape[0] and actual_grads.shape[1] == W.shape[1])

    def test_cross_entropy_softmax_lost(self):
        X = np.array([[0.52818428, 0.04495849, 0.55422462, 0.8550578],
                      [0.60724775, 0.3631421, 0.20652824, 0.95099002],
                      [0.2413483, 0.72092741, 0.28588687, 0.76531959],
                      [0.92140116, 0.26976411, 0.58529224, 0.21398077],
                      [0.60856064, 0.01444655, 0.68838315, 0.78568933]])  # n=5, m=4

        W = np.array([[9.42584723e-01, 3.87141862e-01, 5.65437122e-04, 7.10752235e-01, 5.08893974e-01],
                      [6.04889885e-01, 9.65193922e-01, 4.28946985e-01, 6.08191680e-01, 1.35084548e-01],
                      [2.54619842e-01, 1.78962612e-01, 8.01471760e-01, 6.80141444e-01, 2.35958096e-01]])  # n=3 , n-1=5

        C = np.array([[0, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 1, 0, 1]])  # l=3, m=4
        expected_loss = 1.085366618369291
        actual_loss = forward.cross_entropy_softmax_lost(X, W, C, with_eta=False)
        np.testing.assert_allclose(expected_loss, actual_loss)

    def test_forward_pass_t1(self):
        X = np.random.rand(5, 3)  # n=5, m=3
        W = np.random.rand(2, 5)  # w1=2, w0=5
        W_dict = {'W1': W}
        A_L = forward.forward_pass(X, W_dict)
        self.assertTrue(A_L.shape[0] == 2, A_L.shape[1] == 3)
        np.testing.assert_allclose(np.sum(A_L, axis=0), np.ones(3))
