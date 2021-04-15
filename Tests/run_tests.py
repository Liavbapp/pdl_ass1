import unittest
from Tests import grad_tests
import numpy as np

from Tests.grad_tests import GradTests

X = np.random.randn(2, 8)
C = np.array([[0., 1., 1., 1., 1., 1., 1., 1.],
              [1., 0., 0., 0., 0., 0., 0., 0.]])

def prepare_test_whole_network(grad_tester, suite):
    # testing_whole_network
    layers_dims = [[2, 5, 3, 2], [2, 10, 5, 3, 2], [2, 15, 5, 6, 2, 2]]
    for layers_dim in layers_dims:
        suite.addTests(
            loader.loadTestsFromModule(grad_tests.GradTests.test_whole_network(grad_tester, layers_dim, X, C)))


def prepare_jac_tests(grad_tester, suite):
    W = np.random.randn(3, 2)
    b = np.random.rand(3, 1)
    suite.addTests(loader.loadTestsFromModule(grad_tests.GradTests.test_jac_wrt_x(grad_tester, X, W, b)))
    suite.addTests(loader.loadTestsFromModule(grad_tests.GradTests.test_jac_wrt_w(grad_tester, X, W, b)))
    suite.addTests(loader.loadTestsFromModule(grad_tests.GradTests.test_jac_wrt_b(grad_tester, X, W, b)))

def prepare_grad_cross_entropy_test(grad_tester, suite):
    W = np.random.randn(2, 2)
    suite.addTests(loader.loadTestsFromModule(grad_tests.GradTests.test_grad_cross_entropy_wrt_w(grad_tester, X, C, W)))





loader = unittest.TestLoader()
suite = unittest.TestSuite()
grad_tester = GradTests()
# add tests of the complete network
prepare_test_whole_network(grad_tester, suite)
prepare_jac_tests(grad_tester, suite)
prepare_grad_cross_entropy_test(grad_tester, suite)
# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=0)
result = runner.run(suite)
