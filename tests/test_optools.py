import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.special import erf
import timeit

from optools import optools as op

class TestOptools(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        K = 100  120    sigma = 0.1  0.1    mu = 0.0  0.0    X = 100
            100  120            0.2  0.2         0.1  0.1
        """
        self.X = 100
        self.K = np.array([[100, 120], [100, 120]])
        self.rf = 0.02
        self.y = 0.04
        self.tau = 31/365
        self.mu = np.array([[0., 0.], [0.1, 0.1]])
        self.sigma = np.array([[0.1, 0.1], [0.2, 0.2]])
        self.w = np.array([0.25, 0.75])

        self.f = self.X*np.exp((self.rf-self.y)*self.tau)
        self.f_rn = np.exp(self.mu + 0.5*self.sigma*self.sigma)

        self.true_call = np.array([[1.0769, 0.], [2.2353, 0.0014]])
        self.true_put = np.array([[1.2463, 20.1355], [2.4047, 20.1369]])

    def test_blPrice(self):
        """
        """
        res_call, res_put = \
            op.blPrice(self.f, self.K[1,:], self.rf, self.tau,
                self.sigma[1,:])

        assert_array_almost_equal(res_call, self.true_call[1,:], decimal = 4)
        assert_array_almost_equal(res_put, self.true_put[1,:], decimal = 4)

    def test_blPrice_on_nDim(self):
        """
        """
        res_call, res_put = \
            op.blPrice(self.f, self.K, self.rf, self.tau, self.sigma)

        assert_array_almost_equal(res_call, self.true_call, decimal = 4)
        assert_array_almost_equal(res_put, self.true_put, decimal = 4)

    def test_priceUnderMixture(self):
        """
        """
        res_call, res_put = \
            op.priceUnderMixture(self.X, self.K, self.rf,
                self.rf, self.sigma)

        assert_array_almost_equal(res_call, self.true_call, decimal = 4)
        assert_array_almost_equal(res_put, self.true_put, decimal = 4)

if __name__ == "__main__":
    unittest.main()
