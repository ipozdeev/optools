import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.special import erf
import timeit

from optools import optools as op

class TestSimpleFormulas(unittest.TestCase):
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

        # from a website
        self.true_call = np.array([[1.0769, 0.], [2.2353, 0.0014]])

    def test_bsPrice(self):
        """
        """
        res_call = \
            op.bsPrice(self.f, self.K[1,:], self.rf, self.tau,
                self.sigma[1,:])

        assert_array_almost_equal(res_call, self.true_call[1,:], decimal = 4)

    def test_bsPrice_on_nDim(self):
        """
        N-dimensional inputs
        """
        res_call = \
            op.bsPrice(self.f, self.K, self.rf, self.tau, self.sigma)

        assert_array_almost_equal(res_call, self.true_call, decimal = 4)

    def test_lossFun(self):

        res = op.lossFun(np.array([1,2]), np.array([2,4]),
            1, 1)

        self.assertEqual(res, 5e04)

class TestHarderFormulas(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        K = 100    sigma = 0.50    mu = 4.0    X = 100
            110            0.75         5.0
        """
        self.X = 100
        self.K = np.array([100, 110])
        self.rf = 0.02*31/365

        self.mu = np.array([4., 5.])
        self.sigma = np.array([0.5, 0.75])
        self.par = np.vstack((self.mu, self.sigma))

        self.wght = np.array([0.25, 0.75])
        self.tau = 1

        self.f = np.dot(self.wght, np.exp(self.mu + 0.5*self.sigma*self.sigma))
        self.cHat = \
            op.bsPrice(self.f, self.K, self.rf, self.tau, self.sigma)

        # from Paul Soederlind's function
        self.true_call = np.array([80.7581, 75.4438])

    def test_priceUnderMixture(self):
        """
        """
        res_call = \
            op.priceUnderMixture(self.K,self.rf,self.mu,self.sigma,self.wght)

        assert_array_almost_equal(
            res_call, self.true_call, decimal = 4)

    def test_bsIV(self):
        """
        Draw random sigmas to test if everything matches.
        """
        sigma = np.random.random(2)*2
        self.cHat = \
            op.bsPrice(self.f, self.K, self.rf, self.tau, sigma)

        res = op.bsIV(self.cHat, self.f, self.K, self.rf, self.tau)

        assert_array_almost_equal(res, sigma, decimal = 4)

    def test_objFunForRND(self):
        """
        """
        res_call = \
            op.priceUnderMixture(self.K,self.rf,self.mu,self.sigma,self.wght)

        # omit W
        res = op.objFunForRND(self.par, self.wght, self.K, self.rf,
            res_call, self.f, is_iv = False)

        self.assertAlmostEqual(res, 0.0, places = 2)


if __name__ == "__main__":
    unittest.main()
