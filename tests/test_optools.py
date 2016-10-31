import unittest
from numpy.testing import assert_array_almost_equal
import pandas as pd
import numpy as np

usr = "hsg-spezial"
path = "c:/users/" + usr + "/google drive/" + \
"personal/option_implied_betas_project/"

import logging
# logger settings
logging.basicConfig(filename=path+"/log/optools_test_logger.txt",
    filemode='w',
    format="%(asctime)s || %(levelname)s:%(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from optools import optools as op

class TestSimpleFormulas(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        K = [100  120]; sigma = [0.1  0.2]; X = 100
        """
        self.X = 100
        self.K = np.array([100, 120])
        self.rf = 0.02
        self.y = 0.04
        self.tau = 31/365
        self.sigma = np.array([0.1, 0.2])

        self.f = self.X*np.exp((self.rf-self.y)*self.tau)

        # from a website
        self.c_true = np.array([1.0769, 0.0014])

    def test_bs_price(self):
        """
        bs_price returns correct values for a set of strikes and an equally
        long set of sigmas
        """
        res = \
            op.bs_price(self.f, self.K, self.rf, self.tau, self.sigma)

        assert_array_almost_equal(res, self.c_true, decimal = 4)

    def test_loss_fun(self):
        """
        """
        res = op.loss_fun(np.array([1,2]), np.array([2,4]), 1, 1)

        self.assertEqual(res, 5e04)

    def test_bs_iv(self):
        """
        Draw random sigmas to test if they can be recovered.
        """
        self.sigma = np.random.random(2)*0.2+0.1
        self.c_hat = \
            op.bs_price(self.f, self.K, self.rf, self.tau, self.sigma)

        res = op.bs_iv(self.c_hat, self.f, self.K, self.rf, self.tau)

        assert_array_almost_equal(res, self.sigma, decimal = 4)

class TestHarderFormulas(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        K = [100, 110, 120]; sigma = [0.50, 0.75]; mu = [4.0, 5.0]; X = 100
        """
        self.X = 100
        self.K = np.array([100, 110, 120])
        self.rf = 0.01
        self.tau = 1

        self.mu = np.array([4., 5.])
        self.sigma = np.array([0.5, 0.75])
        self.par = np.concatenate([self.mu, self.sigma])
        self.wght = np.array([0.3, 0.7])

        self.f = np.dot(self.wght, np.exp(self.mu + 0.5*self.sigma*self.sigma))

        # from Paul Soederlind's function
        self.c_true = np.array([74.98, 70.00])

    def test_price_under_mixture(self):
        """
        """
        res = \
            op.price_under_mixture(self.K,self.rf,self.mu,self.sigma,self.wght)

        assert_array_almost_equal(res[:2], self.c_true, decimal = 2)

    def test_price_under_mixture_one_component(self):
        """
        """
        f = np.exp(self.mu[0] + 0.5*self.sigma[0]*self.sigma[0])
        res = \
            op.price_under_mixture(
                self.K,
                self.rf,
                np.array([self.mu[0],]),
                np.array([self.sigma[0],]),
                np.array([1,]))

        res_true = op.bs_price(f, self.K, self.rf, self.tau, self.sigma[0])

        assert_array_almost_equal(res, res_true, decimal=4)

    def test_objective_for_rnd(self):
        """
        """
        res_call = \
            op.price_under_mixture(self.K,self.rf,self.mu,self.sigma,self.wght)

        # omit W
        res = op.objective_for_rnd(self.par, self.wght, self.K, self.rf,
            res_call, self.f, is_iv = False)

        self.assertAlmostEqual(res, 0.0, places = 2)

class TestOptimizationProblem(unittest.TestCase):
    """
    """
    def setUp(self):
        self.K = np.arange(85,95,2)
        self.mu = np.random.random(2)+4
        self.sigma = np.random.random(2)
        self.wght = np.array([0.3, 0.7])
        self.rf = 0.01

        self.f_true = self.wght.dot(np.exp(self.mu +
            0.5*self.sigma*self.sigma))

        self.c = op.price_under_mixture(
            self.K,
            self.rf,
            self.mu,
            self.sigma,
            self.wght)

    def test_estimate_rnd_slsqp(self):
        # call prices from given values of mu, sigma and weights

        res = op.estimate_rnd(self.c, self.f_true, self.K, self.rf,
            is_iv = False, W = None, opt_meth="SLSQP")

        # assert_array_almost_equal(res[1], self.wght, decimal = 1)
        assert_array_almost_equal(
            res[:4],
            np.concatenate((self.mu, self.sigma)), decimal = 1)

    def test_estimate_rnd_diff_evol(self):
        # call prices from given values of mu, sigma and weights

        res = op.estimate_rnd(self.c, self.f_true, self.K, self.rf,
            is_iv = False, W = None, opt_meth="differential_evolution")

        # assert_array_almost_equal(res[1], self.wght, decimal = 1)
        assert_array_almost_equal(
            res[:4],
            np.concatenate((self.mu, self.sigma)), decimal = 1)

    # TODO: write a comparison test

if __name__ == "__main__":
    unittest.main()
