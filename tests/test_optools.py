import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.special import erf
import scipy.integrate as integrate
import timeit
import matplotlib.pyplot as plt
import logging

from optools import optools as op

usr = "hsg-spezial"
path = "c:/users/" + usr + "/google drive/" + \
    "personal/research_proposal/option_implied_betas/est_res/"

# class TestLognormalMixture(unittest.TestCase):
#     """
#     """
#     def setUp(self):
#         """
#         """
#         self.wght = np.array([0.3, 0.7])
#         self.mu = np.array([1, 1.2])
#         self.sigma = np.array([0.25, 0.1])
#         self.x = np.arange(0.01, 5, 0.01)
#
#     def test_pdf(self):
#         """
#         """
#
#         ln_mix = op.lognormal_mixture(
#             self.mu,
#             self.sigma,
#             self.wght)
#
#         p = ln_mix.pdf(self.x)
#
#         # plt.plot(self.x, p)
#         # plt.show()
#
#         self.assertAlmostEqual(np.trapz(p, self.x), 1, places = 2)
#
#     def test_cdf(self):
#         """
#         """
#
#         ln_mix = op.lognormal_mixture(
#             self.mu,
#             self.sigma,
#             self.wght)
#
#         p = ln_mix.pdf(self.x)
#         q = ln_mix.cdf(np.array([3., 10.]))
#         q_star = integrate.quad(lambda xx: ln_mix.pdf(xx), 0., 3.)
#         self.assertAlmostEqual(
#             q[0],
#             q_star[0],
#             places=4)
#
# class TestLognormalMixtureWithR(unittest.TestCase):
#     """
#     """
#     def setUp(self):
#         """
#         """
#         self.wght = np.array([0.5, 0.5])
#         self.mu = np.array([0, 2])
#         self.sigma = np.array([1, 3])
#
#         self.ln_mix = op.lognormal_mixture(
#             self.mu,
#             self.sigma,
#             self.wght)
#
#     def test_pdf(self):
#         """
#         """
#         d = self.ln_mix.pdf(1.5)
#
#         self.assertAlmostEqual(d, 0.1609746, places = 7)
#
#     def test_cdf_1_ln(self):
#         """
#         """
#         ln_mix_1d = op.lognormal_mixture(
#             self.mu,
#             self.sigma,
#             np.array([1, 0]))
#         p = ln_mix_1d.cdf(2)
#
#         self.assertAlmostEqual(p, 0.7558914, places=7)
#
#     def test_cdf_2ln(self):
#         """
#         """
#         p = self.ln_mix.cdf(np.array([0.5, 1.5]))
#
#         assert_array_almost_equal(p, np.array([0.214389, 0.477482]), decimal=6)
#
#     def test_quantile(self):
#         """
#         """
#         q = self.ln_mix.quantile(0.5)
#
#         self.assertAlmostEqual(q, 1.648721, places = 6)
#
#     def test_quantile_multi(self):
#         """
#         """
#         q = self.ln_mix.quantile(np.array([0.1, 0.9]))
#
#         assert_array_almost_equal(q, np.array([0.236147, 92.28633]), decimal=5)
#
# class TestSimpleFormulas(unittest.TestCase):
#     """
#     """
#     def setUp(self):
#         """
#         K = [100  120]; sigma = [0.1  0.2]; X = 100
#         """
#         self.X = 100
#         self.K = np.array([100, 120])
#         self.rf = 0.02
#         self.y = 0.04
#         self.tau = 31/365
#         self.sigma = np.array([0.1, 0.2])
#
#         self.f = self.X*np.exp((self.rf-self.y)*self.tau)
#
#         # from a website
#         self.c_true = np.array([1.0769, 0.0014])
#
#     def test_bs_price(self):
#         """
#         bs_price returns correct values for a set of strikes and an equally long set of sigmas
#         """
#         res = \
#             op.bs_price(self.f, self.K, self.rf, self.tau, self.sigma)
#
#         assert_array_almost_equal(res, self.c_true, decimal = 4)
#
#     def test_loss_fun(self):
#         """
#         """
#         res = op.loss_fun(np.array([1,2]), np.array([2,4]), 1, 1)
#
#         self.assertEqual(res, 5e04)
#
#     def test_bs_iv(self):
#         """
#         Draw random sigmas to test if they can be recovered.
#         """
#         self.sigma = np.random.random(2)*0.2+0.1
#         self.c_hat = \
#             op.bs_price(self.f, self.K, self.rf, self.tau, self.sigma)
#
#         res = op.bs_iv(self.c_hat, self.f, self.K, self.rf, self.tau)
#
#         assert_array_almost_equal(res, self.sigma, decimal = 4)
#
#
#
# class TestHarderFormulas(unittest.TestCase):
#     """
#     """
#     def setUp(self):
#         """
#         K = [100, 110, 120]; sigma = [0.50, 0.75]; mu = [4.0, 5.0]; X = 100
#         """
#         self.X = 100
#         self.K = np.array([100, 110, 120])
#         self.rf = 0.01
#         self.tau = 1
#
#         self.mu = np.array([4., 5.])
#         self.sigma = np.array([0.5, 0.75])
#         self.par = np.concatenate([self.mu, self.sigma])
#         self.wght = np.array([0.3, 0.7])
#
#         self.f = np.dot(self.wght, np.exp(self.mu + 0.5*self.sigma*self.sigma))
#
#         # from Paul Soederlind's function
#         self.c_true = np.array([74.98, 70.00])
#
#     def test_price_under_mixture(self):
#         """
#         """
#         res = \
#             op.price_under_mixture(self.K,self.rf,self.mu,self.sigma,self.wght)
#
#         assert_array_almost_equal(res[:2], self.c_true, decimal = 2)
#
#     def test_price_under_mixture_one_component(self):
#         """
#         """
#         f = np.exp(self.mu[0] + 0.5*self.sigma[0]*self.sigma[0])
#         res = \
#             op.price_under_mixture(
#                 self.K,
#                 self.rf,
#                 np.array([self.mu[0],]),
#                 np.array([self.sigma[0],]),
#                 np.array([1,]))
#
#         res_true = op.bs_price(f, self.K, self.rf, self.tau, self.sigma[0])
#
#         assert_array_almost_equal(res, res_true, decimal=4)
#
#     def test_objective_for_rnd(self):
#         """
#         """
#         res_call = \
#             op.price_under_mixture(self.K,self.rf,self.mu,self.sigma,self.wght)
#
#         # omit W
#         res = op.objective_for_rnd(self.par, self.wght, self.K, self.rf,
#             res_call, self.f, is_iv = False)
#
#         self.assertAlmostEqual(res, 0.0, places = 2)
#
# class TestOptimizationProblem(unittest.TestCase):
#     """
#     """
#     def setUp(self):
#         self.K = np.arange(85,95,2)
#         self.mu = np.random.random(2)+4
#         self.sigma = np.random.random(2)
#         self.wght = np.array([0.3, 0.7])
#         self.rf = 0.01
#
#         self.f_true = self.wght.dot(np.exp(self.mu + 0.5*self.sigma*self.sigma))
#
#         self.c = op.price_under_mixture(
#             self.K,
#             self.rf,
#             self.mu,
#             self.sigma,
#             self.wght)
#
#     def test_estimate_rnd(self):
#         # call prices from given values of mu, sigma and weights
#
#         res = op.estimate_rnd(self.c, self.f_true, self.K, self.rf,
#             is_iv = False, W = None)
#
#         # assert_array_almost_equal(res[0], self.wght, decimal = 1)
#         assert_array_almost_equal(
#             res[1],
#             np.concatenate((self.mu, self.sigma)), decimal = 1)

class TestRealStuff(unittest.TestCase):
    """
    """
    # def setUp(self):
    #     """
    #     """
    #     data = pd.read_csv(
    #         "c:/Users/pozdeev/Desktop/piece_opt_data.txt",
    #         header=0, index_col=0)
    #     iv_quote = data.ix[1,:5].values/100/4  # quarterly?
    #     self.r10 = iv_quote[0]
    #     self.r25 = iv_quote[1]
    #     self.b10 = iv_quote[2]
    #     self.b25 = iv_quote[3]
    #     self.atm = iv_quote[4]
    #
    #     self.S = data["S"].values[1]
    #     self.f = self.S + data["F"].values[1]/10000
    #     self.rf = data["rf"].values[1]/100/4  # quarterly
    #     self.y = data["y"].values[1]/100/4  # quarterly
    #
    #     # self.r25 = 0.18
    #     # self.b25 = 0.15
    #     # self.atm = 4.83
    #     # self.r10 = self.r25
    #     # self.b10 = self.b25
    #     # self.y = 0
    #
    # def test_get_wings(self):
    #     """
    #     """
    #     deltas, ivs = op.get_wings(
    #         self.r25, self.r10, self.b25, self.b10, self.atm, self.y, 1)
    #
    #     self.assertAlmostEqual(deltas[2], 0.5, places=2)
    #     # self.assertEqual(ivs[1], 5.07)
    #
    # def test_strike_from_delta(self):
    #     """
    #     """
    #     deltas, ivs = op.get_wings(
    #         self.r25, self.r10, self.b25, self.b10, self.atm, self.y, 1)
    #     res = op.strike_from_delta(deltas, self.S, self.rf, self.y, 1,
    #         ivs, True)
    #
    #     print(res)

if __name__ == "__main__":
    unittest.main()
