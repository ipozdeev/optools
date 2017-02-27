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

from optools import optools as op, optools_wrappers as opwraps

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
#         bs_price returns correct values for a set of strikes and an equally
#         long set of sigmas
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
#         self.f_true = self.wght.dot(np.exp(self.mu +
#             0.5*self.sigma*self.sigma))
#
#         self.c = op.price_under_mixture(
#             self.K,
#             self.rf,
#             self.mu,
#             self.sigma,
#             self.wght)
#
#     def test_estimate_rnd_slsqp(self):
#         # call prices from given values of mu, sigma and weights
#
#         res = op.estimate_rnd(self.c, self.f_true, self.K, self.rf,
#             is_iv = False, W = None, opt_meth="SLSQP")
#
#         # assert_array_almost_equal(res[1], self.wght, decimal = 1)
#         assert_array_almost_equal(
#             res[:4],
#             np.concatenate((self.mu, self.sigma)), decimal = 1)
#
#     def test_estimate_rnd_diff_evol(self):
#         # call prices from given values of mu, sigma and weights
#
#         res = op.estimate_rnd(self.c, self.f_true, self.K, self.rf,
#             is_iv = False, W = None, opt_meth="differential_evolution")
#
#         # assert_array_almost_equal(res[1], self.wght, decimal = 1)
#         assert_array_almost_equal(
#             res[:4],
#             np.concatenate((self.mu, self.sigma)), decimal = 1)
#
#     # TODO: write a comparison test

class TestMfiv(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        """
        self.K = np.random.random(size=(5,))+1.0
        self.iv = np.array([np.random.random()*0.25+0.1,]*5)
        self.S0 = \
            np.random.random()*(np.max(self.K)-np.min(self.K))+np.min(self.K)
        self.rf = np.random.random()*0.05
        self.tau = 0.25
        self.f = self.S0*np.exp(self.rf*self.tau)
        # option prices
        self.C = op.bs_price(self.f, self.K, self.rf, self.tau, self.iv)
        # ivs

    # def test_interpolate_iv_1d(self):
    #     """ Asserts constant iv surface is interpolated as constant.
    #     """
    #     # ivs_extracted = op.bs_iv(self.C, self.f, self.K,
    #     #     self.rf, self.tau)
    #     # from arrays to DataFrame
    #     iv_surf = pd.DataFrame(data=np.vstack((self.K,self.iv)).T,
    #         columns=["K","iv"])
    #     # interpolate
    #     ivs_interp = op.interpolate_iv(iv_surf)
    #     # assert
    #     self.assertAlmostEqual(ivs_interp.mean(), self.iv.mean(), places=4)

    # def test_interpolate_iv_2d(self):
    #     """ Asserts constant iv surface is interpolated as constant, now with
    #     maturity dimension.
    #     """
    #     # additional strikes
    #     K_add = np.random.random(size=(5,))*2
    #     # from arrays to DataFrame
    #     iv_surf = pd.DataFrame(data=np.vstack((
    #         np.concatenate((self.K,K_add)),
    #         np.tile(self.iv, 2),
    #         np.concatenate(
    #             (np.array([self.tau,]*5),np.array([self.tau*2,]*5))))).T,
    #         columns=["K","iv","tau"])
    #     # interpolate
    #     ivs_interp = op.interpolate_iv(iv_surf, tau=self.tau)
    #     # assert
    #     self.assertAlmostEqual(ivs_interp.mean(), self.iv.mean(), places=4)

    # def test_interpolate_iv_kernel_2d(self):
    #     """ Asserts constant iv surface is interpolated as constant.
    #     """
    #     # from arrays to DataFrame
    #     iv_surf = pd.DataFrame(data=np.vstack((self.K,self.iv)).T,
    #         columns=["K","iv"])
    #     # interpolate
    #     ivs_interp = op.interpolate_iv(iv_surf, method="kernel")
    #     # assert
    #     self.assertAlmostEqual(ivs_interp.mean(), self.iv.mean(), places=4)
    #
    # def test_interpolate_iv_kernel_3d(self):
    #     """ Asserts constant iv surface is interpolated as constant, now with
    #     maturity dimension.
    #     """
    #     X_pred = pd.DataFrame(
    #         data=np.vstack(
    #             (np.linspace(0.5,2,100), np.ones((100,))*self.tau)).T,
    #         columns=["K", "tau"])
    #     # additional strikes
    #     K_add = np.random.random(size=(5,))*2
    #     # from arrays to DataFrame
    #     iv_surf = pd.DataFrame(data=np.vstack((
    #         np.concatenate((self.K,K_add)),
    #         np.tile(self.iv, 2),
    #         np.concatenate(
    #             (np.array([self.tau,]*5),np.array([self.tau*2,]*5))))).T,
    #         columns=["K","iv","tau"])
    #     # interpolate
    #     ivs_interp = op.interpolate_iv(iv_surf, method="kernel", X_pred=X_pred)
    #     # assert
    #     self.assertAlmostEqual(ivs_interp.mean(), self.iv.mean(), places=4)

    def test_mfiv_wrapper(self):
        """
        """
        iv_surf = pd.DataFrame(data=np.vstack((self.K,self.iv)).T,
            columns=["K","iv"])
        res = op.mfiv_wrapper(iv_surf, self.f, self.rf, self.tau)

        self.assertAlmostEqual(np.sqrt(res/self.tau), self.iv.mean(), places=1)

    def test_mfskew_wrapper(self):
        """
        """
        iv_surf = pd.DataFrame(data=np.vstack((self.K,self.iv)).T,
            columns=["K","iv"])
        res = op.mfiskew_wrapper(iv_surf, self.f, self.rf, self.tau, self.S0)
        self.assertAlmostEqual(res, 0, places=1)

class TestOptoolsWrappers(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        """
        sigma_p, sigma_q = 0.1, 0.3
        rho_pq = -0.5
        w_p = 0.2
        w_q = 1-w_p
        s2_m = w_p**2*sigma_p**2 + w_q**2*sigma_q**2 + \
            2*w_p*w_q*sigma_p*sigma_q*rho_pq
        cov_pm = w_p*sigma_p**2 + w_q*sigma_p*sigma_q*rho_pq
        beta_pm = cov_pm/s2_m

        covmat = np.array([
            [sigma_p**2, sigma_p*sigma_q*rho_pq],
            [sigma_p*sigma_q*rho_pq, sigma_q**2]])

        # for implied_co_mat
        X1 = pd.Series(data=np.random.normal(size=(10000,)))
        X2 = pd.Series(data=np.random.normal(size=(10000,)))\
            *np.sqrt(1-rho_pq**2) + X1*rho_pq
        cadusd = X1*sigma_p
        usdchf = X2*sigma_q
        cadchf = cadusd + usdchf
        variances = pd.concat((cadusd,cadchf,usdchf), axis=1).var()
        variances.index = ["cadusd", "cadchf", "usdchf"]

        self.variances = variances

        self.beta_pm = beta_pm
        self.covmat = pd.DataFrame(data=covmat,
            columns=["p","q"],
            index=["p","q"])
        self.wght = pd.Series(data=np.array([w_p, w_q]), index=["p","q"])
        self.s2_m = s2_m
        self.rho_pq = rho_pq
        self.sigma_p = sigma_p
        self.sigma_q = sigma_q

    def test_wrapper_implied_co_mat(self):
        """
        """
        vcv, crm = opwraps.wrapper_implied_co_mat(self.variances)

        self.assertAlmostEqual(crm.iloc[1,0], -1*self.rho_pq, 1)
        self.assertAlmostEqual(vcv.iloc[0,0], self.sigma_p**2, 1)
        self.assertAlmostEqual(vcv.iloc[1,0],
            -1*self.sigma_p*self.sigma_q*self.rho_pq, 1)

    def test_wrapper_beta_from_covmat(self):
        """
        """
        beta, s2_m = opwraps.wrapper_beta_from_covmat(self.covmat, self.wght)
        beta = beta["p"]
        s2_m = s2_m

        self.assertAlmostEqual(beta, self.beta_pm)
        self.assertAlmostEqual(s2_m, self.s2_m)

if __name__ == "__main__":
    unittest.main()


# interpolate.splrep(
#     np.array([1.93073507,1.40213606,1.59884974,1.29882618,1.12422187,1.0]),
#     np.array([0.1,0.2,0.3,0.4,0.5,0.6]))
#
# x = sorted(np.random.normal(size=(10,)))
# interpolate.splrep(x, np.exp(np.sin(x)))
# lol = pd.DataFrame(data=np.random.random((100,2)))
# list(lol.values.T)
# lol.index.values.diff()
