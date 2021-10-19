import unittest
from numpy.testing import assert_array_almost_equal
import pandas as pd
import numpy as np
from optools import pricing as op, wrappers as opwraps


class TestFromWystup(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        """
        delta = 0.25
        rr = 0.18
        bf = 0.15
        atm = 4.83

        self.delta = delta
        self.rr = rr
        self.bf = bf
        self.atm = atm

    def test_wings_from_combies_iv(self):
        """
        """
        res = op.vanillas_from_combinations(self.rr, self.bf, self.atm,
                                            self.delta)

        self.assertAlmostEquals(res.loc[self.delta], 5.07, 2)
        self.assertAlmostEquals(res.loc[1-self.delta], 4.89, 2)


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

    @unittest.skip("")
    def test_mfiv_wrapper(self):
        """
        """
        iv_surf = pd.DataFrame(data=np.vstack((self.K,self.iv)).T,
            columns=["K","iv"])
        res = op.mfiv_wrapper(iv_surf, self.f, self.rf, self.tau,
            version="sarno")

        self.assertAlmostEqual(np.sqrt(res/self.tau), self.iv.mean(), places=3)

    # def test_mfskew_wrapper(self):
    #     """
    #     """
    #     iv_surf = pd.DataFrame(data=np.vstack((self.K,self.iv)).T,
    #         columns=["K","iv"])
    #     res = op.mfiskew_wrapper(iv_surf, self.f, self.rf, self.tau, self.S0)
    #     self.assertAlmostEqual(res, 0, places=1)

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

    @unittest.skip("")
    def test_wrapper_implied_co_mat(self):
        """
        """
        vcv, crm = opwraps.wrapper_implied_co_mat(self.variances)

        self.assertAlmostEqual(crm.iloc[1,0], -1*self.rho_pq, 1)
        self.assertAlmostEqual(vcv.iloc[0,0], self.sigma_p**2, 1)
        self.assertAlmostEqual(vcv.iloc[1,0],
            -1*self.sigma_p*self.sigma_q*self.rho_pq, 1)

    @unittest.skip("")
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


