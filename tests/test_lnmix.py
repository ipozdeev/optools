import unittest
import numpy as np
from scipy.integrate import quad
from optools import lnmix
from numpy.testing import assert_array_equal, assert_array_almost_equal

class TestLognormalMixtureWithR(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        """
        self.wght = np.array([0.5, 0.5])
        self.mu = np.array([0, 2])
        self.sigma = np.array([1, 3])

        self.ln_mix = lnmix.lognormal_mixture(
            self.mu,
            self.sigma,
            self.wght)

    def test_pdf(self):
        """
        """
        d = self.ln_mix.pdf(1.5)

        self.assertAlmostEqual(d, 0.1609746, places = 7)

    def test_cdf_1_ln(self):
        """
        """
        ln_mix_1d = lnmix.lognormal_mixture(
            self.mu,
            self.sigma,
            np.array([1, 0]))
        p = ln_mix_1d.cdf(2)

        self.assertAlmostEqual(p, 0.7558914, places=7)

    def test_cdf_2ln(self):
        """
        """
        p = self.ln_mix.cdf(np.array([0.5, 1.5]))

        assert_array_almost_equal(p, np.array([0.214389, 0.477482]), decimal=6)

    def test_quantile(self):
        """
        """
        q = self.ln_mix.quantile(0.5)

        self.assertAlmostEqual(q, 1.648721, places = 6)

    def test_quantile_multi(self):
        """
        """
        q = self.ln_mix.quantile(np.array([0.1, 0.9]))

        assert_array_almost_equal(q, np.array([0.236147, 92.28633]), decimal=5)

    def test_moments(self):
        """
        """
        # limit parameters: those initially provided are too large for quad
        self.mu[1] = 1
        self.sigma[1] = 1.5
        E_x, Var_x = self.ln_mix.moments()
        E_x_quad, _, = quad(lambda x: x*self.ln_mix.pdf(x), 0, np.inf)
        Var_x_quad, _, = quad(lambda x: (x-E_x)**2*self.ln_mix.pdf(x),
            0, np.inf)
        self.assertAlmostEqual(E_x, E_x_quad, places=4)
        self.assertAlmostEqual(Var_x, Var_x_quad, places=4)


if __name__ == "__main__":
    unittest.main()
