import unittest
import numpy as np
from numpy.testing import assert_almost_equal

from smile import SABR
from strike import strike_from_delta, strike_from_atm


class TestClark(unittest.TestCase):
    def setUp(self) -> None:
        """Table 3.3."""
        self.tau = 1.0
        self.v_atm = 0.1825
        self.data_rest = {
            "spot": 1.3465,
            "forward": 1.3395,
            "r_counter": 2.94 / 100,
            "r_base": 3.46 / 100
        }
        self.delta_conventions = {
            "is_forward": False,
            "is_premiumadj": False,
            "atm_def": "dns"
        }
        self.contracts = {
            0.25: {"ms": 0.950 / 100,
                   "rr": -0.600 / 100},
            0.1: {"ms": 3.806 / 100,
                  "rr": -1.359 / 100}
        }

    def test_strike_from_atm(self):
        """Strike from ATM (DNS)."""
        k = strike_from_atm(
            atm_def=self.delta_conventions["atm_def"],
            is_premiumadj=self.delta_conventions["is_premiumadj"],
            forward=self.data_rest["forward"],
            vola=self.v_atm,
            tau=self.tau
        )
        k_true = 1.3620  # ch. 3.5.4
        self.assertAlmostEqual(k, k_true, places=4)

    def test_strike_from_delta(self):
        """Strike from delta."""
        sigma_ms = self.v_atm + self.contracts[0.25]["ms"]

        k = strike_from_delta(
            delta=np.array([0.25, -0.25]), tau=self.tau, vola=sigma_ms,
            is_call=np.array([True, False]),
            is_forward=self.delta_conventions["is_forward"],
            is_premiumadj=self.delta_conventions["is_premiumadj"],
            **self.data_rest,
        )

        k_true = np.array([1.5449, 1.2050])  # eq. 3.11

        assert_almost_equal(k, k_true, decimal=4)

    def test_sabr_fit_to_fx_25d(self):
        """Calibration of SABR with 25-delta contracts."""
        ctr = self.contracts.copy()
        ctr.pop(0.1)
        sabr = SABR.fit_to_fx(
            tau=self.tau, v_atm=self.v_atm,
            contracts=ctr,
            delta_conventions=self.delta_conventions,
            **self.data_rest
        )
        sabr_par = np.array([sabr.init_vola, sabr.volvol, sabr.rho])
        clark_par = np.array([0.1743106, 0.81694072, -0.11268306])
        assert_almost_equal(sabr_par, clark_par, decimal=2)

    def test_sabr_fit_to_fx(self):
        """Calibration of SABR with 25-delta contracts."""
        sabr = SABR.fit_to_fx(
            tau=self.tau, v_atm=self.v_atm,
            contracts=self.contracts,
            delta_conventions=self.delta_conventions,
            **self.data_rest
        )
        print(sabr)