from unittest import TestCase
import numpy as np
from numpy.testing import assert_almost_equal

from optools.smile import SABR, VolatilitySmile


class TestVolatilitySmile(TestCase):

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

    def test_estimate_risk_neutral_density(self):
        vs = VolatilitySmile(lambda _x: self.v_atm, tau=self.tau)
        rnd = vs.estimate_risk_neutral_density(
            rf=self.data_rest["r_counter"],
            forward=self.data_rest["forward"],
            normalize=True
        )

    def test_estimate_risk_neutral_density_vectorized(self):
        vs = VolatilitySmile(lambda _x: self.v_atm, tau=self.tau)
        domain = np.arange(
            self.data_rest["forward"] / 2, self.data_rest["forward"] * 2, 0.01
        )[:, np.newaxis] * np.ones((1, 3))
        rnd = vs.estimate_risk_neutral_density(
            rf=self.data_rest["r_counter"],
            forward=self.data_rest["forward"] * np.array([[1, 1.05, 1.1]]),
            domain=domain[:, [0]],
            normalize=True
        )
