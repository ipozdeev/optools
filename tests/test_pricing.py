from unittest import TestCase
import numpy as np

from optools.pricing import strike_from_delta


class TestEURUSD(TestCase):

    def setUp(self) -> None:
        self.spot = 1.3465
        self.rf = 2.94 / 100  # usd rate
        self.div_yield = 3.46 / 100  # eur rate
        self.tau = 1
        self.forward = 1.3395
        self.atm_vol = 0.1825
        self.ms_25 = 0.0095
        self.is_forward = False
        self.is_premiumadjusted = False

        self.c_25_strike = 1.5449  # eq. 3.11
        self.p_25_strike = 1.2050  # eq. 3.11

    def test_strike_from_delta(self):
        vol = self.atm_vol + self.ms_25
        strike_call = strike_from_delta(
            delta=0.25, spot=self.spot, forward=self.forward, rf=self.rf,
            div_yield=self.div_yield, tau=self.tau, vol=vol,
            is_call=True,
            is_forward=self.is_forward,
            is_premiumadjusted=self.is_premiumadjusted
        )
        strike_put = strike_from_delta(
            delta=-0.25, spot=self.spot, forward=self.forward, rf=self.rf,
            div_yield=self.div_yield, tau=self.tau, vol=vol,
            is_call=False,
            is_forward=self.is_forward,
            is_premiumadjusted=self.is_premiumadjusted
        )
        self.assertEqual(np.round(strike_call, 4), self.c_25_strike)
        self.assertEqual(np.round(strike_put, 4), self.p_25_strike)
