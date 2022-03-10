from unittest import TestCase
import numpy as np

from optools import blackscholes as bs, blackscholesmix as bsm
from optools.rnd import fit_lognormal_mix


class TestRnd(TestCase):

    def setUp(self) -> None:
        self.tau = 1 / 12
        self.strike = np.random.random(size=(5,))
        self.strike.sort()
        self.vol = (self.strike - self.strike.mean()) ** 2
        self.rf = 0.01
        self.div_yield = 0.0
        self.forward = np.median(self.strike)

        self.c = bs.option_price(self.strike, self.rf, self.tau, self.vol,
                                 self.div_yield, forward=self.forward)

    def test_fit_lognormal_mix(self):
        rnd = fit_lognormal_mix(option_price=self.c,
                                strike=self.strike,
                                forward=self.forward,
                                rf=self.rf)

