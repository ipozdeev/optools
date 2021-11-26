from unittest import TestCase
import numpy as np
import pandas as pd

from optools.volsmile import VolatilitySmile
from optools.blackscholes import bs_price


class TestVolatilitySmile(TestCase):

    def setUp(self) -> None:
        self.k = np.random.random(size=(5, ))
        poly = lambda x: (x - np.random.random())**2
        self.vol = poly(self.k)
        self.vol_series = pd.Series(self.vol, pd.Index(self.k, name="strike"))
        self.vol_series_x = pd.Series(self.vol, pd.Index(self.k))

    def test_init(self):
        self.assertRaises(ValueError, VolatilitySmile,
                          strike=self.k, vol=self.vol, delta=self.k)
        self.assertRaises(ValueError, VolatilitySmile,
                          vol_series=self.vol_series_x)
        smile = VolatilitySmile(vol_series=self.vol_series)
        self.assertTrue(smile.smile.equals(smile.smile.sort_index()))

    def test_interpolate(self):
        smile = VolatilitySmile(vol_series=self.vol_series)
        smile_i = smile.interpolate(extrapolate=False)
        self.assertEqual(min(smile.x), min(smile_i.x))
        smile_x = smile.interpolate(extrapolate=True)
        self.assertLess(min(smile_x.x), min(smile.x))
        self.assertEqual(smile_x.smile.loc[min(smile_x.x)],
                         smile.smile.loc[min(smile.x)])

    def test_get_mfivariance(self):
        """Test MFIVol of B-S (const vol) smile equals to vol."""
        # prices
        strike = np.random.random(size=(5, )) / 4 + 0.5
        rf = np.random.random() / 50 + 0.01
        div_yield = np.random.random() / 50 + 0.04
        vol = np.random.random() / 8 + 0.15
        tau = 1/12
        spot = np.median(strike)
        forward = spot * np.exp((rf - div_yield) * tau)
        # smile (will be flat)
        smile = VolatilitySmile(strike=strike, vol=vol, tau=tau)
        # MFIV
        res = smile\
            .interpolate(extrapolate=True)\
            .get_mfivariance(forward=forward, rf=rf)
        self.assertAlmostEqual(res, vol**2, 4)
