import pandas as pd
import numpy as np
from foolbox.data_mgmt.set_credentials import *
# import numpy as np

from foolbox.linear_models import PureOls, DynamicOLS
from optools.optools_wrappers import normalize_weights

class FactorModelEnvironment():
    """
    """
    def __init__(self, assets, factors):
        """
        """
        self.assets = assets.copy()
        self.factors = factors.copy()

    def get_betas(self, method="simple", denom=False, **kwargs):
        """Calculate betas.

        Parameters
        ----------
        method : str
            'simple', 'rolling', 'expanding', 'grouped_by';
            example with grouped_by: [lambda x: x.year, lambda x: x.month]

        Returns
        -------
        res : pandas.core.generic.NDFrame
            with betas: DataFrame or Panel, depending on the input
        """
        if method not in ("simple", "rolling", "expanding", "grouped_by"):
            raise ValueError("Method not implemented.")

        if method == "simple":
            mod = PureOls(y0=self.assets, X0=self.factors, add_constant=True)
            res = mod.fit(**kwargs).T
        else:
            mod = DynamicOLS(y0=self.assets, x0=self.factors)
            res = mod.fit(method=method, denom=denom, **kwargs)

        return res

    @classmethod
    def from_weights(cls, assets, weights=None, exclude_self=False):
        """
        """
        if weights is None:
            # rowsums
            weights = assets.notnull().astype(float)

        # if exclude_self
        if exclude_self:
            factors = [
                construct_factor(
                    assets=assets,
                    weights=weights.drop(c, axis=1)).rename(c)
                for c in assets.columns]
            factors = pd.concat(factors, axis=1)

            # one environment for each asset
            res = {
                k: cls(assets=assets[k], factors=factors[k].rename("factor")) \
                for k in assets.columns}

        else:
            # else as usual
            factors = construct_factor(assets, weights).to_frame("factor")
            res = cls(assets=assets, factors=factors)

        return res

def construct_factor(assets, weights):
    """
    """
    w = norm_weights(weights)
    factor = assets.mul(w, axis=0).sum(axis=1)

    return factor

def norm_weights(weights):
    """
    """
    new_weights = weights.copy()*np.nan

    for t, row in weights.iterrows():
        new_weights.loc[t, :] = normalize_weights(row)

    return new_weights


if __name__ == "__main__":

    pass
