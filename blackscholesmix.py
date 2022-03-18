import numpy as np

from .blackscholes import option_price as option_price_bs

from typing import Union


def forward_price(w, mu, vol) -> Union[float, np.ndarray]:
    """

    Parameters
    ----------
    w
    mu
    vol

    Returns
    -------

    """
    res = np.exp(mu + vol ** 2 / 2) @ w

    return res


def option_price(w, mu, vol, strike, rf, is_call) \
        -> Union[float, np.ndarray]:
    """

    Parameters
    ----------
    w : np.ndarray
    mu : np.ndarray
    vol : np.ndarray
    strike : np.ndarray
    rf
    is_call

    Returns
    -------

    """
    forward_ = np.exp(mu + vol ** 2 / 2)
    res = option_price_bs(strike.reshape(-1, 1), rf,
                          forward=forward_.reshape(1, -1), tau=1,
                          vola=vol.reshape(1, -1), is_call=is_call) \
          @ w

    return res


if __name__ == '__main__':
    pass
