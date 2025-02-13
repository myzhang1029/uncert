"""Common helper functions for the package."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray


@np.vectorize
def get_significant_digit_one(u: np.floating[Any]) -> int:
    """Get the negative index of MSD for rounding uncertainties.

    See Also
    --------
    Uncertainty.get_significant_digit
    """
    if np.isnan(u) or np.isinf(u):
        warnings.warn("NaN or inf uncertainty encountered", RuntimeWarning)
        return 0
    if u == 0:
        return 0
    absv = abs(u)
    # Rounding away this power of 10 (MSE exponent - 1)
    npow = int(math.floor(math.log10(absv)))
    # Find the most significant digit
    msd = int(absv/10**npow)
    if msd == 1:
        # Keep the next (lower) power of ten
        npow -= 1
        # Check for edge case:
        # If the next two digits will round up, too bad. Erase them.
        # XXX: is there a better way?
        tryround = abs(round(u, -npow))
        trymsd, trynmsd = divmod(int(tryround/10**npow), 10)
        if trymsd == 2 and trynmsd == 0:
            npow += 1
    return -npow


def round_arr_or_scalar(
    num: np.floating[Any] | NDArray[np.floating[Any]],
    digits: np.integer[Any] | NDArray[np.integer[Any]],
) -> float | NDArray[np.floating[Any]]:
    """round(num, digits) or that threaded over np.ndarray

    Examples
    --------
    >>> round_arr_or_scalar(10.123, 1)
    10.1
    >>> round_arr_or_scalar([0.12,0.234,3.0], 2)
    array([0.12, 0.23, 3.  ])
    >>> round_arr_or_scalar([0.12,0.234,3.0], [0, 2, 1])
    array([0.  , 0.23, 3.  ])
    """
    npnum = np.asarray(num)
    npdig = np.asarray(digits)
    if npnum.shape != ():
        if npdig.shape != ():
            if len(npnum) != len(npdig):
                raise ValueError(
                    "The lengths of `num` and `digits` must match")
            return np.array([round(u, n) for u, n in zip(npnum, npdig)])
        # Else just use np.round(arr, scalar)
        return npnum.round(npdig)
    # Both are scalars
    return round(float(npnum), int(npdig))
