"""Uncertainty for measurements."""

import warnings
from collections.abc import Callable, Iterable, Iterator
from typing import Any, SupportsIndex, SupportsRound, cast, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._common import (FloatOrArray, IntegerOrArray, get_significant_digit_one,
                      round_arr_or_scalar)


class Uncertainty:
    """An uncertainty that gives correct string printout.

    Supports addition with other uncertainties with a given correlation
    coefficient and the full floating point precision is kept until we
    convert this object to a string.

    If the content is an array, this type will internally represent the
    data as a NumPy array. Otherwise, the internal representation is a
    `np.float64`.

    Examples
    --------
    `str` keeps only one significant digit:

    >>> u = Uncertainty(9123)
    >>> str(u)
    '9000'

    But if the leading digit is 1, `str` keeps two siginificant digits:

    >>> u = Uncertainty(1.1243)
    >>> str(u)
    '1.1'
    >>> u = Uncertainty(0.104)
    >>> str(u)
    '0.10'

    Edge case behaviour:
    >>> u = Uncertainty(0.198)
    >>> str(u)
    '0.2'
    >>> u = Uncertainty(1.96)
    >>> str(u)
    '2'

    Adding `Uncertainty` is done in quadrature by default:

    >>> Uncertainty(1.14923) + Uncertainty(0.84213)
    Uncertainty(1.4, full=1.4247499885243025)

    Specify a custom correlation coefficient with `add_uncert`:

    >>> Uncertainty(1.14923).add_uncert(Uncertainty(0.84213), r=1)
    Uncertainty(2, full=1.99136)

    `Uncertainty` can be multiplied or divided with/by a scalar:

    >>> 2 * Uncertainty(13)
    Uncertainty(30, full=26.0)
    >>> Uncertainty(36) / 7
    Uncertainty(5, full=5.142857142857143)

    One can convert between array-type `Uncertainty` and a list of
    `Uncertainty`:

    >>> Uncertainty([1, 2, 15, 23]).as_simple_list()
    [Uncertainty(1.0, full=1.0), Uncertainty(2, full=2.0), Uncertainty(15, full=15.0), Uncertainty(20, full=23.0)]
    >>> Uncertainty.from_simple_list([Uncertainty(1), Uncertainty(2), Uncertainty(15), Uncertainty(23)])
    Uncertainty([1.0, 2, 15, 20], full=[1.0, 2.0, 15.0, 23.0])

    Array-type `Uncertainty` supports NumPy-like arithmetic directly:

    >>> 3 * Uncertainty([10, 10]) + Uncertainty([10, 10])
    Uncertainty([30, 30], full=[31.622776601683793, 31.622776601683793])

    Arithmetic between scalar and array `Uncertainty` threads like NumPy operations:

    >>> Uncertainty([10, 10]) + Uncertainty(5)
    Uncertainty([11, 11], full=[11.180339887498949, 11.180339887498949])
    """

    def __init__(
            self,
            uncert: ArrayLike,
            full: ArrayLike | None = None,
    ):
        # These to allow useful `repr` while still upholding the contract
        # of outputting a representation that can be used to recreate the object
        if full is not None:
            uncert = full
        # Fix negative inputs
        self.u: np.float64 | NDArray[np.float64] = abs(np.asarray(uncert, dtype=np.float64))
        # Generate comparison methods
        self._make_comparison_methods()

    def get_significant_digit(self) -> IntegerOrArray:
        """Get the negative index of MSD for rounding uncertainties.

        Find $n$ such that $10^{-n}$ is the decimal weight of the most
        significant digit of `self`, unless when that digit is one, in which
        case the resulting $n$ shall correspond to the next digit.

        Returns
        -------
        n : int or ndarray[int]
            The index as described above, useful for passing into `round`.
        """
        return get_significant_digit_one(self.u)

    def get_value(self) -> FloatOrArray:
        """Get the underlying uncertainty value."""
        return self.u

    def get_rounded_value(self) -> float | NDArray[np.floating[Any]]:
        """Get the underlying uncertainty value after rounding."""
        npow = self.get_significant_digit()
        return round_arr_or_scalar(self.u, npow)

    def is_array_type(self) -> bool:
        """Check if this `Uncertainty` is an array or a scalar."""
        return not isinstance(self.u, np.float64)

    def as_simple_list(self) -> "list[Uncertainty] | Uncertainty":
        """Convert an array `Uncertainty` to a scalar `Uncertainty` list."""
        if not self.is_array_type():
            return self
        return list(iter(self))

    @classmethod
    def from_simple_list(cls, items: Iterable["Uncertainty"]) -> "Uncertainty":
        """Create an array `Uncertainty` from a scalar `Uncertainty` list."""
        return cls([x.u for x in items])

    def __iter__(self) -> Iterator["Uncertainty"]:
        return map(Uncertainty, self.u)

    def __getitem__(self, idx: int) -> "Uncertainty":
        return Uncertainty(self.u[idx])

    def __setitem__(self, idx: int, value: "Uncertainty | float | np.floating[Any]") -> None:
        if isinstance(value, Uncertainty):
            value = float(value.u)
        self.u[idx] = value

    def __delitem__(self, idx: int) -> None:
        self.u = np.delete(self.u, idx)

    def __len__(self) -> int:
        return len(self.u)

    def extend(self, other: "Uncertainty | ArrayLike") -> None:
        """Extend this `Uncertainty` with another `Uncertainty` or list.

        Examples
        --------
        >>> u = Uncertainty([1, 2, 3])
        >>> u.extend(Uncertainty([4, 5]))
        >>> u
        Uncertainty([1.0, 2, 3, 4, 5], full=[1.0, 2.0, 3.0, 4.0, 5.0])
        >>> u.extend([5])
        >>> u
        Uncertainty([1.0, 2, 3, 4, 5, 5], full=[1.0, 2.0, 3.0, 4.0, 5.0, 5.0])
        """
        if not self.is_array_type():
            raise ValueError("Cannot extend a scalar Uncertainty")
        if not isinstance(other, Uncertainty):
            other = Uncertainty(other)
        if not other.is_array_type():
            raise ValueError(
                "Cannot extend with a scalar Uncertainty (use `append` instead)")
        self.u = np.concatenate((self.u, other.u))

    def append(self, other: "Uncertainty" | ArrayLike) -> None:
        """Append a scalar `Uncertainty` to this array-type `Uncertainty`.

        Examples
        --------
        >>> u = Uncertainty([1, 2, 3])
        >>> u.append(4)
        >>> u
        Uncertainty([1.0, 2, 3, 4], full=[1.0, 2.0, 3.0, 4.0])
        >>> u.append(Uncertainty(5))
        >>> u
        Uncertainty([1.0, 2, 3, 4, 5], full=[1.0, 2.0, 3.0, 4.0, 5.0])
        """
        if not self.is_array_type():
            raise ValueError("Cannot append to a scalar Uncertainty")
        if not isinstance(other, Uncertainty):
            other = Uncertainty(other)
        if other.is_array_type():
            raise ValueError(
                "Cannot append an array Uncertainty (use `extend` instead)")
        self.u = np.append(self.u, other.u)

    def __str__(self) -> str:
        def str_one(u: float | np.float64, npow: int) -> str:
            uncert = round(u, npow)
            if npow >= 0:
                return f"{uncert:.{npow}f}"
            # npow negative => keep only int part
            return str(int(uncert))
        npow = self.get_significant_digit()
        if self.is_array_type():
            npow = cast(NDArray[np.integer[Any]], npow)
            return "[" + ", ".join(str_one(u, n) for u, n in zip(self.u, npow)) + "]"
        return str_one(self.u, cast(int, npow))

    def __repr__(self) -> str:
        # I don't like NumPy's default repr so `tolist`
        # it also works on scalars because we store them as `np.float64`
        return f"Uncertainty({self}, full={self.u.tolist()})"

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Pretty-print for IPython."""
        p.text(str(self) if not cycle else '...')

    def add_uncert(
            self,
            other: "Uncertainty | NDArray[Any]",
            r: float | NDArray[Any] = 0.0
    ) -> "Uncertainty":
        """Add two uncertainties assuming a given correlation coefficient.

        Parameters
        ----------
        other : Uncertainty or int or float or ndarray
            The other uncertainty to add.
        r : int or float or ndarray, optional
            The correlation coefficient between the two measurements. The
            default is 0 (no correlation).

        Returns
        -------
        Uncertainty
            Resulting uncertainty.
        """
        if isinstance(other, Uncertainty):
            other_u = other.u
        else:
            other_u = other
        m = self.u**2 + other_u**2 + 2 * self.u * other_u * r
        return Uncertainty(m ** 0.5)

    def __add__(self, other: "Uncertainty | NDArray[Any]") -> "Uncertainty":
        # Assume independence
        return self.add_uncert(other)

    def __radd__(self, other: "Uncertainty | NDArray[Any]") -> "Uncertainty":
        # Assume independence
        return self.add_uncert(other)

    def __iadd__(self, other: "Uncertainty | NDArray[Any]") -> "Uncertainty":
        # Assume independence
        self.u = self.add_uncert(other).u
        return self

    def __mul__(self, other: float | int | NDArray[Any]) -> "Uncertainty":
        return Uncertainty(self.u * other)

    def __rmul__(self, other: float | int | NDArray[Any]) -> "Uncertainty":
        return Uncertainty(other * self.u)

    def __imul__(self, other: float | int | NDArray[Any]) -> "Uncertainty":
        self.u *= other
        return self

    def __truediv__(self, other: float | int | NDArray[Any]) -> "Uncertainty":
        return Uncertainty(self.u / other)
        # no r*div

    def __itruediv__(self, other: float | int | NDArray[Any]) -> "Uncertainty":
        self.u /= other
        return self

    def __floordiv__(self, other: float | int | NDArray[Any]) -> "Uncertainty":
        warnings.warn("Are you sure you want to floordiv an uncertainty?")
        return Uncertainty(self.u // other)
        # no r*div

    def __ifloordiv__(self, other: float | int | NDArray[Any]) -> "Uncertainty":
        warnings.warn("Are you sure you want to floordiv an uncertainty?")
        self.u //= other
        return self

    def __int__(self) -> int:
        return int(self.u)

    def __float__(self) -> float:
        return float(self.u)

    def _make_comparison_methods(self) -> None:
        """Generate comparison methods for `Measurement`."""
        for operation in ("lt", "le", "eq", "ne", "gt", "ge"):
            method_name = f"__{operation}__"
            # Make sure `method_name` is captured in the closure

            def generate_method(method_name: str) -> Callable[["Uncertainty", Any], bool]:
                def comparison_method(self: "Uncertainty", other: Any) -> bool:
                    if isinstance(other, Uncertainty):
                        result = getattr(self.u, method_name)(other.u)
                    else:
                        result = getattr(self.u, method_name)(other)
                    return cast(bool, result)
                comparison_method.__name__ = method_name
                comparison_method.__qualname__ = f"Uncertainty.{method_name}"
                return comparison_method
            comparison_method = generate_method(method_name)
            setattr(self, method_name, comparison_method)
