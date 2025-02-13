"""Uncertainty for measurements."""

import warnings

import numpy as np

from ._common import get_significant_digit_one, round_arr_or_scalar


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

    The full precision is kept in `__repr__`:

    >>> Uncertainty(1.23456789)
    Uncertainty(1.2, full=1.23456789)

    This allows the repr to be used to recreate the object:

    >>> Uncertainty(99, full=1.23456789)
    Uncertainty(1.2, full=1.23456789)


    Adding `Uncertainty` is done in quadrature by default:

    >>> Uncertainty(1.14923) + Uncertainty(0.84213)
    Uncertainty(1.4, full=1.4247499885243025)

    Specify a custom correlation coefficient with `add_uncert`:

    >>> Uncertainty(1.14923).add_uncert(Uncertainty(0.84213), r=1)
    Uncertainty(2, full=1.99136)

    `Uncertainty` can be multiplied or divided with/by a scalar:

    >>> 2 * Uncertainty(13)
    Uncertainty(30, full=26)
    >>> Uncertainty(36) / 7
    Uncertainty(5, full=5.142857142857143)

    One can convert between array-type `Uncertainty` and a list of
    `Uncertainty`:

    >>> Uncertainty([1, 2, 15, 23]).as_simple_list()
    [Uncertainty(1.0, full=1), Uncertainty(2, full=2), Uncertainty(15, full=15), Uncertainty(20, full=23)]
    >>> Uncertainty.from_simple_list([Uncertainty(1), Uncertainty(2), Uncertainty(15), Uncertainty(23)])
    Uncertainty([1.0, 2, 15, 20], full=[1, 2, 15, 23])

    Array-type `Uncertainty` supports NumPy-like arithmetic directly:

    >>> 3 * Uncertainty([10, 10]) + Uncertainty([10, 10])
    Uncertainty([30, 30], full=[31.622776601683793, 31.622776601683793])

    Arithmetic between scalar and array `Uncertainty` threads like NumPy operations:

    >>> Uncertainty([10, 10]) + Uncertainty(5)
    Uncertainty([11, 11], full=[11.180339887498949, 11.180339887498949])

    Python element operations are supported:

    >>> u = Uncertainty([1, 2, 3])
    >>> u.append(4)
    >>> u
    Uncertainty([1.0, 2, 3, 4], full=[1, 2, 3, 4])
    >>> del u[1]
    >>> u
    Uncertainty([1.0, 3, 4], full=[1, 3, 4])
    >>> u[1] = 2
    >>> u
    Uncertainty([1.0, 2, 4], full=[1, 2, 4])
    """

    def __init__(self, uncert, full=None):
        # These to allow useful `repr` while still upholding the contract
        # of outputting a representation that can be used to recreate the object
        if full is not None:
            uncert = full
        # Fix negative inputs
        self.u = abs(np.asarray(uncert))
        # Generate comparison methods
        self._make_comparison_methods()

    def get_significant_digit(self):
        """Get the negative index of MSD for rounding uncertainties.

        Find $n$ such that $10^{-n}$ is the decimal weight of the most
        significant digit of `self`, unless when that digit is one, in which
        case the resulting $n$ shall correspond to the next digit.

        Returns
        -------
        n : int
            The index as described above, useful for passing into `round`.
        """
        return get_significant_digit_one(self.u)

    def get_rounded_value(self):
        """Get the underlying uncertainty value after rounding."""
        npow = self.get_significant_digit()
        return round_arr_or_scalar(self.u, npow)

    def is_array_type(self):
        """Check if this `Uncertainty` is an array or a scalar."""
        return len(self.u.shape) != 0

    def as_simple_list(self):
        """Convert an array `Uncertainty` to a scalar `Uncertainty` list."""
        if not self.is_array_type():
            return self
        return list(iter(self))

    @classmethod
    def from_simple_list(cls, items):
        """Create an array `Uncertainty` from a scalar `Uncertainty` list."""
        return cls([x.u for x in items])

    def __iter__(self):
        return map(Uncertainty, self.u)

    def __getitem__(self, idx):
        return Uncertainty(self.u[idx])

    def __setitem__(self, idx, value):
        self.u[idx] = value

    def __delitem__(self, idx):
        self.u = np.delete(self.u, idx)

    def __len__(self):
        return len(self.u)

    def extend(self, other):
        """Extend this `Uncertainty` with another `Uncertainty` or list.

        Examples
        --------
        >>> u = Uncertainty([1, 2, 3])
        >>> u.extend(Uncertainty([4, 5]))
        >>> u
        Uncertainty([1.0, 2, 3, 4, 5], full=[1, 2, 3, 4, 5])
        >>> u.extend([5])
        >>> u
        Uncertainty([1.0, 2, 3, 4, 5, 5], full=[1, 2, 3, 4, 5, 5])
        """
        if not self.is_array_type():
            raise ValueError("Cannot extend a scalar Uncertainty")
        if not isinstance(other, Uncertainty):
            other = Uncertainty(other)
        if not other.is_array_type():
            raise ValueError("Cannot extend with a scalar Uncertainty (use `append` instead)")
        self.u = np.concatenate((self.u, other.u))

    def append(self, other):
        """Append a scalar `Uncertainty` to this array-type `Uncertainty`.

        Examples
        --------
        >>> u = Uncertainty([1, 2, 3])
        >>> u.append(4)
        >>> u
        Uncertainty([1.0, 2, 3, 4], full=[1, 2, 3, 4])
        >>> u.append(Uncertainty(5))
        >>> u
        Uncertainty([1.0, 2, 3, 4, 5], full=[1, 2, 3, 4, 5])
        """
        if not self.is_array_type():
            raise ValueError("Cannot append to a scalar Uncertainty")
        if not isinstance(other, Uncertainty):
            other = Uncertainty(other)
        if other.is_array_type():
            raise ValueError("Cannot append an array Uncertainty (use `extend` instead)")
        self.u = np.append(self.u, other.u)

    def __str__(self):
        def str_one(u, npow):
            uncert = round(u, npow)
            if npow >= 0:
                return f"{uncert:.{npow}f}"
            # npow negative => keep only int part
            return str(int(uncert))
        npow = self.get_significant_digit()
        if self.is_array_type():
            return "[" + ", ".join(str_one(u, n) for u, n in zip(self.u, npow)) + "]"
        return str_one(self.u, npow)

    def __repr__(self):
        # I don't like NumPy's default repr so `tolist`
        # it also works on scalars because we store them as `np.float64`
        return f"Uncertainty({self}, full={self.u.tolist()})"

    def _repr_pretty_(self, p, cycle):
        """Pretty-print for IPython."""
        p.text(str(self) if not cycle else '...')

    def add_uncert(self, other, r=0.0):
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

    def __add__(self, other):
        # Assume independence
        return self.add_uncert(other)

    def __radd__(self, other):
        # Assume independence
        return self.add_uncert(other)

    def __iadd__(self, other):
        # Assume independence
        self.u = self.add_uncert(other).u
        return self

    def __mul__(self, other):
        return Uncertainty(self.u * other)

    def __rmul__(self, other):
        return Uncertainty(other * self.u)

    def __imul__(self, other):
        self.u *= other
        return self

    def __truediv__(self, other):
        return Uncertainty(self.u / other)
        # no r*div

    def __itruediv__(self, other):
        self.u /= other
        return self

    def __floordiv__(self, other):
        warnings.warn("Are you sure you want to floordiv an uncertainty?")
        return Uncertainty(self.u // other)
        # no r*div

    def __ifloordiv__(self, other):
        warnings.warn("Are you sure you want to floordiv an uncertainty?")
        self.u //= other
        return self

    def __int__(self):
        return int(self.u)

    def __float__(self):
        return float(self.u)

    def _make_comparison_methods(self):
        """Generate comparison methods for `Measurement`."""
        for operation in ("lt", "le", "eq", "ne", "gt", "ge"):
            method_name = f"__{operation}__"
            # Make sure `method_name` is captured in the closure
            def generate_method(method_name):
                def comparison_method(self, other):
                    if isinstance(other, Uncertainty):
                        return getattr(self.u, method_name)(other.u)
                    return getattr(self.u, method_name)(other)
                comparison_method.__name__ = method_name
                comparison_method.__qualname__ = f"Uncertainty.{method_name}"
                return comparison_method
            comparison_method = generate_method(method_name)
            setattr(self, method_name, comparison_method)
