"""Represents a quantity with uncertainty."""

import warnings

import numpy as np

from ._common import round_arr_or_scalar
from ._derivative_table import OPERATIONS
from .uncertainty import Uncertainty


class Measurement:
    """Represents a quantity with uncertainty.

    Arithmetics on `Measurement` should propagate uncertainties correctly
    assuming independence. No rounding on intermediate results until the
    final string conversion.

    Examples
    --------
    Uncertainties are propagated assuming independence:

    >>> val = Measurement(10.12, 1.999) + Measurement(20, 3.1)
    >>> str(val)
    '30 ± 4'

    However, if the two `Measurement`s are the same one, the correlation
    coefficient is assumed to be 1:

    >>> str(val - val)
    '0 ± 0'
    >>> str(val + val)
    '60 ± 7'

    The Python `__repr__` of `Measurement` retains the full precision while
    still letting the user see the rounded values:

    >>> val
    Measurement(30, 4, full_center=30.119999999999997, full_uncert=3.6886313179823222)

    This allows the repr to be used to recreate the object:

    >>> Measurement(10, 2, full_center=123.456, full_uncert=0.0156)
    Measurement(123.456, 0.016, full_center=123.456, full_uncert=0.0156)

    Also useful for formatting a single value with uncertainty:

    >>> import numpy as np
    >>> arr = np.array([1.623, 2.123, 2.623])
    >>> str(Measurement(np.mean(arr), np.std(arr)))
    '2.1 ± 0.4'

    Basic arithmetic operations work (also assuming independence):

    >>> str(Measurement(10.12, 1.999) * Measurement(20, 1.1))
    '200 ± 40'
    >>> str(10 * Measurement(20, 1.1))
    '200 ± 11'
    >>> str(Measurement(10.12, 1.999) / Measurement(20, 1.1))
    '0.51 ± 0.10'
    >>> str(1 / Measurement(10, 1))
    '0.100 ± 0.010'

    NumPy functions are supported with uncertainties propagated:

    >>> np.arccos(Measurement(0.5, 0.1))
    Measurement(1.05, 0.12, full_center=1.0471975511965976, full_uncert=0.11547005383792518)

    There is also array-type `Measurement`:

    >>> mar = Measurement(np.arange(5), np.arange(0.1, 0.3, 0.04))
    >>> str(mar)
    '[0.00 ± 0.10, 1.00 ± 0.14, 2.00 ± 0.18, 3.0 ± 0.2, 4.0 ± 0.3]'

    They work just like arrays:

    >>> mar[2]
    Measurement(2.00, 0.18, full_center=2, full_uncert=0.18000000000000002)
    >>> str(mar[4])
    '4.0 ± 0.3'

    Array-type `Measurement` supports NumPy-like arithmetic directly:

    >>> 3 * mar
    Measurement([0.0, 3.0, 6.0, 9.0, 12.0], [0.3, 0.4, 0.5, 0.7, 0.8], full_center=[0, 3, 6, 9, 12], full_uncert=[0.30000000000000004, 0.42000000000000004, 0.54, 0.6600000000000001, 0.78])

    Array-type `Measurement` can be converted to and from a list of `Measurement`:

    >>> lm = mar.as_simple_list()
    >>> lm
    [Measurement(0.00, 0.10, full_center=0, full_uncert=0.1), Measurement(1.00, 0.14, full_center=1, full_uncert=0.14), Measurement(2.00, 0.18, full_center=2, full_uncert=0.18000000000000002), Measurement(3.0, 0.2, full_center=3, full_uncert=0.22000000000000003), Measurement(4.0, 0.3, full_center=4, full_uncert=0.26)]
    >>> Measurement.from_simple_list(lm)
    Measurement([0.00, 1.00, 2.00, 3.0, 4.0], [0.10, 0.14, 0.18, 0.2, 0.3], full_center=[0, 1, 2, 3, 4], full_uncert=[0.1, 0.14, 0.18000000000000002, 0.22000000000000003, 0.26])

    Python element operations are supported:

    >>> mar[1] = Measurement(10, 1)
    >>> str(mar)
    '[0.00 ± 0.10, 10.0 ± 1.0, 2.00 ± 0.18, 3.0 ± 0.2, 4.0 ± 0.3]'
    >>> mar[2] = (9, 0.154)
    >>> str(mar)
    '[0.00 ± 0.10, 10.0 ± 1.0, 9.00 ± 0.15, 3.0 ± 0.2, 4.0 ± 0.3]'
    >>> mar[1] += Measurement(10, 1)
    >>> str(mar)
    '[0.00 ± 0.10, 20.0 ± 1.4, 9.00 ± 0.15, 3.0 ± 0.2, 4.0 ± 0.3]'
    >>> del mar[1]
    >>> str(mar)
    '[0.00 ± 0.10, 9.00 ± 0.15, 3.0 ± 0.2, 4.0 ± 0.3]'
    """

    def __init__(self, center, uncert, full_center=None, full_uncert=None):
        # These to allow useful `repr` while still upholding the contract
        # of outputting a representation that can be used to recreate the object
        if full_center is not None:
            center = full_center
        if full_uncert is not None:
            uncert = full_uncert
        self.center = np.asarray(center)
        if isinstance(uncert, Uncertainty):
            # No conversion needed
            self.uncert = uncert
        else:
            self.uncert = Uncertainty(uncert)
        if self.uncert.is_array_type() and len(self.center) != len(self.uncert):
            raise ValueError("The lengths of `center` and `uncert` must match")
        # array-type uncert implies array center,
        # but array center does not imply array-type uncert
        # Generate NumPy hooks for all the functions
        self._make_numpy_hook_functions()
        # Generate comparison methods
        self._make_comparison_methods()

    def get_rounded_center(self):
        """Get the rounded center value of self."""
        npow = self.uncert.get_significant_digit()
        return round_arr_or_scalar(self.center, npow)

    def get_rounded_uncert(self):
        """Get the rounded uncertainty of self."""
        return self.uncert.get_rounded_value()

    def is_array_type(self):
        """Check if this `Measurement` is an array or a scalar."""
        return len(self.center.shape) != 0

    def as_simple_list(self):
        """Convert an array `Measurement` to a scalar `Measurement` list."""
        if not self.is_array_type():
            return self
        return list(iter(self))

    @classmethod
    def from_simple_list(cls, items):
        """Create an array `Measurement` from a scalar `Measurement` list."""
        return cls([x.center for x in items], Uncertainty.from_simple_list([x.uncert for x in items]))

    def __iter__(self):
        return map(lambda x: Measurement(x[0], x[1]), zip(self.center, self.uncert))

    def __getitem__(self, idx):
        return Measurement(self.center[idx], self.uncert[idx])

    def __setitem__(self, idx, value):
        if isinstance(value, Measurement):
            self.center[idx] = value.center
            self.uncert[idx] = value.uncert
        else:
            self.center[idx] = value[0]
            self.uncert[idx] = value[1]

    def __delitem__(self, idx):
        self.center = np.delete(self.center, idx)
        if self.uncert.is_array_type():
            del self.uncert[idx]

    def __len__(self):
        return len(self.center)

    def extend(self, other):
        """Extend the array-type `Measurement` with another `Measurement`.

        Examples
        --------
        >>> a = Measurement([1, 2], [0.1, 0.2])
        >>> b = Measurement([3, 4], [0.3, 0.4])
        >>> a.extend(b)
        >>> a
        Measurement([1.00, 2.0, 3.0, 4.0], [0.10, 0.2, 0.3, 0.4], full_center=[1, 2, 3, 4], full_uncert=[0.1, 0.2, 0.3, 0.4])
        """
        if not self.is_array_type():
            raise ValueError("Cannot extend a scalar Measurement")
        if not other.is_array_type():
            raise ValueError("Cannot extend with a scalar Measurement (use `append` instead)")
        self.center = np.concatenate((self.center, other.center))
        if self.uncert.is_array_type():
            self.uncert.extend(other.uncert)
        else:
            self.uncert.append(other.uncert)

    def append(self, other):
        """Append a scalar `Measurement` to this array-type `Measurement`.

        Examples
        --------
        >>> a = Measurement([1, 2], [0.1, 0.2])
        >>> b = Measurement(3, 0.3)
        >>> a.append(b)
        >>> a
        Measurement([1.00, 2.0, 3.0], [0.10, 0.2, 0.3], full_center=[1, 2, 3], full_uncert=[0.1, 0.2, 0.3])
        """
        if not self.is_array_type():
            raise ValueError("Cannot append to a scalar Measurement")
        if other.is_array_type():
            raise ValueError("Cannot append an array Measurement (use `extend` instead)")
        self.center = np.append(self.center, other.center)
        self.uncert.append(other.uncert)

    @staticmethod
    def _shared_stringify(center, uncert):
        # We do not use `get_rounded_x` here to save one round of computation
        npow = uncert.get_significant_digit()
        center = np.round(center, npow)
        uncertstr = str(uncert)
        if npow >= 0:
            centerstr = f"{center:.{npow}f}"
        else:
            # npow negative => keep only int part
            centerstr = str(int(center))
        return centerstr, uncertstr

    def __str__(self):
        individual_formatter = "{0} ± {1}"
        if self.is_array_type():
            if self.uncert.is_array_type():
                return "[" + ", ".join(
                    individual_formatter.format(*self._shared_stringify(c, u)) for c, u in zip(self.center, self.uncert)
                ) + "]"
            return "[" + ", ".join(
                individual_formatter.format(*self._shared_stringify(c, self.uncert)) for c in self.center
            ) + "]"
        return individual_formatter.format(*self._shared_stringify(self.center, self.uncert))

    def __repr__(self):
        if self.is_array_type():
            data = [self._shared_stringify(c, u) for c, u in zip(self.center, self.uncert)]
            full_center = self.center.tolist()
            # Either `float` or `list[float]`
            full_uncert = self.uncert.u.tolist()
            centerstrs, uncertstrs = zip(*data)
            centerstr = ", ".join(centerstrs)
            uncertstr = ", ".join(uncertstrs)
            return f"Measurement([{centerstr}], [{uncertstr}], full_center={full_center}, full_uncert={full_uncert})"
        centerstr, uncertstr = self._shared_stringify(self.center, self.uncert)
        return f"Measurement({centerstr}, {uncertstr}, full_center={self.center}, full_uncert={self.uncert.u})"

    def _repr_pretty_(self, p, cycle):
        """Pretty-print for IPython."""
        p.text(str(self) if not cycle else '...')

    @staticmethod
    def _check_other_is_us(other):
        if not isinstance(other, Measurement):
            raise TypeError("Use normal Python operators instead")

    def add_with_correlation(self, other, r=0.0):
        """Add two `Measurement`s with the given correlation coefficient."""
        self._check_other_is_us(other)
        new_uncert = self.uncert.add_uncert(other.uncert, r=r)
        return Measurement(self.center + other.center, new_uncert)

    def __add__(self, other):
        if other is self:
            # Always full correlation in this case
            return self.add_with_correlation(other, r=1)
        if isinstance(other, Measurement):
            return self.add_with_correlation(other)
        # Assume `other` is a pure number
        return Measurement(self.center + other, self.uncert)

    def __radd__(self, other):
        # if isinstance(other, Measurement):
        #     unreachable: Python should call other's __add__
        # Assume `other` is a pure number
        return Measurement(other + self.center, self.uncert)

    def sub_with_correlation(self, other, r=0.0):
        """Subtract two `Measurement`s with the given correlation coefficient."""
        self._check_other_is_us(other)
        new_uncert = self.uncert.add_uncert(other.uncert, r=-r)
        return Measurement(self.center - other.center, new_uncert)

    def __sub__(self, other):
        if other is self:
            # Always full correlation in this case
            return self.sub_with_correlation(other, r=1)
        if isinstance(other, Measurement):
            return self.sub_with_correlation(other)
        # Assume `other` is a pure number
        return Measurement(self.center - other, self.uncert)

    def __rsub__(self, other):
        # if isinstance(other, Measurement):
        #     unreachable: Python should call other's __sub__
        # Assume `other` is a pure number
        return Measurement(other - self.center, self.uncert)

    def mul_with_correlation(self, other, r=0.0):
        """Multiply two `Measurement`s with the given correlation coefficient."""
        self._check_other_is_us(other)
        # u(f)**2 = (partial(f,a)u(a))**2+(partial(f,b)u(b))**2+corrterm
        # u(f)**2 = (u(a)b)**2 + (u(b)a)**2+corrterm
        # (u(f)/f)**2 = (u(a)/a)**2 + (u(b)/b)**2+corrterm/(ab)**2
        new_reluncert = (
            self.uncert/self.center).add_uncert(other.uncert/other.center, r=r)
        new_center = self.center * other.center
        return Measurement(new_center, new_reluncert * new_center)

    def __mul__(self, other):
        if other is self:
            # Always full correlation in this case
            return self.mul_with_correlation(other, r=1)
        if isinstance(other, Measurement):
            return self.mul_with_correlation(other)
        # Assume `other` is a pure number
        return Measurement(self.center * other, self.uncert * other)

    def __rmul__(self, other):
        # if isinstance(other, Measurement):
        #     unreachable: Python should call other's __mul__
        # Assume `other` is a pure number
        return Measurement(other * self.center, other * self.uncert)

    def truediv_with_correlation(self, other, r=0.0):
        """Multiply two `Measurement`s with the given correlation coefficient."""
        self._check_other_is_us(other)
        new_center = self.center / other.center
        new_reluncert = (self.uncert/self.center).add_uncert(
            other.uncert / other.center, r=-r)
        return Measurement(new_center, new_reluncert * new_center)

    def __truediv__(self, other):
        if other is self:
            # Always full correlation in this case
            return self.truediv_with_correlation(other, r=1)
        if isinstance(other, Measurement):
            return self.truediv_with_correlation(other)
        # Assume `other` is a pure number
        return Measurement(self.center / other, self.uncert / other)

    def __rtruediv__(self, other):
        # if isinstance(other, Measurement):
        #     unreachable: Python should call other's __mul__
        # Assume `other` is a pure number
        new_center = other / self.center
        reluncert = self.uncert / self.center
        return Measurement(new_center, reluncert * new_center)

    def __floordiv__(self, other):
        # Does not really make much sense to produce an uncertainty for this
        return self.center // other

    def __rfloordiv__(self, other):
        # Does not really make much sense to produce an uncertainty for this
        return other // self.center

    # I'll leave it for Python to implement the default in-place methods

    def __abs__(self):
        return Measurement(abs(self.center), self.uncert)

    def tscore(self, other, r=0.0):
        """Compute the t-score between two `Measurement`s.

        Examples
        --------
        >>> a = Measurement(10, 1)
        >>> b = Measurement(11, 1)
        >>> a.tscore(b)
        0.7071067811865475

        The other parameter does not have to be a `Measurement` (in which
        case it is assumed to be exact):
        >>> a.tscore(11)
        1.0

        `tscore` also accepts a given correlation coefficient:
        >>> a.tscore(b, r=1)
        0.5
        """
        if isinstance(other, Measurement):
            diff = self.sub_with_correlation(other, r=-r)
        else:
            # This might raise TypeError if `other` is not to be added
            diff = self - other
        return (abs(diff.center) / diff.uncert.u).tolist()

    def _make_numpy_hook_functions(self):
        """Generate NumPy hooks for supported numerical operations."""
        for funcname, deriv in OPERATIONS:
            if not hasattr(np, funcname):
                continue
            # Make sure `funcname` and `deriv` are captured in the closure
            def generate_func(funcname, deriv):
                func = getattr(np, funcname)
                def helper_func(self, *args, **kwargs):
                    # This is the actual operation
                    new_center = func(self.center, *args, **kwargs)
                    new_uncert = self.uncert.u * np.abs(deriv(self.center))
                    return Measurement(new_center, new_uncert)
                helper_func.__name__ = funcname
                helper_func.__doc__ = f"Apply `np.{funcname}` to the center value and propagate uncertainty."
                helper_func.__qualname__ = f"Measurement.{funcname}"
                return helper_func
            helper_func = generate_func(funcname, deriv)
            setattr(Measurement, funcname, helper_func)

    def _make_comparison_methods(self):
        """Generate comparison methods for `Measurement`."""
        for operation in ("lt", "le", "eq", "ne", "gt", "ge"):
            method_name = f"__{operation}__"
            # Make sure `method_name` is captured in the closure
            def generate_method(method_name):
                def comparison_method(self, other):
                    if isinstance(other, Measurement):
                        warnings.warn("Comparison of measurements compares the center value only."
                                      " For statistical comparison, use `Measurement.tscore`")
                        return getattr(self.center, method_name)(other.center)
                    return getattr(self.center, method_name)(other)
                comparison_method.__name__ = method_name
                comparison_method.__qualname__ = f"Measurement.{method_name}"
                return comparison_method
            comparison_method = generate_method(method_name)
            setattr(self, method_name, comparison_method)
