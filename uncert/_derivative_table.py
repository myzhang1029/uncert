"""Numerical derivatives of common NumPy functions."""

import numpy as np

# NumPy math ufuncs and their (weak) derivatives
OPERATIONS = (
    ("cos", lambda x: -np.sin(x)),
    ("sin", np.cos),
    ("tan", lambda x: 1/np.cos(x)**2),
    ("arccos", lambda x: -1/np.sqrt(1-x**2)),
    ("acos", lambda x: -1/np.sqrt(1-x**2)),
    ("arcsin", lambda x: 1/np.sqrt(1-x**2)),
    ("asin", lambda x: 1/np.sqrt(1-x**2)),
    ("arctan", lambda x: 1/(1+x**2)),
    ("atan", lambda x: 1/(1+x**2)),
    ("cosh", np.sinh),
    ("sinh", np.cosh),
    ("tanh", lambda x: 1/np.cosh(x)**2),
    ("arccosh", lambda x: 1/np.sqrt(x**2-1)),
    ("acosh", lambda x: 1/np.sqrt(x**2-1)),
    ("arcsinh", lambda x: 1/np.sqrt(x**2+1)),
    ("asinh", lambda x: 1/np.sqrt(x**2+1)),
    ("arctanh", lambda x: 1/(1-x**2)),
    ("atanh", lambda x: 1/(1-x**2)),
    ("exp", np.exp),
    ("exp2", lambda x: np.log(2)*2**x),
    ("expm1", np.exp),
    ("log", np.reciprocal),
    ("log2", lambda x: 1/(x*np.log(2))),
    ("log10", lambda x: 1/(x*np.log(10))),
    ("log1p", np.reciprocal),
    ("sqrt", lambda x: 0.5/np.sqrt(x)),
    ("square", lambda x: 2*x),
    ("reciprocal", lambda x: -1/x**2),
    ("negative", lambda x: -1),
    ("abs", np.sign),
)
