"""Numerical derivatives of common NumPy functions."""

import numpy as np

# NumPy math ufuncs and their (weak) derivatives
OPERATIONS = (
    (np.cos, lambda x: -np.sin(x)),
    (np.sin, np.cos),
    (np.tan, lambda x: 1/np.cos(x)**2),
    (np.arccos, lambda x: -1/np.sqrt(1-x**2)),
    (np.acos, lambda x: -1/np.sqrt(1-x**2)),
    (np.arcsin, lambda x: 1/np.sqrt(1-x**2)),
    (np.asin, lambda x: 1/np.sqrt(1-x**2)),
    (np.arctan, lambda x: 1/(1+x**2)),
    (np.atan, lambda x: 1/(1+x**2)),
    (np.cosh, np.sinh),
    (np.sinh, np.cosh),
    (np.tanh, lambda x: 1/np.cosh(x)**2),
    (np.arccosh, lambda x: 1/np.sqrt(x**2-1)),
    (np.acosh, lambda x: 1/np.sqrt(x**2-1)),
    (np.arcsinh, lambda x: 1/np.sqrt(x**2+1)),
    (np.asinh, lambda x: 1/np.sqrt(x**2+1)),
    (np.arctanh, lambda x: 1/(1-x**2)),
    (np.atanh, lambda x: 1/(1-x**2)),
    (np.exp, np.exp),
    (np.exp2, lambda x: np.log(2)*2**x),
    (np.expm1, np.exp),
    (np.log, np.reciprocal),
    (np.log2, lambda x: 1/(x*np.log(2))),
    (np.log10, lambda x: 1/(x*np.log(10))),
    (np.log1p, np.reciprocal),
    (np.sqrt, lambda x: 0.5/np.sqrt(x)),
    (np.square, lambda x: 2*x),
    (np.reciprocal, lambda x: -1/x**2),
    (np.negative, lambda x: -1),
    (np.abs, np.sign),
)
