import numpy as np

# Not going to depend on Scipy just for the factorial...


def fac(n):
    if n < 0:
        raise ValueError('Factorial of a negative number is invalid')
    else:
        return np.prod(np.arange(1, n+1))


class Stencil(object):
    """
    A Stencil is a list of integer that work as relative indices in an array
    of regularly spaced evaluations of a function y. These points are meant to
    be used as cohefficients in finite differences schemes for differentiation
    and integration. For example, the stencil [0, 1, 2] refers to the elements
    f(x), f(x+dx) and f(x+2dx), commonly used in the integration scheme known
    as Simpson's rule. The maximum order allowed by a stencil of length L is
    L-1.

    A Stencil object provides methods to compute cohefficients to integrate or
    differentiate a function using it, as well as the actual integration and
    differentiation methods.

    Parameters:
    |   stencil ([int]): list of integers representing the relative indices
    |                    of the function evaluations to combine.
    """

    def __init__(self, stencil):

        # Check that it is valid
        stencil = np.array(stencil)
        if not (len(stencil.shape) == 1) or not (stencil % 1 == 0).all():
            raise ValueError('Invalid stencil')

        self._stencil = np.array(stencil).astype(int)
        self._L = len(stencil)

    @property
    def stencil(self):
        return self._stencil.copy()

    def difference_weights(self):
        pass


def stencil_coeffs(s, d, div_fac=False):
    s = np.array(s)
    N = len(s)
    if d >= N:
        raise ValueError(
            'Stencil length must be greater than derivative order d')
    A = s[None, :]**np.arange(0, N)[:, None]
    b = np.zeros(N)
    b[d] = 1 if div_fac else factorial(d)
    return np.linalg.solve(A, b)
