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

    Arguments:
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

    def difference_weights(self, n, div_fac=False):
        """ 
        Return the weight cohefficients w_i for computing the derivative of
        order n with the given stencil, such that

        d^n f      1
        -----  = ---- (w_1*f(x+s_1*h) + w_2*f(x+s_2*h) + ... )
        d x^n     h^n

        where the s_i are the indices of the stencil and h the step.

        Arguments:
        |   n (int): order of requested derivative
        |   div_fac (bool): if True, return the cohefficients of the
        |                   derivative divided by n!. This is convenient for
        |                   use in Taylor series.

        Returns:
        |   weights (np.ndarray): differentiation weights

        """

        if n >= self._L:
            raise ValueError(
                'Stencil length must be greater than derivative order n')

        s = self.stencil

        A = s[None, :]**np.arange(0, self._L)[:, None]
        b = np.zeros(self._L)
        b[n] = 1 if div_fac else fac(n)
        return np.linalg.solve(A, b)

    def integral_weights(self, n):
        """
        Return the weight cohefficients w_i for computing the integral of f
        with a Taylor expansion of order n and the given stencil, such that

          _
         | x+h
         | 
         |     f(x) dx = (w_1*f(x+s_1*h) + w_2*f(x+s_2*h) + ... )*h
        _| x


        where the s_i are the indices of the stencil and h the step.

        Arguments:
        |   n (int): order of the Taylor approximation to use.

        Returns:
        |   weights (np.ndarray): integration weights

        """

        cint = np.zeros(self._L)
        s_ext = max(self.stencil)-min(self.stencil)  # Extent of the stencil

        for i in range(n+1):
            cf = self.difference_weights(i, True)
            cint += cf/(i+1.0)*(s_ext)**i

        return cint
