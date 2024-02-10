import numpy as np

# Not going to depend on Scipy just for the factorial...


def fac(n):
    if n < 0:
        raise ValueError('Factorial of a negative number is invalid')
    else:
        return np.prod(np.arange(1, n+1))


class Stencil(object):
    """
    A Stencil is a list of integers that work as relative indices in an array
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
        
        for i in range(n+1):
            cf = self.difference_weights(i, True)
            cint += cf/(i+1.0)

        return cint

    def difference_matrix(self, n, m, h, div_fac=False, fix_edge=True):
        """
        Return an lxl matrix for differentiation of order n of a function
        evaluated on l points:

        A.f = df/dx

        Arguments:
        |   n (int): order of requested derivative
        |   m (int): size of the matrix
        |   h (float): evaluation step
        |   div_fac (bool):  if True, return the cohefficients of the
        |                    derivative divided by n!. This is convenient for
        |                    use in Taylor series (default False).
        |   fix_edge (bool): if True, set to zero all rows corresponding to
        |                    edge points (default True).

        Returns:
        |   diff_matrix (np.ndarray): matrix that if dotted to an array will
        |                             compute its derivative

        """

        A = np.zeros((m, m))
        weights = self.difference_weights(n, div_fac)

        for s, w in zip(self.stencil, weights):
            A += np.diag([w]*(m-abs(s)), k=s)

        # Fix edges
        if fix_edge:
            A *= np.where(np.isclose(np.sum(A, axis=1), 0), 1, 0)[:, None]

        return A/h**n

    def derive(self, x, y, n=1):
        """
        Compute the derivative of y in x at the order n using this stencil.
        Warning: the derivative may be unreliable at the edges.

        Arguments:
        |   x (np.ndarray): x axis (assumed to be equally spaced)
        |   y (np.ndarray): y axis (function evaluated at points x)
        |   n (int): order of the desired derivative

        Returns:
        | dny_dxn (np.ndarray): derivative

        """

        m = len(x)
        h = x[1]-x[0]

        return np.dot(self.difference_matrix(n, m, h), y)

    def integral_matrix(self, n, m, h, fix_edge=True):
        """
        Return an lxl matrix for integration of a function
        evaluated on l points:
               _ 
              | 
        A.f = |  f dx
             _| 

        Arguments:
        |   n (int): order of the Taylor approximation to use.
        |   m (int): size of the matrix
        |   h (float): evaluation step
        |   fix_edge (bool): if True, normalize all rows corresponding to
        |                    edge points (default True).

        Returns:
        |   int_matrix (np.ndarray): matrix that if dotted to an array will
        |                            compute its integral

        """

        A = np.zeros((m, m))
        weights = self.integral_weights(n)

        for s, w in zip(self.stencil, weights):
            A += np.diag([w]*(m-abs(s)), k=s)

        # Fix edges
        if fix_edge:
            A /= np.sum(A, axis=1)[:,None]

        # Cumulation
        A = np.dot(np.tril(np.ones((m, m))), A)

        return A*h

    def integrate(self, x, y, n=1, initial=0.0):
        """
        Compute the integral of y in x at the order n using this stencil.
        Warning: the integral may be unreliable at the edges.

        Arguments:
        |   x (np.ndarray): x axis (assumed to be equally spaced)
        |   y (np.ndarray): y axis (function evaluated at points x)
        |   n (int): order of the desired Taylor approximation

        Returns:
        | Y (np.ndarray): integral

        """

        m = len(x)
        h = x[1]-x[0]
        
        y_int = self.integral_matrix(n, m, h)[:-1]@y
        y_int = np.concatenate([[initial], y_int+initial])

        return y_int
