#!/usr/bin/env python

import unittest
import numpy as np
from stencils import Stencil


class TestStencil(unittest.TestCase):

    test_funcs = {
        'x': {
            'f': lambda x: x,
            'df': {
                1: lambda x: 1+0*x,
            },
            'F': lambda x: 0.5*x**2
        },
        'sin(x)': {
            'f': np.sin,
            'df': {
                1: np.cos,
                2: lambda x: -np.sin(x)
            },
            'F': lambda x: -np.cos(x)
        },
        'exp(-x)': {
            'f': lambda x: np.exp(-x),
            'df': {
                1: lambda x: -np.exp(-x),
                2: lambda x: np.exp(-x),
            },
            'F': lambda x: -np.exp(-x),
        },
    }

    def test_badarg(self):

        with self.assertRaises(ValueError):
            Stencil([1, 2, 3.5])

        with self.assertRaises(ValueError):
            Stencil('hi')

        with self.assertRaises(ValueError):
            Stencil([[1, 2], [3, 4]])

    def test_fac(self):

        from stencils.stencils import fac

        self.assertEqual(fac(0), 1)
        self.assertEqual(fac(4), 24)

    def test_diff(self):

        st = Stencil([-1, 0, 1])
        self.assertTrue(np.allclose(st.difference_weights(1), [-0.5, 0, 0.5]))
        self.assertTrue(np.allclose(st.difference_weights(2), [1, -2, 1]))

        # Differentiation functions to test
        x = np.linspace(0, 10, 1000)
        h = x[1]-x[0]

        for fname, fdef in self.test_funcs.iteritems():
            f = fdef['f'](x)
            for n in fdef['df']:
                df = fdef['df'][n](x)
                self.assertTrue(np.allclose(df[1:-1],
                                            st.derive(x, f, n)[1:-1],
                                            rtol=1e-3))

    def test_int(self):

        # Trapezoidal rule
        st = Stencil([0, 1])
        self.assertTrue(np.allclose(st.integral_weights(1), [0.5, 0.5]))
        # Simpson's rule
        st = Stencil([0, 1, 2])
        self.assertTrue(np.allclose(st.integral_weights(2)*6, [1, 4, 1]))

        # Integration functions to test
        x = np.linspace(0, 10, 1000)
        h = x[1]-x[0]

        for fname, fdef in self.test_funcs.iteritems():
            f = fdef['f'](x)
            F = fdef['F'](x)-fdef['F'](x[0])
            self.assertLess(np.sum((F[1:-1]-st.integrate(x, f, 2)[1:-1])**2
                                   )**0.5 /
                            np.sum(F[1:-1]**2)**0.5, 1e-2)

if __name__ == '__main__':
    unittest.main()
