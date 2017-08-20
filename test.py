#!/usr/bin/env python

import unittest
import numpy as np
from stencils import Stencil


class TestStencil(unittest.TestCase):

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

    def test_int(self):

        # Trapezoidal rule
        st = Stencil([0, 1])
        self.assertTrue(np.allclose(st.integral_weights(1), [0.5, 0.5]))
        # Simpson's rule
        st = Stencil([0, 1, 2])
        self.assertTrue(np.allclose(st.integral_weights(2)*6, [1, 4, 1]))

if __name__ == '__main__':
    unittest.main()
