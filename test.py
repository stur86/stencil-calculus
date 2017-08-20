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

if __name__ == '__main__':
    unittest.main()
