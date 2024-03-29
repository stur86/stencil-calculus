#!/usr/bin/env python

import pytest
import numpy as np
from stencils import Stencil


class TestStencil:
    example_funcs = {
        "x": {
            "f": lambda x: x,
            "df": {
                1: lambda x: 1 + 0 * x,
            },
            "F": lambda x: 0.5 * x**2,
        },
        "sin(x)": {
            "f": np.sin,
            "df": {1: np.cos, 2: lambda x: -np.sin(x)},
            "F": lambda x: -np.cos(x),
        },
        "exp(-x)": {
            "f": lambda x: np.exp(-x),
            "df": {
                1: lambda x: -np.exp(-x),
                2: lambda x: np.exp(-x),
            },
            "F": lambda x: -np.exp(-x),
        },
    }

    def test_badarg(self):
        with pytest.raises(ValueError):
            Stencil([1, 2, 3.5])

        with pytest.raises(ValueError):
            Stencil("hi")

        with pytest.raises(ValueError):
            Stencil([[1, 2], [3, 4]])

    def test_fac(self):
        from stencils.stencils import fac

        assert fac(0) == 1
        assert fac(4) == 24

    def test_diff(self):
        st = Stencil([-1, 0, 1])
        assert np.allclose(st.difference_weights(1), [-0.5, 0, 0.5])
        assert np.allclose(st.difference_weights(2), [1, -2, 1])

        # Differentiation functions to test
        x = np.linspace(0, 10, 1000)

        for fdef in self.example_funcs.values():
            f = fdef["f"](x)
            for n in fdef["df"]:
                df = fdef["df"][n](x)
                assert np.allclose(df[1:-1], st.derive(x, f, n)[1:-1], rtol=1e-3)

    def test_int(self):
        # Trapezoidal rule
        st = Stencil([0, 1])
        assert np.allclose(st.integral_weights(1), [0.5, 0.5])
        # Three point stencil
        st = Stencil([-1, 0, 1])
        assert np.allclose(st.integral_weights(2), [-1 / 12, 2 / 3, 5 / 12])

        # Integration functions to test
        x = np.linspace(0, 10, 1000)

        for fdef in self.example_funcs.values():
            f = fdef["f"](x)
            F = fdef["F"](x) - fdef["F"](x[0])

            st2 = Stencil(np.arange(0, 2))
            st3 = Stencil(np.arange(0, 3))
            st4 = Stencil(np.arange(0, 4))

            Fs2 = st2.integrate(x, f, 1)
            Fs3 = st3.integrate(x, f, 2)
            Fs4 = st4.integrate(x, f, 3)

            assert np.allclose(Fs2, F, rtol=1e-4)
            assert np.allclose(Fs3, F, rtol=1e-4)
            assert np.allclose(Fs4, F, rtol=1e-4)
