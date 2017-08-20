# stencil-calculus
A Python class for finite differences calculus. 

This library provides only one class, `Stencil`, which is meant to perform discrete differentiation and integration with arbitrary stencils and orders of expansion. Numerical differentiation rules are expressed as linear combinations of evaluations of a given function; for example, if `g` is tò be the 2nd derivative of `f`, a common formula is:

    g[i] = (f[i-1]-2*f[i]+f[i+1])/h**2

where `h` is the step. This function uses evaluations at the given index, one step forward, and one step backwards; in other words, a stencil `[-1, 0, 1]`. Similarly, the integration rule known as Simpson's rule:

    F[i] = F[i-1] + (f[i] + 4*f[i+1] + f[i+2])*h/6

uses a `[0, 1, 2]` stencil and is built on a 2nd order Taylor series expansion of `f`. 
The `Stencil` class allows one to generalise these formulas and compute finite difference and integration formulas of any order for any stencil. The only restriction is that the order of the approximation must be strictly lower than the length of the stencil. This class was inspired by the excellent explanation and applet on [Cameron Taylor's website](http://web.media.mit.edu/~crtaylor/calculator.html), and works off the same formula.

## Usage

The `Stencil` class is instantiated with `Stencil(s)`, where `s` is an array of integers representing the needed stencil. The class object created then has the following methods:

* `difference_weights(n, div_fac=False)`: generates array of weights for differentiation of order `n`. If `div_fac` is set to `True`, the weights will be returned for the derivative divided by `n!` (this is potentially convenient if the derivative is to be used in a Taylor series)
* `integral_weights(n)`: generates array of weights for integration, based on an expansion of order `n`
* `difference_matrix(n, l, h, div_fac=False)`: return an `lxl` matrix `A` for differentiation of order `n` of a function evaluated on `l` points with spacing `h` such that `np.dot(A, f)` will be the desired derivative
* `derive(x, y, n=1)`: return the derivative of order `n` of function `y` evaluated on points `x`
* `integral_matrix(n, l, h)`: return an `lxl` matrix `A` for integration using an expansion of order `n` of a function evaluated on `l` points with spacing `h` such that `np.dot(A, f)` will be the desired integral
* `integrate(x, y, n=1)`: return the integral using an expansion of order `n` of function `y` evaluated on points `x`

