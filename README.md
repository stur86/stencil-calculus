# stencil-calculus
A Python class for finite differences calculus. 

This library provides only one class, `Stencil`, which is meant to perform discrete differentiation and integration with arbitrary stencils and orders of expansion. Numerical differentiation rules are expressed as linear combinations of evaluations of a given function; for example, if `g` is tò be the 2nd derivative of `f`, a common formula is:

    g[i] = (f[i-1]-2*f[i]+f[i+1])/h**2

where `h` is the step. This function uses evaluations at the given index, one step forward, and one step backwards; in other words, a stencil `[-1, 0, 1]`. Similarly, the integration rule known as Simpson's rule:

    F[i] = F[i-1] + (f[i] + 4*f[i+1] + f[i+2])*h/6

uses a `[0, 1, 2]` stencil and is built on a 2nd order Taylor series expansion of `f`. 
The `Stencil` class allows one to generalise these formulas and compute finite difference and integration formulas of any order for any stencil. This was inspired by the excellent explanation and applet on [Cameron Taylor's website](http://web.media.mit.edu/~crtaylor/calculator.html), and works off the same formula.

