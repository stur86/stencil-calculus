#!/usr/bin/env python

from distutils.core import setup

setup(
    name="Stencyl Calculus",
    version="0.5",
    description="A small set of Python functions for finite differences calculus with"
    " given stencils",
    author="Simone Sturniolo",
    author_email="simonesturniolo@gmail.com",
    url="https://github.com/stur86/stencil-calculus",
    packages=["stencils"],
    install_requites=["numpy"],
)
