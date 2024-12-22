#!/usr/bin/env python
# -*- mode: python -*-
import numpy
from setuptools import Extension, setup

setup(
    ext_modules=[Extension("quickspikes.spikes", sources=["quickspikes/spikes.pyx"])],
    include_dirs=[numpy.get_include()],
)
