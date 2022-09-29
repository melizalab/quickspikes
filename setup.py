#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
from setuptools import setup, Extension
import numpy

setup(
    ext_modules=[Extension("quickspikes.spikes", sources=["quickspikes/spikes.pyx"])],
    include_dirs=[numpy.get_include()],
)
