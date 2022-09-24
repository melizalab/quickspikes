#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules=cythonize("quickspikes/spikes.pyx"))
