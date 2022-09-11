#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
from setuptools import setup, Extension

setup(ext_modules=[Extension("quickspikes.spikes", sources=["quickspikes/spikes.pyx"])])
