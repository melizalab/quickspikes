#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -*- mode: python -*-
from setuptools import setup, find_packages, Extension
import sys
if sys.version_info[:2] < (2, 7) or (3, 0) <= sys.version_info[:2] < (3, 2):
    raise RuntimeError("Python version 2.7 or >= 3.2 required.")

try:
    from Cython.Distutils import build_ext
    SUFFIX = '.pyx'
except ImportError:
    from distutils.command.build_ext import build_ext
    SUFFIX = '.c'

_spikes = Extension('quickspikes.spikes', sources=['quickspikes/spikes' + SUFFIX])


# ---- Metadata ---- #
VERSION = '1.3.5'

cls_txt = """
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License (GPL)
Programming Language :: Python
Topic :: Scientific/Engineering
Operating System :: Unix
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Natural Language :: English
"""

long_desc = """
This is a very basic but very fast window discriminator for detecting and
extracting spikes in a time series. It was developed for analyzing extracellular
neural recordings, but also works with intracellular data and probably many
other kinds of time series.
"""

#####

setup(
    name='quickspikes',
    version=VERSION,
    packages=find_packages(exclude=["*test*"]),
    ext_modules=[_spikes],
    cmdclass={'build_ext': build_ext},

    scripts=[],

    description="detect and extract spikes in time series data",
    long_description=long_desc,
    classifiers=[x for x in cls_txt.split("\n") if x],
    setup_requires=["numpy>=1.10"],

    author="Dan Meliza",
    maintainer="Dan Meliza",
    url='http://github.com/melizalab/quickspikes',
    test_suite='nose.collector',
)
