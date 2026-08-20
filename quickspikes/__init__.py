# -*- mode: python -*-
"""spike detection routines

Copyright (C) 2013-2024 Dan Meliza <dmeliza@gmail.com>
Created Wed Jul 24 09:26:36 2013
"""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

from .spikes import detector, peaks, subthreshold
from .tools import filter_times, realign_spikes

try:
    __version__ = _version("quickspikes")
except _PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["detector", "filter_times", "peaks", "realign_spikes", "subthreshold"]
