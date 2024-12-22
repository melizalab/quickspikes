# -*- mode: python -*-
"""spike detection routines

Copyright (C) 2013-2024 Dan Meliza <dmeliza@gmail.com>
Created Wed Jul 24 09:26:36 2013
"""
from .spikes import detector, peaks, subthreshold
from .tools import filter_times, realign_spikes

try:
    from importlib.metadata import version
    __version__ = version("quickspikes")
except ImportError:
    # For Python < 3.8
    from importlib_metadata import version
    __version__ = version("quickspikes")
except Exception:
    # If package is not installed (e.g. during development)
    __version__ = "unknown"
__all__ = ["detector", "peaks", "subthreshold", "realign_spikes", "filter_times"]
