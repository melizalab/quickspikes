# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Functions to extract spikes from an intracellular recording

Copyright (C) 2012 Dan Meliza <dmeliza@gmail.com>
Created 2012-03-21
"""

import numpy as nx


def spike_times(x, spike_thresh=5, dt=0.02, peak_window=1.5, **kwargs):
    """ Detect spikes in a signal

    Returns an array of times corresponding to the peaks in the signal that were
    above spike_thresh.

    - x : signal to analyze
    - spike_thresh: the threshold for the detector
    - dt : time step of signal
    - peak_window: how far to search forward (in dt units) for the peak
    """
    from quickspikes._spikes import spike_times as st
    t = st(x, spike_thresh, int(peak_window/dt))
    return t.nonzero()[0]

def spike_waveforms(x, tt, dt=0.02, peak_window=1.5, spike_onset=3.5, spike_offset=8.,
                    min_isi=8., **kwargs):
    """ Extract spike waveforms

    Returns a 2D array containing the waveforms of the spikes in x with peaks at tt

    - x : signal to analyze
    - tt : times of the peaks (in samples)
    - dt : time step of signal
    - spike_onset : amount of time (in dt units) before peak to include
    - spike_offset : amount of time (in dt units) after peak to include
    - min_isi : exclude spikes with isi < min_isi
    """
    from quickspikes._spikes import extract_spikes
    # drop spikes that won't give a complete waveform for this step
    tta = tt[(tt > int(spike_onset/dt)) & ((tt + int(spike_offset/dt)) < x.size)]
    # drop spikes with isi < min_isi
    ind = ((nx.diff(tta) * dt) > min_isi).nonzero()[0]
    tta = tta[ind+1]
    if tta.size > 0:
        return extract_spikes(x, tta, int(spike_onset/dt), int(spike_offset/dt)).T
    else:
        return None

def extract_subthreshold(x, tt, dt=0.02, thresh_v=-50, thresh_dv=0, min_size=10, **kwargs):
    """ Extract subthreshold activity

    Spikes are removed from the voltage trace by beginning at each peak and
    moving in either direction until V drops below thresh_v OR dv drops below
    thresh_dv.

    - x : signal to analyze
    - tt : times of the peaks (in samples)
    - dt : time step of signal
    - thresh_v : do not include points contiguous with the peak with V > thresh_v
    - thresh_dv : do not include points contiguous with the peak with dV/dt > thresh_dv.
                  negative values corrspond to positive-going voltages
    - min_size : always remove at least this many points on either side of peak
    """
    from quickspikes._spikes import extract_subthreshold as es
    return es(x, tt, thresh_v, thresh_dv * dt, min_size)


def spike_lag(spk, lag=3):
    """
    Given a 2D array of spikes (T by N), return V(t), V(t-lag)
    """
    return spk[:-lag,:], spk[lag:,:]

def spike_dt(spk, lag=3):
    """
    Given a 2D array of spikes (T by N), calculate discrete difference
    for a phase space plot.  Resamples as needed.

    @return V(t), V(t)-V(t-lag)
    """
    Vt,Vtt = spike_lag(spk, lag)
    return Vt,Vtt-Vt

def fftresample(S, npoints, axis=1, padding='flip'):
    """
    Resample a signal using discrete fourier transform. The signal
    is transformed in the fourier domain and then padded or truncated
    to the correct sampling frequency.  This should be equivalent to
    a sinc resampling.
    """
    from numpy.fft import rfft, irfft
    if padding == 'flip':
        S = nx.concatenate([S[::-1], S, S[::-1]],0)
        npoints *=3
    Sf = rfft(S, axis=axis)
    S = (1. * npoints / S.shape[axis]) * irfft(Sf, npoints, axis=axis)
    if padding == 'flip':
        return S[npoints/3:npoints*2/3]
    else:
        return S


# Variables:
# End:
