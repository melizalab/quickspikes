# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""spike detection and extraction

Copyright (C) 2013 Dan Meliza <dmeliza@gmail.com>
Created Fri Jul 12 14:05:16 2013
"""
from __future__ import division
from __future__ import print_function


def filter_times(times, min, max):
    return tuple(t for t in times if (t > min) and (t < max))


def realign_spikes(times, spikes, upsample, jitter=1, reflect_fft=False, expected_peak=None):
    """Realign spikes to their peaks using bandwidth-limited resampling

    times    : one-dimensional array of spike times, in units of samples
    spikes   : array of spike waveforms, with dimensions (nspikes, npoints)
    upsample : integer, the factor by which to upsample the data for realignment
    jitter   : samples (in original data) to search for true peak
    expected_peak : if supplied, searches around this index for the peak. If not supplied,
                    uses the argmax of the mean spike

    Returns (times, spikes), with the sampling rate increased by a factor of upsample

    """
    from numpy import mean, zeros, asarray
    upsample = int(upsample)
    assert upsample > 1, "Upsampling factor must be greater than 1"
    nevents, nsamples = spikes.shape

    # first infer the expected peak time
    if expected_peak is None:
        expected_peak = mean(spikes, 0).argmax() * upsample
    spikes = fftresample(spikes, int(nsamples * upsample), reflect=reflect_fft)
    # find peaks within upsample samples of mean peak
    shift = find_peaks(spikes, expected_peak, upsample * jitter)
    start = shift + upsample * jitter
    nshifted = (nsamples - 2 * jitter) * upsample
    shifted = zeros((nevents, nshifted))
    for i, spike in enumerate(spikes):
        shifted[i, :] = spike[start[i]:start[i]+nshifted]
    return (asarray(times) * upsample + start, shifted)


def find_peaks(spikes, peak, window):
    """Locate the peaks in an array of spikes.

    spikes: resampled spike waveforms, dimensions (nspikes, nsamples)
    peak:   the expected peak location
    window: the number of samples to either side of the peak to look for the peak

    Returns array of shift values relative to peak

    """
    r = slice(peak - window, peak + window + 1)
    return spikes[:, r].argmax(1) - window


def fftresample(S, npoints, axis=1, reflect=False):
    """Resample a signal using discrete fourier transform. The signal
    is transformed in the fourier domain and then padded or truncated
    to the correct sampling frequency.  This should be equivalent to
    a sinc resampling.

    Set reflect to True to pad the sample with reflected copies on either end.
    Do this if the shape of the spike really matters.

    """
    from numpy import column_stack
    from numpy.fft import rfft, irfft
    if reflect:
        Srev = S[:, ::-1]
        S = column_stack([Srev, S, Srev])
        npoints = npoints * 3
    Sf = rfft(S, axis=axis)
    Srs = (1. * npoints / S.shape[axis]) * irfft(Sf, npoints, axis=axis)
    if reflect:
        npoints = npoints // 3
        return Srs[:, npoints:npoints*2]
    else:
        return Srs
