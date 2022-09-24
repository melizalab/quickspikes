# -*- coding: utf-8 -*-
# -*- mode: cython -*-
# cython: language_level=3
"""spike detection routines

Copyright (C) 2013 Dan Meliza <dmeliza@gmail.com>
Created Wed Jul 24 09:26:36 2013
"""
from cython cimport view, boundscheck
import numpy as np

ctypedef fused sample_t:
    short
    int
    long
    float
    double

cdef extern from "math.h":
    double NAN

cdef enum DetectorState:
    BELOW_THRESHOLD = 1
    BEFORE_PEAK = 2
    AFTER_PEAK = 3

cdef inline bint compare_sign(sample_t x, sample_t y):
    """return True iff ((x > 0) && (y > 0)) || ((x < 0) && (y < 0))"""
    return ((x >= 0) and (y >= 0)) or ((x < 0) and (y < 0))

cdef class detector:
    """Detect spikes in a continuous stream of samples.

    This implementation allows samples to be sent to the detector in blocks, and
    maintains state across successive calls.
    """
    cdef:
        double thresh
        double scaled_thresh
        double prev_val
        size_t n_after
        size_t n_after_crossing
        DetectorState state

    def __init__(self, double thresh, size_t n_after):
        """Construct spike detector.

        Parameters
        ----------
        thresh : double
          The crossing threshold that triggers the detector. NB: to detect
          negative-going peaks, invert the signal sent to send()
        n_after : size_t
          The maximum number of samples after threshold crossing to look for the
          peak. If a peak has not been located within this window, the crossing
          is considered an artifact and is not counted.

        """
        self.thresh = self.scaled_thresh = thresh
        self.n_after = n_after
        self.reset()

    def scale_thresh(self, double mean, double sd):
        """Adjust threshold for the mean and standard deviation of the signal.

        The effective threshold will be (thresh * sd + mean)

        """
        self.scaled_thresh = self.thresh * sd + mean

    def send(self, double[:] samples):
        """Detect spikes in a time series.

        Returns a list of indices corresponding to the peaks in the data.
        Retains state between calls. Call reset() if there is a gap in the
        signal.

        """
        cdef double x
        cdef size_t i = 0
        out = []

        for i in range(samples.shape[0]):
            x = samples[i]
            if self.state is BELOW_THRESHOLD:
                if x >= self.scaled_thresh:
                    self.prev_val = x
                    self.n_after_crossing = 0
                    self.state = BEFORE_PEAK
            elif self.state is BEFORE_PEAK:
                if self.prev_val > x:
                    out.append(i - 1)
                    self.state = AFTER_PEAK
                elif self.n_after_crossing > self.n_after:
                    self.state = BELOW_THRESHOLD
                else:
                    self.prev_val = x
                    self.n_after_crossing += 1
            elif self.state is AFTER_PEAK:
                if x < self.scaled_thresh:
                    self.state = BELOW_THRESHOLD
        return out

    def __call__(self, double[:] samples):
        """Detect spikes in a time series.

        Returns a list of indices corresponding to the peaks in the data.
        Resets state between calls.

        """
        self.reset()
        return self.send(samples)

    def reset(self):
        """Reset the detector's internal state"""
        self.state = BELOW_THRESHOLD


@boundscheck(False)
def peaks(sample_t[:] samples, times, size_t n_before=75, size_t n_after=400):
    """Extracts samples around times

    Returns a 2D array with len(times) rows and (n_before + n_after) columns
    containing the values surrounding the sample indices in times.

    Note: all values of times must be greater than n_before and less than
    samples.size - n_after. See `tools.filter_spikes`

    """
    cdef size_t i = 0
    cdef size_t event, start, stop
    if sample_t is short:
        dtype = np.int16
    elif sample_t is int:
        dtype = np.int32
    elif sample_t is long:
        dtype = np.int64
    elif sample_t is float:
        dtype = np.float
    elif sample_t is double:
        dtype = np.double
    out = np.empty(shape=(len(times), n_before + n_after), dtype=dtype)
    cdef sample_t [:, :] arr = out
    for i,event in enumerate(times):
        start = event - n_before
        stop = event + n_after
        if start < 0:
            raise ValueError("spike %d waveform starts before input data" % i)
        elif stop >= samples.size:
            raise ValueError("spike %d waveform ends after input data" % i)
        else:
            arr[i,:] = samples[event-n_before:event+n_after]

    return out


def find_run(sample_t[:] values, sample_t thresh, size_t min_run):
    """ Return the index of the first element in values that starts a run of at
    least min_run in length, or None if no such run exists. """
    cdef size_t i
    cdef size_t run_start = -1
    cdef bint in_run = False
    for i in range(values.size):
        if values[i] > thresh:
            if not in_run:
                in_run = True
                run_start = i
            elif i - run_start == min_run:
                return run_start
        else:
            in_run = False
    return None


def subthreshold(double[:] samples, times,
                 double v_thresh=-50, double dv_thresh=0, size_t min_size=10):
    """Removes spikes from time series

    Spikes are removed from the voltage trace by beginning at each peak and
    moving in either direction until V drops below thresh_v OR dv drops below
    thresh_dv. This algorithm does not work with negative peaks, so invert your
    signal if you need to.

    - samples : signal to analyze
    - times : times of the peaks (in samples)
    - thresh_v : do not include points contiguous with the peak with V > thresh_v
    - thresh_dv : do not include points contiguous with the peak with deltaV > thresh_dv.
                  negative values correspond to positive-going voltages
    - min_size : always remove at least this many points on either side of peak

    Returns a copy of samples with the data points around spike times set to NaN

    """
    cdef size_t i, j, spikestart, spikestop
    cdef size_t nsamples = samples.size
    cdef double[:] out = samples.copy()
    for i in times:
        if samples[i] < v_thresh:
            #print "spike peak at %d is below threshold; skipping"
            continue
        # iterate back and blank out samples before v or dv cross threshold
        j = 0
        spikestart = i
        while spikestart >= 0:
            if (j > min_size and
                (samples[spikestart + 1] - samples[spikestart] < dv_thresh) or
                (samples[spikestart] < v_thresh)):
                break
            out[spikestart] = NAN
            j += 1
            spikestart -= 1
        # iterate back and blank out samples before v or dv cross threshold
        j = 0
        spikestop = i
        while spikestop < nsamples:
            if (j > min_size and
                (samples[spikestop - 1] - samples[spikestop] < dv_thresh) or
                (samples[spikestop] < v_thresh)):
                break
            out[spikestop] = NAN
            j += 1
            spikestop += 1
    return out


# Variables:
# End:
