# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""spike detection and extraction

Copyright (C) 2013 Dan Meliza <dmeliza@gmail.com>
Created Fri Jul 12 14:05:16 2013
"""
from typing import Iterable, Union, Tuple, Optional, Iterator, Mapping
import numpy as np

numeric = Union[int, float, np.number]


def filter_times(
    times: Iterable[numeric], min: numeric, max: numeric
) -> Tuple[numeric]:
    return tuple(t for t in times if (t > min) and (t < max))


def realign_spikes(
    times: Iterable[numeric],
    spikes: np.ndarray,
    *,
    upsample: int,
    jitter: int = 3,
    reflect_fft: bool = False,
    expected_peak: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Realign spikes to their peaks using bandwidth-limited resampling

    times    : one-dimensional array of spike times, in units of samples
    spikes   : array of spike waveforms, with dimensions (nspikes, npoints)
    upsample : integer, the factor by which to upsample the data for realignment (must be greater than 1)
    jitter   : samples (in original data) to search for true peak
    expected_peak : if supplied, searches around this index (in original data) for the peak. If not supplied,
                    uses the argmax of the mean spike

    Returns (times, spikes), with the sampling rate increased by a factor of upsample

    """
    assert (
        isinstance(upsample, int) and upsample > 1
    ), "Upsampling factor must be an integer greater than 1"
    nevents, nsamples = spikes.shape

    # first infer the expected peak time
    if expected_peak is None:
        expected_peak = np.mean(spikes, 0).argmax() * upsample
    else:
        expected_peak *= upsample
    spikes = fftresample(spikes, int(nsamples * upsample), reflect=reflect_fft)
    # find peaks within upsample samples of mean peak
    shift = find_peaks(spikes, expected_peak, upsample * jitter)
    start = shift + upsample * jitter
    nshifted = (nsamples - 2 * jitter) * upsample
    shifted = np.zeros((nevents, nshifted))
    for i, spike in enumerate(spikes):
        shifted[i, :] = spike[start[i] : start[i] + nshifted]
    return (np.asarray(times) * upsample + shift, shifted)


def find_peaks(spikes: np.ndarray, peak: int, window: int) -> np.ndarray:
    """Locate the peaks in an array of spikes.

    spikes: resampled spike waveforms, dimensions (nspikes, nsamples)
    peak:   the expected peak location
    window: the number of samples to either side of the peak to look for the peak

    Returns array of shift values relative to peak

    """
    r = slice(peak - window, peak + window + 1)
    return spikes[:, r].argmax(1) - window


def fftresample(S: np.ndarray, npoints: int, *, reflect: bool = False) -> np.ndarray:
    """Resample an array of waveforms using discrete fourier transform. The
    signal is transformed in the fourier domain and then padded or truncated to
    the desired size. This should be equivalent to a sinc resampling.

    S: a 2D array of waveforms (nspikes, nsamples)
    npoints: the desired duration of the resampled signal
    reflect: if True, pad the sample with reflected copies on either end to suppress edge effects

    """
    from numpy.fft import rfft, irfft

    if reflect:
        Srev = S[:, ::-1]
        S = np.column_stack([Srev, S, Srev])
        npoints = npoints * 3
    Sf = rfft(S, axis=1)
    Srs = (1.0 * npoints / S.shape[1]) * irfft(Sf, npoints, axis=1)
    if reflect:
        npoints = npoints // 3
        return Srs[:, npoints : npoints * 2]
    else:
        return Srs


def trim_waveforms(
    spikes: Iterable[np.ndarray], times: Iterable[int], *, peak_t: int, n_rise: int
) -> Iterator[Tuple[int, np.ndarray]]:
    """Trims spike waveforms to remove overlapping peaks.

    The peaks() function is used to extract waveforms in a window surrounding
    peak times. The windows for successive peaks may overlap each other if the
    peak times are closer together than the window size, which can result in
    secondary peaks occuring later in the window corresponding to the subsequent
    peak(s). If this is not desired, this function can be used to trim the
    samples after n_rise before the next peak.

    times: the times of the spikes (in samples)
    spikes: a list of spike waveforms, or a 2D array with dimensions (nspikes, nsamples)
    peak_t: the index of the peak in the spike waveforms
    n_rise: trim up to this many samples before the subsequent peak

    Note that there is nothing to prevent a spike from being completely trimmed
    if the subsequent spike is very close and n_rise is too large. In practice
    this is highly unlikely.

    """
    diffs = np.diff(times)
    for t, dt, s in zip(times, diffs, spikes):
        i_last = min(
            peak_t + dt - n_rise,
            # s[peak_t:].argmin() + peak_t,
            s.size,
        )
        yield t, s[:i_last]
    # the last spike
    s = spikes[-1]
    # i_last = min(s[peak_t:].argmin() + peak_t, s.size)
    yield times[-1], s[: s.size]


def runlength_encode(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run length encoding of arr. Returns (runlengths, startpositions, values)"""
    # from stackoverflow 1066758
    n = len(arr)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(arr[1:] != arr[:-1])  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return (z, p, arr[i])
