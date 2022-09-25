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
    expected_peak : if supplied, searches around this index for the peak. If not supplied,
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


def fftresample(S: np.ndarray, npoints: int, reflect: bool = False) -> np.ndarray:
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
    spikes: Iterable[np.ndarray], times: Iterable[int], peak_t: int, n_rise: int
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


def spike_shape(
    spike: np.ndarray,
    dt: float,
    deriv_thresh: float = 10.0,
    t_baseline: float = 2.0,
    min_rise: float = 0.25,
) -> Mapping[str, float]:
    """Computes spike shape features:

    takeoff: the voltage/time when the derivative of the waveform exceeds
            `deriv_thresh` standard deviations over a baseline period (
            `t_baseline`) for at least `min_rise` ms.

    trough: The minimum value between the spike peak and the first sample where
            the derivative becomes greater than zero for at least `min_rise`
            ms. If the derivative never crosses this threshold, then it
            this is the global minimum after the peak.

    half_rise: the half-way point between the takeoff and the peak on the rising phase
    half_decay: the halfway point between the peak and the takeoff on the decay phase
    max_deriv: the maximum dV/dt on the rising phase
    min_deriv: the minimum dV/dt on the falling phase

    The default values work well for an intracellular spike recorded at 50 kHz
    with a clearly defined onset and a width 1-2 ms. If the spikes are narrower,
    min_rise should be reduced. If the spikes are too close together, it may be
    difficult to establish a good baseline for calculating the threshold.

    All features are returned as both a time and a value. If no takeoff point
    can be found, then half_rise and half_decay are undefined.

    """
    from quickspikes.spikes import find_run

    n_baseline = int(t_baseline / dt)
    min_rise = int(min_rise / dt)
    peak_ind = spike.argmax()
    peak_v = spike[peak_ind]
    deriv = np.gradient(spike, dt)
    # trough
    dV0_ind = find_run(deriv[peak_ind:], 0, min_rise) or spike.size
    trough_ind = spike[peak_ind : peak_ind + dV0_ind].argmin()
    # min and max dV/dt
    dVmax_ind = deriv[:peak_ind].argmax()
    dVmin_ind = deriv[peak_ind:].argmin()

    # takeoff
    mdV = deriv[:n_baseline].mean()
    sdV = deriv[:n_baseline].std()
    thresh = mdV + deriv_thresh * sdV
    n, p, v = _rle(deriv[:peak_ind] > thresh)
    pp = p[(n >= min_rise) & v]
    try:
        takeoff_ind = pp[-1]
    except IndexError:
        takeoff_t = takeoff_v = half_rise_t = half_decay_t = None
    else:
        takeoff_v = spike[takeoff_ind]
        takeoff_t = (peak_ind - takeoff_ind) * dt
        half_ampl = (peak_v + takeoff_v) / 2
        half_rise_ind = find_run(spike[:peak_ind], half_ampl, 1) or 0
        half_rise_t = (peak_ind - half_rise_ind) * dt
        half_decay_ind = (
            find_run(-spike[peak_ind:], -half_ampl, 2) or spike.size - peak_ind
        )
        half_decay_t = (half_decay_ind - peak_ind) * dt
    return dict(
        peak_t=peak_ind * dt,
        peak=peak_v,
        takeoff_t=takeoff_t,
        takeoff=takeoff_v,
        trough_t=trough_ind * dt,
        trough=spike[peak_ind + trough_ind],
        half_rise_t=half_rise_t,
        half_decay_t=half_decay_t,
        max_dV=deriv[dVmax_ind],
        max_dV_t=(peak_ind - dVmax_ind) * dt,
        min_dV=deriv[dVmin_ind + peak_ind],
        min_dV_t=dVmin_ind * dt,
    )


def _rle(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
