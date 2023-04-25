# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""specialized functions for intracellular data"""
from typing import Tuple, Optional, Iterator, Mapping
from collections import namedtuple
import logging
import numpy as np

log = logging.getLogger("quickspikes")

Spike = namedtuple(
    "Spike",
    [
        "peak_t",
        "peak_V",
        "takeoff_t",
        "takeoff_V",
        "trough_t",
        "trough_V",
        "half_rise_t",
        "half_decay_t",
        "max_dV_t",
        "max_dV",
        "min_dV_t",
        "min_dV",
    ],
)


def spike_shape(
    spike: np.ndarray,
    dt: float,
    *,
    deriv_thresh: float = 10.0,
    t_baseline: float = 2.0,
    min_rise: float = 0.25,
) -> Optional[Spike]:
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
    can be found, then half_rise and half_decay are undefined. If the spike is
    very badly formed (e.g. the peak is at the very beginning or end), then None
    is returned.

    """
    from quickspikes.spikes import find_run
    from quickspikes.tools import runlength_encode

    n_baseline = int(t_baseline / dt)
    min_rise = int(min_rise / dt)
    peak_ind = spike.argmax()
    if peak_ind in (0, spike.size - 1):
        return None
    peak_v = spike[peak_ind]
    deriv = np.gradient(spike, dt)
    # trough
    dV0_ind = find_run(deriv[peak_ind:], thresh=0, min_run=min_rise) or spike.size
    trough_ind = spike[peak_ind : peak_ind + dV0_ind].argmin()
    # min and max dV/dt
    dVmax_ind = deriv[:peak_ind].argmax()
    dVmin_ind = deriv[peak_ind:].argmin()

    # takeoff
    mdV = deriv[:n_baseline].mean()
    sdV = deriv[:n_baseline].std()
    thresh = mdV + deriv_thresh * sdV
    n, p, v = runlength_encode(deriv[:peak_ind] > thresh)
    pp = p[(n >= min_rise) & v]
    try:
        takeoff_ind = pp[-1]
    except IndexError:
        takeoff_t = takeoff_v = half_rise_t = half_decay_t = None
    else:
        takeoff_v = spike[takeoff_ind]
        takeoff_t = (peak_ind - takeoff_ind) * dt
        half_ampl = (peak_v + takeoff_v) / 2
        half_rise_ind = find_run(spike[:peak_ind], thresh=half_ampl, min_run=1) or 0
        half_rise_t = (peak_ind - half_rise_ind) * dt
        half_decay_ind = (
            find_run(-spike[peak_ind:], thresh=-half_ampl, min_run=2)
            or spike.size - peak_ind
        )
        half_decay_t = half_decay_ind * dt
    return Spike(
        peak_t=peak_ind * dt,
        peak_V=peak_v,
        takeoff_t=takeoff_t,
        takeoff_V=takeoff_v,
        trough_t=trough_ind * dt,
        trough_V=spike[peak_ind + trough_ind],
        half_rise_t=half_rise_t,
        half_decay_t=half_decay_t,
        max_dV=deriv[dVmax_ind],
        max_dV_t=(peak_ind - dVmax_ind) * dt,
        min_dV=deriv[dVmin_ind + peak_ind],
        min_dV_t=dVmin_ind * dt,
    )


class SpikeFinder:
    """Dynamic spike finder for intracellular voltage data.

    Intracellular recordings present some special challenges for detecting
    spikes. The data are not centered, and spike waveforms can change over the
    course of stimulation. The approach here is to use the first spike in the
    recording, which is typically the largest and has the best-defined onset, to
    set the threshhold for the window discriminator.

    `n_rise`: the approximate rise time of the spikes to be detected, in samples
    `n_before`: the number of samples before the spike peaks to analyze/extract
    `n_after`: the number of samples after the spike peaks to analyze/extract
    """

    def __init__(self, n_rise: int, n_before: int, n_after: int):
        self.n_rise = n_rise
        self.n_before = n_before
        self.n_after = n_after

    def calculate_threshold(
        self,
        V: np.ndarray,
        *,
        thresh_rel: float = 0.35,
        thresh_min: float = -50,
        deriv_thresh: float = 10.0,
    ) -> Optional[Spike]:
        """Calculate the detection threshold from the amplitude of the first spike in V.

        If no spike can be detected in the signal, returns None. Otherwise, the
        instance's threshold is set at `thresh_rel` times the amplitude of the
        first spike, or `thresh_min` (whichever is greater), and the spike shape
        is returned.

        """
        self.spike_thresh = self.first_spike_amplitude = None
        first_peak_idx = V[self.n_before : -self.n_after].argmax() + self.n_before
        first_spike = V[first_peak_idx - self.n_before : first_peak_idx + self.n_after]
        shape = spike_shape(
            first_spike,
            dt=1,
            deriv_thresh=deriv_thresh,
            t_baseline=self.n_before // 2,
            min_rise=self.n_rise // 4,
        )
        if shape is None or shape.takeoff_t is None:
            return
        spike_base = shape.takeoff_V
        first_spike_amplitude = shape.peak_V - spike_base
        self.spike_thresh = max(
            spike_base + thresh_rel * first_spike_amplitude,
            thresh_min,
        )
        return shape

    def detect_spikes(self, V: np.ndarray) -> Iterator[int]:
        """Using the calculated threshold, detect spikes that are extractable"""
        from quickspikes import detector, filter_times

        det = detector(self.spike_thresh, self.n_rise)
        return filter_times(det.send(V), self.n_before, V.size - self.n_after)

    def extract_spikes(
        self, V: np.ndarray, *, min_amplitude: float, upsample: int = 2, jitter: int = 4
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Detect and extract spikes from V.

        V: input signal
        min_amplitude: only yield spikes that are at least this much above threshold
        upsample: the upsampling factor to use in realigning spikes
        jitter: the expected jitter in peak position

        Yields (t, spike)
        """
        from quickspikes import peaks
        from quickspikes.tools import trim_waveforms, realign_spikes

        spike_times = self.detect_spikes(V)
        if len(spike_times) == 0:
            return
        spikes = peaks(V, spike_times, n_before=self.n_before, n_after=self.n_after)
        spike_times, spikes = realign_spikes(
            spike_times,
            spikes,
            upsample=upsample,
            reflect_fft=True,
            expected_peak=self.n_before,
        )
        # original version of this located the peak after resampling, but I
        # don't think this is generally necessary
        for time, spike in trim_waveforms(
            spikes, spike_times, peak_t=self.n_before, n_rise=self.n_rise * 2
        ):
            if not np.any(spike - self.spike_thresh > min_amplitude):
                continue
            yield (time // upsample, spike)
