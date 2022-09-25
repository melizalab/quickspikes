# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""specialized functions for intracellular data"""
from typing import Tuple, Optional, Iterator
import logging
import numpy as np

log = logging.getLogger("quickspikes")


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
        thresh_rel: float = 0.35,
        thresh_min: float = -50,
        deriv_thresh: float = 10.0,
    ) -> Optional[Tuple[float, float, float]]:
        """Calculate the detection threshold from the amplitude of the first spike in V.

        If no spike can be detected in the signal, returns None. Otherwise, the
        instance's threshold is set at `thresh_rel` times the amplitude of the
        first spike, or `thresh_min` (whichever is greater), and (peak_index,
        threshold voltage, takeoff_index [relative to spike peak], takeoff voltage) is returned.

        """
        from quickspikes.tools import spike_shape

        self.spike_thresh = self.first_spike_amplitude = None
        first_peak_idx = V.argmax()
        first_spike = V[first_peak_idx - self.n_before : first_peak_idx + self.n_after]
        shape = spike_shape(
            first_spike,
            dt=1,
            deriv_thresh=deriv_thresh,
            t_baseline=self.n_before // 2,
            min_rise=self.n_rise // 4,
        )
        if shape["takeoff_t"] is None:
            return
        spike_base = shape["takeoff"]
        first_spike_amplitude = shape["peak"] - spike_base
        self.spike_thresh = max(
            spike_base + thresh_rel * first_spike_amplitude,
            thresh_min,
        )
        return (first_peak_idx, self.spike_thresh, shape["takeoff_t"], shape["takeoff"])

    def extract_spikes(
        self, V: np.ndarray, min_amplitude: float, upsample: int = 2, jitter: int = 4
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """Detect and extract spikes from V.

        V: input signal
        min_amplitude: only yield spikes that are at least this much above threshold
        upsample: the upsampling factor to use in realigning spikes
        jitter: the expected jitter in peak position

        Yields (t, spike)
        """
        from quickspikes import detector, peaks
        from quickspikes.tools import trim_waveforms, filter_times, realign_spikes

        detector = detector(self.spike_thresh, self.n_rise)
        spike_times = filter_times(
            detector.send(V), self.n_before, V.size - self.n_after
        )
        spikes = peaks(V, spike_times, self.n_before, self.n_after)
        spike_times, spikes = realign_spikes(
            spike_times,
            spikes,
            upsample,
            reflect_fft=True,
            expected_peak=self.n_before,
        )
        # original version of this located the peak after resampling, but I
        # don't think this is generally necessary
        for time, spike in trim_waveforms(
            spikes, spike_times, self.n_before, self.n_rise * 2
        ):
            if not np.any(spike - self.spike_thresh > min_amplitude):
                continue
            yield (time // upsample, spike)
