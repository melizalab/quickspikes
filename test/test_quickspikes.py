# -*- coding: utf-8 -*-
# -*- mode: python -*-
import pytest

import numpy as np

from quickspikes.spikes import detector, peaks, find_run
from quickspikes.tools import filter_times, realign_spikes, trim_waveforms
from quickspikes.intracellular import SpikeFinder, spike_shape

# a nice surrogate spike with 20 samples before peak and 40 after
a_spike = np.array([-1290,  -483,  -136,  -148,  -186,   637,   328,    41,    63,
                    42,   377,   872,   639,   -17,   538,   631,   530,   693,
                    743,  3456,  6345,  5868,  4543,  3087,  1691,   830,   241,
                    -350,  -567,  -996,  -877, -1771, -1659, -1968, -2013, -2290,
                    -2143, -1715, -1526, -1108,  -500,   333,    25,  -388,  -368,
                    -435,  -817,  -858,  -793, -1089,   -16,  -430,  -529,  -252,
                    -3,  -786,   -47,  -266,  -963,  -365], dtype=np.int16)
t_peak = a_spike.argmax()
t_trough = a_spike.argmin()


@pytest.fixture
def extrac_times():
    return  [100, 400, 1200, 1500, 5000, 5200, 6123, 9730]


@pytest.fixture(params=[np.double, np.int32])
def extrac_recording(extrac_times, request):
    a_recording = np.zeros(10000, dtype=request.param)
    for t in extrac_times:
        a_recording[t:t + a_spike.size] += a_spike
    return a_recording
    

@pytest.fixture(params=["wide", "narrow"])
def intrac_recording(request):
    # a bit klunky
    if request.param == "wide":
        recording = np.load("test/intra_spike.npy")
        times = [7635, 8412, 9363, 10424, 11447, 12661, 13887, 15079, 16373,
           17753, 19168, 20682, 22357, 23979, 25574, 27209, 28989,
           30508, 32088, 33778]
        takeoff = 24
    elif request.param == "narrow":
        recording = np.load("test/intra_spike_narrow.npy")
        times = [8325, 8816, 9368, 9985, 10619, 11276, 11968, 12610, 13240, 13900, 14485, 15193, 15840, 16601]
        takeoff = 14
    return recording, times, takeoff


def test_detect_extrac_spikes(extrac_recording, extrac_times):
    det = detector(2000, 40)
    assert det.send(extrac_recording) == [t + t_peak for t in extrac_times]
    assert det.send(-extrac_recording) == [t + t_trough for t in extrac_times]


def test_extract_spikes_nofilter(extrac_recording, extrac_times):
    with pytest.raises(ValueError):
        peaks(extrac_recording, [t + t_peak for t in extrac_times], n_before=20, n_after=300)


def test_extract_spikes(extrac_recording, extrac_times):
    n_before = 20
    n_after = 300
    times = filter_times([t + t_peak for t in extrac_times], n_before, extrac_recording.size - n_after)
    x = peaks(extrac_recording, times, n_before=n_before, n_after=n_after)
    # last peak should get dropped
    assert x.shape[0] == len(extrac_times) - 1
    assert x.shape[1] == n_before + n_after
    assert np.all(a_spike == x[0,:a_spike.size])


def test_detect_intrac_spikes(intrac_recording):
    recording, times, *_ = intrac_recording
    det = detector(-20, 100)
    assert det.send(recording) == times


def test_align_spikes(intrac_recording):
    recording, times, *_ = intrac_recording
    jitter = 4
    spikes = peaks(recording, times, 200, 400)
    aln_times, aligned = realign_spikes(times, spikes, 3, jitter)
    apeak = aligned.argmax(1)
    assert all(apeak == apeak[0])
    # times should have shifted no more than jitter
    assert len(times) == aln_times.size
    for t1, t2 in zip(times, aln_times):
        assert abs(t1 - t2 // 3) <= jitter


def test_trim(intrac_recording):
    """spikes can be trimmed to eliminate overlapping peaks """
    recording, times, *_ = intrac_recording
    det = detector(-20, 100)
    spikes = peaks(recording, times, 200, 700)
    for _, spike_w in trim_waveforms(spikes, times, 200, 100):
        t = det(spike_w.astype("d"))
        assert t == [200]


def test_extrac_shape():
    shape = spike_shape(a_spike, 1)
    assert shape.peak_t == t_peak
    assert t_peak + shape.trough_t == t_trough


def test_intrac_trough(intrac_recording):
    """spike_shape finds trough under normal conditions"""
    recording, times, *_ = intrac_recording
    spikes = peaks(recording, times, 200, 400)
    for i, spike in enumerate(spikes):
        shape = spike_shape(spike, dt=1, t_baseline=100, min_rise=13)
        peak = shape.peak_t
        assert shape.trough_t == spike[peak:].argmin()


def test_intrac_trough_no_min(intrac_recording):
    """spike_shape finds trough when the spike is clipped"""
    recording, times, *_ = intrac_recording
    spikes = peaks(recording, times[:1], 100, 100)
    for spike in spikes:
        shape = spike_shape(spike, dt=1, t_baseline=100, min_rise=13)
        peak = shape.peak_t
        assert shape.trough_t == spike[peak:].argmin()


def test_intrac_onset(intrac_recording):
    """correctly detect onset time """
    # this case is based on manual inspection of the spike waveform
    recording, times, takeoff = intrac_recording
    spike = peaks(recording, times[:1], 200, 100)[0]
    shape = spike_shape(spike, dt=1, t_baseline=100, min_rise=13)
    assert shape.takeoff_t == takeoff


def test_intrac_no_onset(intrac_recording):
    recording, times, takeoff = intrac_recording
    spike = peaks(recording, times[:1], 20, 100)[0]
    shape = spike_shape(spike, dt=1, t_baseline=80, min_rise=13)
    assert shape.takeoff_t is None


def test_bad_spike_peak_at_edge():
    """ spike_shape returns None if max is at the edge of the waveform """
    bad = np.arange(100, -100, -1)
    assert spike_shape(bad, dt=1, t_baseline=100, min_rise=13) is None
    assert spike_shape(bad[::-1], dt=1, t_baseline=100, min_rise=13) is None


def test_spike_finder_intrac(intrac_recording):
    recording, times, takeoff = intrac_recording
    detector = SpikeFinder(50, 350, 5000)
    shape = detector.calculate_threshold(recording)
    assert shape.takeoff_t == takeoff
    for i, (time, spike) in enumerate(
            detector.extract_spikes(recording, 10, upsample=2, jitter=4)
    ):
        assert time == pytest.approx(times[i], abs=4)


def test_no_spikes():
    """ extract_spikes should yield nothing if there are no spikes detected """
    # In some rare cases, there will be a single spike that passes the first spike
    # threshold but is not detected due to the minimum height requirement
    # (thresh_min).
    detector = SpikeFinder(50, 350, 5000)
    signal = np.random.randn(100000)
    detector.spike_thresh = 50
    times = [t for t, s in detector.extract_spikes(signal, 40)]
    assert len(times) == 0


def test_thresh_for_max_at_edge_of_signal(intrac_recording):
    """ calculate_threshold should ignore spikes that are too close to the edge of the signal """
    recording, times, takeoff = intrac_recording
    detector = SpikeFinder(50, 350, 5000)
    clip = times[0] - 300
    _ = detector.calculate_threshold(recording[clip:])
    peaks = detector.detect_spikes(recording[clip:])
    assert peaks[0] + clip == times[1]


def test_run():
    a = np.arange(-10, 10)
    assert find_run(a, 0, 5) == 11


def test_no_run():
    a = np.zeros(20)
    a[10:14] = 1
    assert find_run(a, 0, 5) is None


