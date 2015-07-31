# -*- coding: utf-8 -*-
# -*- mode: python -*-

from nose.tools import *
import numpy as nx

# a nice surrogate spike with 20 samples before peak and 40 after
a_spike = nx.array([-1290,  -483,  -136,  -148,  -186,   637,   328,    41,    63,
                    42,   377,   872,   639,   -17,   538,   631,   530,   693,
                    743,  3456,  6345,  5868,  4543,  3087,  1691,   830,   241,
                    -350,  -567,  -996,  -877, -1771, -1659, -1968, -2013, -2290,
                    -2143, -1715, -1526, -1108,  -500,   333,    25,  -388,  -368,
                    -435,  -817,  -858,  -793, -1089,   -16,  -430,  -529,  -252,
                    -3,  -786,   -47,  -266,  -963,  -365], dtype=nx.int16)
t_peak = a_spike.argmax()
t_trough = a_spike.argmin()
a_recording = nx.zeros(10000, dtype='d')
times = [100, 400, 1200]
for t in times:
    a_recording[t:t + a_spike.size] += a_spike

def test_spike_times():
    from quickspikes.spikes import spike_times

    expected = [t + t_peak for t in times]
    assert_sequence_equal(spike_times(a_recording, 500, dt=0.1), expected)
    assert_sequence_equal(spike_times(a_recording, 4000, dt=0.1), expected)

def test_detect_spikes():
    from quickspikes.spikez import detect_spikes

    detector = detect_spikes(2000, 40)
    assert_sequence_equal(detector.send(a_recording), [t + t_peak for t in times])
    assert_sequence_equal(detector.send(-a_recording), [t + t_trough for t in times])

    # detector = detect_spikes(-4000, 40)
    # assert_sequence_equal(detector.send(a_recording), [])
    # assert_sequence_equal(detector.send(-a_recording), [t + t_peak for t in times])
