# -*- coding: utf-8 -*-
# -*- mode: python -*-
import unittest

import numpy as np

from quickspikes.spikes import detector, peaks, find_run
from quickspikes.tools import filter_times, realign_spikes, find_trough

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
a_recording_dbl = np.zeros(10000, dtype='d')
a_recording_int = np.zeros(10000, dtype=a_spike.dtype)
a_times = [100, 400, 1200, 1500, 5000, 5200, 6123, 9730]
for t in a_times:
    a_recording_dbl[t:t + a_spike.size] += a_spike
    a_recording_int[t:t + a_spike.size] += a_spike

try:
    b_recording = np.load("intra_spike.npy").astype('d')
except FileNotFoundError:
    b_recording = np.load("test/intra_spike.npy").astype('d')
b_times = [7635, 8412, 9363, 10424, 11447, 12661, 13887, 15079, 16373,
           17753, 19168, 20682, 22357, 23979, 25574, 27209, 28989,
           30508, 32088, 33778]


class TestQuickspikes(unittest.TestCase):

    def test_detect_extrac_spikes(self):
        det = detector(2000, 40)
        self.assertSequenceEqual(det.send(a_recording_dbl), [t + t_peak for t in a_times])
        self.assertSequenceEqual(det.send(-a_recording_dbl), [t + t_trough for t in a_times])
        self.assertSequenceEqual(det(a_recording_dbl), [t + t_peak for t in a_times])

    def test_extract_spikes_nofilter(self):
        with self.assertRaises(ValueError):
            x = peaks(a_recording_dbl, [t + t_peak for t in a_times], n_before=20, n_after=300)

    def test_extract_spikes_double(self):
        n_before = 20
        n_after = 300
        times = filter_times([t + t_peak for t in a_times], n_before, a_recording_dbl.size - n_after)
        x = peaks(a_recording_dbl, times, n_before=n_before, n_after=n_after)
        # last peak should get dropped
        self.assertEqual(x.shape[0], len(a_times) - 1)
        self.assertEqual(x.shape[1], 320)
        self.assertTrue(np.all(a_spike == x[0,:a_spike.size]))

    def test_extract_spikes_integer(self):
        n_before = 20
        n_after = 300
        times = filter_times([t + t_peak for t in a_times], n_before, a_recording_int.size - n_after)
        x = peaks(a_recording_int, times, n_before=n_before, n_after=n_after)
        # last peak should get dropped
        self.assertEqual(x.shape[0], len(a_times) - 1)
        self.assertEqual(x.shape[1], 320)
        self.assertTrue(np.all(a_spike == x[0,:a_spike.size]))

    def test_detect_intrac_spikes(self):
        det = detector(0, 100)
        self.assertSequenceEqual(det.send(b_recording), b_times)
        det = detector(-20, 100)
        self.assertSequenceEqual(det.send(b_recording), b_times)

    def test_align_spikes(self):
        spikes = peaks(b_recording, b_times, 200, 400)
        times, aligned = realign_spikes(b_times, spikes, 3, 4)
        apeak = aligned.argmax(1)
        self.assertTrue((apeak == apeak[0]).all())


class TestTools(unittest.TestCase):

    def test_run(self):
        a = np.arange(-10, 10)
        self.assertEqual(find_run(a, 0, 5), 11)

    def test_no_run(self):
        a = np.zeros(20)
        a[10:14] = 1
        self.assertEqual(find_run(a, 0, 5), None)

    def test_extrac_trough(self):
        trough = find_trough(a_spike[t_peak:])
        self.assertEqual(t_peak + trough, t_trough)

    def test_intrac_trough(self):
        spikes = peaks(b_recording, b_times, 200, 400)
        for spike in spikes:
            trough = find_trough(spike[200:])
            self.assertEqual(trough, spike[200:].argmin())

    def test_intrac_trough_no_min(self):
        spikes = peaks(b_recording, b_times[:1], 100, 100)
        for spike in spikes:
            trough = find_trough(spike[100:])
            self.assertEqual(trough, spike[100:].argmin())
