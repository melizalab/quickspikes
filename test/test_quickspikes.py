# -*- coding: utf-8 -*-
# -*- mode: python -*-
import unittest
from distutils import version

import sys
import numpy as np

if sys.hexversion < 0x03000000:
    FileNotFoundError = IOError

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
a_recording = np.zeros(10000, dtype='d')
a_times = [100, 400, 1200, 1500, 5000, 5200, 6123, 9730]
for t in a_times:
    a_recording[t:t + a_spike.size] += a_spike

try:
    b_recording = np.load("intra_spike.npy").astype('d')
except FileNotFoundError:
    b_recording = np.load("test/intra_spike.npy").astype('d')
b_times = [7635, 8412, 9363, 10424, 11447, 12661, 13887, 15079, 16373,
           17753, 19168, 20682, 22357, 23979, 25574, 27209, 28989,
           30508, 32088, 33778]

class TestQuickspikes(unittest.TestCase):

    def test_detect_extrac_spikes(self):
        from quickspikes.spikes import detector
        det = detector(2000, 40)

        self.assertSequenceEqual(det.send(a_recording), [t + t_peak for t in a_times])
        self.assertSequenceEqual(det.send(-a_recording), [t + t_trough for t in a_times])
        self.assertSequenceEqual(det(a_recording), [t + t_peak for t in a_times])


    def test_extract_spikes_nofilter(self):
        from quickspikes.spikes import peaks
        with self.assertRaises(ValueError):
            x = peaks(a_recording, [t + t_peak for t in a_times], n_before=20, n_after=300)

    def test_extract_spikes(self):
        from quickspikes import peaks, filter_times

        n_before = 20
        n_after = 300
        times = filter_times([t + t_peak for t in a_times], n_before, a_recording.size - n_after)
        x = peaks(a_recording, times, n_before=n_before, n_after=n_after)
        # last peak should get dropped
        self.assertEqual(x.shape[0], len(a_times) - 1)
        self.assertEqual(x.shape[1], 320)

        self.assertTrue(np.all(a_spike == x[0,:a_spike.size]))

    def test_detect_intrac_spikes(self):
        from quickspikes.spikes import detector

        det = detector(0, 100)
        self.assertSequenceEqual(det.send(b_recording), b_times)

        det = detector(-20, 100)
        self.assertSequenceEqual(det.send(b_recording), b_times)

    def test_align_spikes(self):
        from quickspikes.spikes import peaks
        from quickspikes.tools import realign_spikes

        spikes = peaks(b_recording, b_times, 200, 400)
        times, aligned = realign_spikes(b_times, spikes, 3, 4)
        peaks = aligned.argmax(1)
        self.assertTrue((peaks == peaks[0]).all())
