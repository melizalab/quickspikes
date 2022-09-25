# -*- coding: utf-8 -*-
# -*- mode: python -*-
import unittest

import numpy as np

from quickspikes.spikes import detector, peaks, find_run
from quickspikes.tools import filter_times, realign_spikes, spike_shape, trim_waveforms
from quickspikes.intracellular import SpikeFinder

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

b_recording = np.load("test/intra_spike.npy")
b_times = [7635, 8412, 9363, 10424, 11447, 12661, 13887, 15079, 16373,
           17753, 19168, 20682, 22357, 23979, 25574, 27209, 28989,
           30508, 32088, 33778]
b_takeoff = 24

c_recording = np.load("test/intra_spike_narrow.npy")
c_times = [8325, 8816, 9368, 9985, 10619, 11276, 11968, 12610, 13240, 13900, 14485, 15193, 15840, 16601]
c_takeoff = 14


class TestQuickspikes(unittest.TestCase):

    def test_detect_extrac_spikes(self):
        det = detector(2000, 40)
        self.assertSequenceEqual(det.send(a_recording_dbl), [t + t_peak for t in a_times])
        self.assertSequenceEqual(det.send(-a_recording_dbl), [t + t_trough for t in a_times])
        self.assertSequenceEqual(det(a_recording_dbl), [t + t_peak for t in a_times])

    def test_extract_spikes_nofilter(self):
        with self.assertRaises(ValueError):
            peaks(a_recording_dbl, [t + t_peak for t in a_times], n_before=20, n_after=300)

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
        jitter = 4
        spikes = peaks(b_recording, b_times, 200, 400)
        times, aligned = realign_spikes(b_times, spikes, 3, jitter)
        apeak = aligned.argmax(1)
        self.assertTrue((apeak == apeak[0]).all())
        # times should have shifted no more than jitter
        self.assertEqual(len(b_times), times.size)
        for t1, t2 in zip(b_times, times):
            self.assertAlmostEqual(t1, t2 // 3, delta=jitter)

    def test_detect_intrac_spikes_narrow(self):
        det = detector(-20, 100)
        self.assertSequenceEqual(det.send(c_recording), c_times)


class TestTools(unittest.TestCase):

    def test_trim(self):
        # empirical test: windows should have no more than one peak after
        # trimming
        det = detector(-20, 100)
        spikes = peaks(c_recording, c_times, 200, 700)
        for _, spike_w in trim_waveforms(spikes, c_times, 200, 100):
            t = det(spike_w.astype("d"))
            self.assertEqual(len(t), 1)
            self.assertSequenceEqual(t, [200])

    def test_run(self):
        a = np.arange(-10, 10)
        self.assertEqual(find_run(a, 0, 5), 11)

    def test_no_run(self):
        a = np.zeros(20)
        a[10:14] = 1
        self.assertEqual(find_run(a, 0, 5), None)

    def test_extrac_shape(self):
        shape = spike_shape(a_spike, 1)
        self.assertEqual(shape["peak_t"], t_peak)
        self.assertEqual(t_peak + shape["trough_t"], t_trough)

    def test_intrac_trough(self):
        spikes = peaks(b_recording, b_times, 200, 400)
        for i, spike in enumerate(spikes):
            shape = spike_shape(spike, dt=1, t_baseline=100, min_rise=13)
            peak = shape["peak_t"]
            self.assertEqual(shape["trough_t"], spike[peak:].argmin())

    def test_intrac_narrow_trough(self):
        spikes = peaks(c_recording, c_times, 200, 400)
        for spike in spikes:
            shape = spike_shape(spike, dt=1, t_baseline=100, min_rise=13)
            peak = shape["peak_t"]
            self.assertEqual(shape["trough_t"], spike[peak:].argmin())

    def test_intrac_trough_no_min(self):
        spikes = peaks(b_recording, b_times[:1], 100, 100)
        for spike in spikes:
            shape = spike_shape(spike, dt=1, t_baseline=100, min_rise=13)
            peak = shape["peak_t"]
            self.assertEqual(shape["trough_t"], spike[peak:].argmin())

    def test_intrac_onset(self):
        # this case is based on manual inspection of the spike waveform
        spike = peaks(b_recording, b_times[:1], 200, 100)[0]
        shape = spike_shape(spike, dt=1, t_baseline=100, min_rise=13)
        self.assertEqual(shape["takeoff_t"], b_takeoff)

    def test_intrac_no_onset(self):
        spike = peaks(b_recording, b_times[:1], 100, 100)[0]
        shape = spike_shape(spike, dt=1, t_baseline=80, min_rise=13)
        self.assertIsNone(shape["takeoff_t"])

    def test_intrac_narrow_onset(self):
        # this case is based on manual inspection of the spike waveform
        spike = peaks(c_recording, c_times[:1], 200, 100)[0]
        shape = spike_shape(spike, dt=1, t_baseline=100, min_rise=13)
        self.assertEqual(shape["takeoff_t"], c_takeoff)


class TestDynamicExtractor(unittest.TestCase):

    def test_intrac_broad(self):
        detector = SpikeFinder(50, 350, 5000)
        peak, thresh, takeoff, base = detector.calculate_threshold(b_recording)
        self.assertEqual(takeoff, b_takeoff)
        for i, (time, spike) in enumerate(
                detector.extract_spikes(b_recording, 10, upsample=2, jitter=4)
        ):
            self.assertAlmostEqual(time, b_times[i], delta=4)

    def test_intrac_narrow(self):
        detector = SpikeFinder(50, 350, 5000)
        peak, thresh, takeoff, base = detector.calculate_threshold(c_recording)
        self.assertEqual(takeoff, c_takeoff)
        for i, (time, spike) in enumerate(
            detector.extract_spikes(c_recording, 10, upsample=2, jitter=4)
        ):
            self.assertAlmostEqual(time, c_times[i], delta=4)
