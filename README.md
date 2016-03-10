
## quickspikes

This is a very basic but very fast window discriminator for detecting and
extracting spikes in a time series. It was developed for analyzing extracellular
neural recordings, but also works with intracellular data and probably many
other kinds of time series.

Here's how it works:

![detection diagram](algorithm.png)

The algorithm iterates through the time series. When the signal crosses the threshold (1) going away from zero, the algorithm then looks for a peak (2) that occurs within some number of samples from the threshold crossing. The number of samples can be adjusted to filter out broad spikes that are likely to be artifacts. If a peak occurs, its sample index is added to an array. These times can be used as-is, or they can be used to extract samples to either side of the peak for further analysis (e.g. spike sorting).

The algorithm uses a streaming pattern, so it's suitable for realtime operations. Many signals of interest will require highpass filtering to remove slow variations.

### Installation and Use

The algorithm is written in cython. You can get a python package from PyPI:

    pip install quickspikes

Or to build from a copy of the repository:

    setup.py install

To detect peaks, you instantiate the detector with parameters that match the events you want to detect, and then send the detector chunks of data. For example, an extracellular recording at 20 kHz stored in 16-bit integers may have a noise floor around 2000, and the spikes will be on the order of 20 samples wide:

```python
import quickspikes as qs
det = qs.detector(1000, 30)
times = det.send(samples)
```

You can adjust the detector's threshold to compensate for shifts in the mean and standard deviation of the signal:

```python
reldet = qs.detector(2.5, 30)
reldet.scale_thresh(samples.mean(), samples.std())
times = reldet.send(samples)
```

To detect negative-going events, you'll need to invert the signal.

There is also a reference copy of an ANSI C implementation and an `f2py` wrapper in `f2py/`. This algorithm is slightly less efficient and flexible, but may give better results if included directly in a C codebase.

### License

Free for use under the terms of the GNU General Public License. See [[COPYING]]
for details.

[![Build Status](https://travis-ci.org/melizalab/quickspikes.png?branch=master)](https://travis-ci.org/melizalab/quickspikes)
