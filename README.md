
## quickspikes

This is a very basic but very fast window discriminator for detecting and
extracting spikes in a time series. It was developed for analyzing extracellular
neural recordings, but also works with intracellular data and probably many
other kinds of time series.

Here's how it works:

![detection diagram](algorithm.png)

The algorithm iterates through the time series. When the signal crosses the threshold (1) going away from zero, the algorithm then looks for a peak (2) that occurs within some number of samples from the threshold crossing. The number of samples can be adjusted to filter out broad spikes that are likely to be artifacts. If a peak occurs, its sample index is added to an array. These times can be used as-is, or they can be used to extract samples to either side of the peak for further analysis (e.g. spike sorting).

The algorithm is designed to be efficient for events on the order of 10-20 samples in duration. Many signals of interest will require highpass filtering to remove slow variations.

### Use

The algorithm is implemented in portable ANSI C code, which you can include in your own project, and there is an `f2py` wrapper for use in python. You can get a python package from PyPI:

    pip install quickspikes

Or to build from a copy of the repository:

    setup.py install



### License

Free for use under the terms of the GNU General Public License. See [[COPYING]]
for details.
