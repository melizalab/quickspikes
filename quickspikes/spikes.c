/* @(#)spikes.c
 *
 * Various C functions for analyzing extracellular data.
 */

#include <string.h>
#include <math.h>
#include <stdio.h>

void
spike_times(short *out, const double *samples, int nsamples, double thresh, int window)
{
        int i,j, peak_ind;
        double peak_val;
        memset(out, 0, nsamples * sizeof(short));

        for (i = 0; i < nsamples; ++i) {
                // trigger if signal crosses threshold
                if (samples[i] > thresh) {
                        peak_val = samples[i];
                        peak_ind = i;
                        // search for max value in window
                        for (j = i; j < i + window; ++j) {
                                if (samples[j] > peak_val) {
                                        peak_val = samples[j];
                                        peak_ind = j;
                                }
                        }
                        // mark as an event iff there was a peak
                        if (peak_val > samples[j])
                                out[peak_ind] = 1;
                        /// search for first sample below threshold
                        for (i = peak_ind; i < nsamples && samples[i] > thresh; ++i) {}
                }
        }
}


void
extract_spikes(double *out, const double *samples, int nsamples, const int *times, int ntimes,
               int windowstart, int windowstop)
{
        const int window = windowstart + windowstop;
        int i, event;
        for (i = 0; i < ntimes; ++i) {
                event = times[i];
                if ((event - windowstart < 0) || (event + windowstop > nsamples)) continue;
                memcpy(out+(i*window), samples+event-windowstart, window*sizeof(double));
        }
}

void
extract_subthreshold(double *out, const double *samples, int nsamples, const int *times, int ntimes,
                     double vthresh, double dvthresh, int skip)
{
        int i, j, spikestart, spikestop;
        memcpy(out, samples, nsamples*sizeof(double));
        for (i = 0; i < ntimes; ++i) {
                if (samples[times[i]] < vthresh) {
                        fprintf(stderr, "Spike peak at %d is below threshold; skipping\n", i);
                        continue;
                }
                // iterate back to find when spike crosses threshold
                for (spikestart = times[i], j = 0; spikestart >= 0; --spikestart, ++j) {
                        if ((j > skip) &&
                            ((samples[spikestart + 1] - samples[spikestart] < dvthresh) ||
                             (samples[spikestart] < vthresh))) break;
                        out[spikestart] = NAN;
                }
                // iterate forward to find when spike crosses threshold
                for (spikestop = times[i], j = 0; spikestop < nsamples; ++spikestop, ++j) {
                        if ((j > skip) &&
                            ((samples[spikestop - 1] - samples[spikestop] < dvthresh) ||
                             (samples[spikestop] < vthresh))) break;
                        out[spikestop] = NAN;
                }

        }
}
