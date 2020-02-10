#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import km3pipe as kp
import matplotlib.pyplot as plt

import orcasong.modules as modules

__author__ = 'Stefan Reck'


class TimePlotter:
    """
    For plotting the time distribution of hits, in a histogram,
    and finding a good binning.

    The plot will have some bins attached in both directions for
    better overview.

    Attributes:
    -----------
    files : list
        The .h5 files to read hits from.
    do_mc_hits : bool
        Read mchits instead of hits.
    det_file : str, optional
        Path to a .detx detector geometry file, which can be used to
        calibrate the hits.
    center_time : bool
        Subtract time of first triggered hit from all hit times. Will
        also be done for McHits if they are in the blob [default: True].
    add_t0 : bool
        If true, add t0 to the time of hits. If using a det_file,
        this will already have been done automatically [default: False].
    inactive_du : int, optional
        Dont plot hits from this du.

    """
    def __init__(self, files,
                 do_mchits=False,
                 det_file=None,
                 center_time=True,
                 add_t0=False,
                 subtract_t0_mchits=False,
                 inactive_du=None):
        if isinstance(files, str):
            self.files = [files]
        else:
            self.files = files
        self.do_mchits = do_mchits
        self.det_file = det_file
        self.add_t0 = add_t0
        self.subtract_t0_mchits = subtract_t0_mchits
        self.center_time = center_time
        self.inactive_du = inactive_du

        self.data = np.array([])
        for file in self.files:
            self._read(file, self.det_file)

    def hist(self, bins=50, padding=0.5, **kwargs):
        """
        Plot the hits as a histogram.

        Parameters
        ----------
        bins : int or np.array
            Number of bins, or bin edges. If bin edges are given, some
            bins are attached left and right (next to red vertical lines)
            for better overview.
        padding : float
            Fraction of total number of bins to attach left and right
            in the plot.

        """
        plt.grid(True, zorder=-10)
        plt.xlabel("time")
        if self.do_mchits:
            plt.ylabel("mchits / bin")
        else:
            plt.ylabel("hits / bin")
        if isinstance(bins, int):
            plt.hist(self.data, bins=bins, zorder=10, **kwargs)
        else:
            self.print_binstats(bins)
            plt.hist(
                self.data,
                bins=_get_padded_bins(bins, padding),
                zorder=10,
                **kwargs
            )
            for bin_line_x in (bins[0], bins[-1]):
                plt.axvline(
                    x=bin_line_x, color='firebrick', linestyle='--', zorder=20
                )

    def print_binstats(self, bin_edges):
        print(f"Cutoff left: {np.mean(self.data < bin_edges[0]):.2%}")
        print(f"Cutoff right: {np.mean(self.data > bin_edges[-1]):.2%}")
        print(f"Avg. time per bin: {np.mean(np.diff(bin_edges)):.2f}")

    def _read(self, file, det_file=None):
        if det_file is not None:
            cal = modules.DetApplier(det_file=det_file)
        else:
            cal = None
        if self.center_time or self.add_t0:
            time_pp = modules.TimePreproc(
                add_t0=self.add_t0,
                center_time=self.center_time,
                subtract_t0_mchits=self.subtract_t0_mchits,
            )
        else:
            time_pp = None

        pump = kp.io.hdf5.HDF5Pump(filename=file)
        for i, blob in enumerate(pump):
            if i % 1000 == 0:
                print("Blob {}".format(i))
            if cal is not None:
                blob = cal.process(blob)
            if time_pp is not None:
                blob = time_pp.process(blob)
            self.data = np.concatenate(
                [self.data, self._get_data_one_event(blob)])
        pump.close()

    def _get_data_one_event(self, blob):
        if self.do_mchits:
            fld = "McHits"
        else:
            fld = "Hits"
        blob_data = blob[fld]["time"]

        if self.inactive_du is not None:
            dus = blob[fld]["du"]
            blob_data = blob_data[dus != self.inactive_du]
        return blob_data


def _get_padded_bins(bin_edges, padding):
    """ Add fraction of total # of bins, with same width. """
    bin_width = bin_edges[1] - bin_edges[0]
    n_bins = len(bin_edges) - 1
    bins_to_add = np.ceil(padding * n_bins)
    return np.linspace(
        bin_edges[0] - bins_to_add * bin_width,
        bin_edges[-1] + bins_to_add * bin_width,
        int(n_bins + 2 * bins_to_add + 1),
    )
