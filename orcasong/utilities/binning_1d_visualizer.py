#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import km3pipe as kp
import matplotlib.pyplot as plt


class FieldPlotter:
    """
    For investigating the ideal binning, based on the info in calibrated
    .h5 files.

    Intended for 1d binning, like time or pos_z.
    Workflow:
    1. Initialize with files, then run .plot() to extract and store
       the data, and show the plot interactively.
    2. Choose a binning via .set_binning.
    3. Run .plot() again to show the plot with the adjusted binning on the
       stored data.
    4. Repeat step 2 and 3 unitl happy with binning.
    (5.) Save plot via .plot(savepath)

    The plot will have some bins attached in both directions for
    better overview.

    Attributes:
    -----------
    files : list or str
        The .h5 file(s).
    field : str
        The field to look stuff up, e.g. "time", "pos_z", ...
    only_mc : bool
        If true, will look up "McHits" in the blob. Otherwise "Hits".
    center_events : int
        For centering events with their median.
        0 : No centering.
        1 : Center with median of triggered hits.
        2 : Center with median of all hits.
    data : ndarray
        The extracted data.
    n_events : int
        The number of events in the extracted data.
    limits : List
        Left- and right-most edge of the binning.
    n_bins : int
        The number of bins.
    plot_padding : List
        Fraction of bins to append to left and right direction
        (only in the plot).
    x_label : str
    y_label : str
    hist_kwargs : dict
        Kwargs for plt.hist
    xlim : List
        The xlimits of the hist plot.
    show_plots : bool
        If True, auto plt.show() the plot.

    """
    def __init__(self, files, field, only_mc=False, center_events=0):
        self.files = files
        self.field = field
        self.only_mc = only_mc
        self.center_events = center_events

        self.data = None
        self.n_events = None

        self.limits = None
        self.n_bins = 100
        self.plot_padding = [0.2, 0.2]

        # Plotting stuff
        self.xlabel = None
        self.ylabel = 'Fraction of hits'
        self.hist_kwargs = {
            "histtype": "stepfilled",
            "density": True,
        }
        self.xlim = None
        self.show_plots = True

    def plot(self, save_path=None):
        """
        Generate and store or load the data, then make the plot.

        Parameters
        ----------
        save_path : str, optional
            Save plot to here.

        Returns
        -------
        fig : pyplot figure
            The plot.

        """
        if self.data is None:
            self.data = self.get_events_data()
        fig = self.make_histogram(save_path)
        return fig

    def set_binning(self, limits, n_bins):
        """
        Set the desired binning.

        Parameters
        ----------
        limits : List
            Left- and right-most edge of the binning.
        n_bins : int
            The number of bins.

        """
        self.limits = limits
        self.n_bins = n_bins

    def get_binning(self):
        """
        Get the stored binning.

        Returns
        -------
        limits : List
            Left- and right-most edge of the binning.
        n_bins : int
            The number of bins.

        """
        return self.limits, self.n_bins

    def get_events_data(self):
        """
        Get the content of a field from all events in the file(s).

        Returns:
        --------
        data : ndarray
            The desired data.

        """
        data_all_events = None
        self.n_events = 0

        if not isinstance(self.files, list):
            files = [self.files]
        else:
            files = self.files

        for fname in files:
            print("File " + fname)
            event_pump = kp.io.hdf5.HDF5Pump(filename=fname)

            for i, event_blob in enumerate(event_pump):
                self.n_events += 1

                if i % 2000 == 1:
                    print("Blob no. "+str(i))

                data_one_event = self._get_hits(event_blob)

                if data_all_events is None:
                    data_all_events = data_one_event
                else:
                    data_all_events = np.concatenate(
                        [data_all_events, data_one_event], axis=0)

        print("Number of events: " + str(self.n_events))
        return data_all_events

    def make_histogram(self, save_path=None):
        """
        Plot the hist data. Can also save it if given a save path.

        Parameters
        ----------
        save_path : str, optional
            Save the fig to this path.

        Returns
        -------
        fig : pyplot figure
            The plot.

        """
        if self.data is None:
            raise ValueError("Can not make histogram, no data extracted yet.")

        bin_edges = self._get_bin_edges()

        fig, ax = plt.subplots()
        n, bins, patches = plt.hist(self.data, bins=bin_edges, **self.hist_kwargs)
        print("Size of first bin: " + str(bins[1] - bins[0]))

        plt.grid(True, zorder=0, linestyle='dotted')

        if self.limits is not None:
            for bin_line_x in self.limits:
                plt.axvline(x=bin_line_x, color='firebrick', linestyle='--')

        if self.xlabel is None:
            plt.xlabel(self._get_xlabel())

        if self.xlim is not None:
            plt.xlim(self.xlim)

        plt.ylabel(self.ylabel)
        plt.tight_layout()

        if save_path is not None:
            print("Saving plot to "+str(save_path))
            plt.savefig(save_path)

        if self.show_plots:
            plt.show()

        return fig

    def _get_bin_edges(self):
        """
        Get the padded bin edges.

        """
        limits, n_bins = self.get_binning()

        if limits is None:
            bin_edges = n_bins

        else:
            total_range = limits[1] - limits[0]
            bin_size = total_range / n_bins

            addtnl_bins = [
                int(self.plot_padding[0] * n_bins),
                int(self.plot_padding[1] * n_bins)
            ]

            padded_range = [
                limits[0] - bin_size * addtnl_bins[0],
                limits[1] + bin_size * addtnl_bins[1]
            ]
            padded_n_bins = n_bins + addtnl_bins[0] + addtnl_bins[1]
            bin_edges = np.linspace(padded_range[0], padded_range[1],
                                    padded_n_bins + 1)

        return bin_edges

    def _get_hits(self, event_blob):
        """
        Get desired attribute from a event blob.

        Parameters
        ----------
        event_blob
            The km3pipe event blob.

        Returns
        -------
        blob_data : ndarray
            The desired data.

        """
        if self.only_mc:
            field_name = "McHits"
        else:
            field_name = "Hits"

        blob_data = event_blob[field_name][self.field]

        if self.center_events == 1:
            triggered = event_blob[field_name].triggered
            median_trigger = np.median(blob_data[triggered == 1])
            blob_data = np.subtract(blob_data, median_trigger)

        elif self.center_events == 2:
            median = np.median(blob_data)
            blob_data = np.subtract(blob_data, median)

        return blob_data

    def _get_xlabel(self):
        """
        Some saved xlabels.

        """
        if self.field == "time":
            xlabel = "Time [ns]"
        elif self.field == "pos_z":
            xlabel = "Z position [m]"
        else:
            xlabel = None
        return xlabel


class TimePlotter(FieldPlotter):
    """
    For plotting the time.
    """
    def __init__(self, files, only_mc=False):
        field = "time"

        if only_mc:
            center_events = 2
        else:
            center_events = 1

        FieldPlotter.__init__(self, files,
                              field,
                              only_mc=only_mc,
                              center_events=center_events)


class ZPlotter(FieldPlotter):
    """
    For plotting the z dim.
    """
    def __init__(self, files, only_mc=False):
        field = "pos_z"
        center_events = 0
        FieldPlotter.__init__(self, files,
                              field,
                              only_mc=only_mc,
                              center_events=center_events)

        self.plotting_bins = 100

    def _get_bin_edges(self):
        """
        Get the padded bin edges.

        """
        return self.plotting_bins

    def set_binning(self, limits, n_bins):
        """
        Set the desired binning.

        Parameters
        ----------
        limits : List
            Left- and right-most edge of the binning.
        n_bins : int
            The number of bins.

        """
        bin_edges = np.linspace(limits[0], limits[1],
                                n_bins + 1)
        self.limits = bin_edges
        self.n_bins = n_bins
