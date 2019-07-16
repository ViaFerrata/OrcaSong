"""
Custom km3pipe modules for making nn input files.
"""

import warnings
import numpy as np
import km3pipe as kp

__author__ = 'Stefan Reck'


class McInfoMaker(kp.Module):
    """
    Get the desired mc_info from the blob.

    Attributes
    ----------
    mc_info_extr : function
        Function to extract the info. Takes the blob as input, outputs
        a dict with the desired mc_infos.
    store_as : str
        Store the mcinfo with this name in the blob.

    """
    def configure(self):
        self.mc_info_extr = self.require('mc_info_extr')
        self.store_as = self.require('store_as')

    def process(self, blob):
        track = self.mc_info_extr(blob)
        dtypes = [(key, np.float64) for key in track.keys()]
        kp_hist = kp.dataclasses.Table(track,
                                       dtype=dtypes,
                                       h5loc='y',
                                       name='event_info')

        blob[self.store_as] = kp_hist
        return blob


class TimePreproc(kp.Module):
    """
    Preprocess the time in the blob.

    Can add t0 to hit times.
    Times of hits and mchits can be centered with the time of the first
    triggered hit.

    Attributes
    ----------
    add_t0 : bool
        If true, t0 will be added.
    center_time : bool
        If true, center hit and mchit times.

    """
    def configure(self):
        self.add_t0 = self.require('add_t0')
        self.center_time = self.get('center_time', default=True)

        self.has_mchits = None
        self._t0_flag = False
        self._cent_hits_flag = False
        self._cent_mchits_flag = False

    def process(self, blob):
        if self.has_mchits is None:
            self.has_mchits = "McHits" in blob

        if self.add_t0:
            blob = self.add_t0_time(blob)
        blob = self.center_hittime(blob)

        return blob

    def add_t0_time(self, blob):
        if not self._t0_flag:
            self._t0_flag = True
            print("Adding t0 to hit times")
        hits_time = blob["Hits"].time
        hits_t0 = blob["Hits"].t0
        blob["Hits"].time = np.add(hits_time, hits_t0)

        return blob

    def center_hittime(self, blob):
        hits_time = blob["Hits"].time
        hits_triggered = blob["Hits"].triggered
        t_first_trigger = np.min(hits_time[hits_triggered == 1])

        if self.center_time:
            if not self._cent_hits_flag:
                print("Centering time of Hits")
                self._cent_hits_flag = True
            blob["Hits"].time = np.subtract(hits_time, t_first_trigger)

        if self.has_mchits:
            if not self._cent_mchits_flag:
                print("Centering time of McHits")
                self._cent_mchits_flag = True
            mchits_time = blob["McHits"].time
            blob["McHits"].time = np.subtract(mchits_time, t_first_trigger)

        return blob


class ImageMaker(kp.Module):
    """
    Make a n-d histogram from the blob.

    Attributes
    ----------
    bin_edges_list : List
        List with the names of the fields to bin, and the respective bin edges,
        including the left- and right-most bin edge.
    store_as : str
        Store the images with this name in the blob.

    """
    def configure(self):
        self.bin_edges_list = self.require('bin_edges_list')
        self.store_as = self.require('store_as')

    def process(self, blob):
        data, bins, name = [], [], ""

        for bin_name, bin_edges in self.bin_edges_list:
            data.append(blob["Hits"][bin_name])
            bins.append(bin_edges)
            name += bin_name + "_"

        histogram = np.histogramdd(data, bins=bins)[0]
        title = name + "event_images"

        hist_one_event = histogram[np.newaxis, ...].astype(np.uint8)
        kp_hist = kp.dataclasses.NDArray(hist_one_event, h5loc='x', title=title)

        blob[self.store_as] = kp_hist
        return blob


class BinningStatsMaker(kp.Module):
    """
    Generate a histogram of the number of hits for each binning field name.

    E.g. if the bin_edges_list contains "pos_z", this will make a histogram
    of #Hits vs. "pos_z", together with how many hits were outside
    of the bin edges in both directions.

    Per default, the resolution of the histogram (width of bins) will be
    higher then the given bin edges, and the edges will be stored seperatly.
    The time is the exception: The plotted bins have exactly the
    given bin edges.

    Attributes
    ----------
    bin_edges_list : List
        List with the names of the fields to bin, and the respective bin edges,
        including the left- and right-most bin edge.
    bin_plot_freq : int
        Extract data for the histograms only every given number of blobs
        (reduces time the pipeline takes to complete).
    res_increase : int
        Increase the number of bins by this much in the hists (so that one
        can see if the edges have been placed correctly). Is never used
        for the time binning (field name "time").

    """
    def configure(self):
        self.bin_edges_list = self.require('bin_edges_list')

        self.pdf_path = self.get('pdf_path', default=None)
        self.bin_plot_freq = self.get("bin_plot_freq", default=1)
        self.res_increase = self.get('res_increase', default=5)
        self.plot_bin_edges = self.get('plot_bin_edges', default=True)

        self.hists = {}
        for bin_name, org_bin_edges in self.bin_edges_list:
            if bin_name == "time":
                bin_edges = org_bin_edges
            else:
                bin_edges = self._space_bin_edges(org_bin_edges)

            self.hists[bin_name] = {
                "hist": np.zeros(len(bin_edges) - 1),
                "hist_bin_edges": bin_edges,
                "bin_edges": org_bin_edges,
                # below smallest edge, above largest edge:
                "cut_off": np.zeros(2),
            }

        self.i = 0

    def _space_bin_edges(self, bin_edges):
        """
        Increase resolution of given binning.
        """
        increased_n_bins = (len(bin_edges) - 1) * self.res_increase + 1
        bin_edges = np.linspace(bin_edges[0], bin_edges[-1],
                                increased_n_bins)

        return bin_edges

    def process(self, blob):
        """
        Extract data from blob for the hist plots.
        """
        if self.i % self.bin_plot_freq == 0:
            for bin_name, hists_data in self.hists.items():
                hist_bin_edges = hists_data["hist_bin_edges"]

                data = blob["Hits"][bin_name]
                hist = np.histogram(data, bins=hist_bin_edges)[0]

                out_pos = data[data > np.max(hist_bin_edges)].size
                out_neg = data[data < np.min(hist_bin_edges)].size

                self.hists[bin_name]["hist"] += hist
                self.hists[bin_name]["cut_off"] += np.array([out_neg, out_pos])

        self.i += 1
        return blob

    def finish(self):
        """
        Get the hists, which are the stats of the binning.

        Its a dict with each binning field name containing the following
        ndarrays:

        bin_edges : The actual bin edges.
        cut_off : How many events were cut off in positive and negative
            direction due to this binning.
        hist_bin_edges : The bin edges for the plot in finer resolution then
            the actual bin edges.
        hist : The number of hist in each bin of the hist_bin_edges.

        """
        return self.hists


class EventSkipper(kp.Module):
    """
    Skip events based on some user function.

    Attributes
    ----------
    event_skipper : func
        Function that takes the blob as an input, and returns a bool.
        If the bool is true, the blob will be skipped.

    """
    def configure(self):
        self.event_skipper = self.require('event_skipper')

    def process(self, blob):
        skip_event = self.event_skipper(blob)
        if skip_event:
            return
        else:
            return blob


class DetApplier(kp.Module):
    """
    Apply calibration to the Hits with a detx file.

    Attributes
    ----------
    det_file : str
        Path to a .detx detector geometry file.

    """
    def configure(self):
        self.det_file = self.require("det_file")
        self.assert_t0_is_added = self.get("check_t0", default=False)

        self.calib = kp.calib.Calibration(filename=self.det_file)

    def process(self, blob):
        if self.assert_t0_is_added:
            original_time = blob["Hits"].time

        blob = self.calib.process(blob, key="Hits", outkey="Hits")
        if "McHits" in blob:
            blob = self.calib.process(blob, key="McHits", outkey="McHits")

        if self.assert_t0_is_added:
            actual_time = blob["Hits"].time
            t0 = blob["Hits"].t0
            target_time = np.add(original_time, t0)
            if not np.array_equal(actual_time, target_time):
                print(actual_time)
                print(target_time)
                raise AssertionError("t0 not added!")
            else:
                print("t0 was added ok")

        return blob
