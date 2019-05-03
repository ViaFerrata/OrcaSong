"""
Custom km3pipe modules for making nn input files.
"""

import km3pipe as kp
import numpy as np

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

    t0 will be added to the time for real data, but not simulations.
    Time hits and mchits will be shifted by the time of the first
    triggered hit.

    """
    def process(self, blob):
        correct_mchits = "McHits" in blob
        blob = time_preproc(blob,
                            correct_hits=True,
                            correct_mchits=correct_mchits)
        return blob


def time_preproc(blob, correct_hits=True, correct_mchits=True):
    """
    Preprocess the time in the blob.

    t0 will be added to the time for real data, but not simulations.
    Time hits and mchits will be shifted by the time of the first
    triggered hit.

    """
    hits_time = blob["Hits"].time

    if "McHits" not in blob:
        # add t0 only for real data, not sims
        hits_t0 = blob["Hits"].t0
        hits_time = np.add(hits_time, hits_t0)

    hits_triggered = blob["Hits"].triggered
    t_first_trigger = np.min(hits_time[hits_triggered == 1])

    if correct_hits:
        blob["Hits"].time = np.subtract(hits_time, t_first_trigger)

    if correct_mchits:
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
