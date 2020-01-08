"""
Custom km3pipe modules for making nn input files.
"""

import numpy as np
import km3pipe as kp

__author__ = 'Stefan Reck'


class McInfoMaker(kp.Module):
    """
    Store mc info as float64 in the blob.

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
        kp_hist = kp.dataclasses.Table(
            track, dtype=dtypes,  h5loc='y', name='event_info')

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
            self.cprint("Adding t0 to hit times")
        blob["Hits"].time = np.add(blob["Hits"].time, blob["Hits"].t0)

        if self.has_mchits:
            blob["McHits"].time = np.add(blob["McHits"].time,
                                         blob["McHits"].t0)

        return blob

    def center_hittime(self, blob):
        hits_time = blob["Hits"].time
        hits_triggered = blob["Hits"].triggered
        t_first_trigger = np.min(hits_time[hits_triggered == 1])

        if self.center_time:
            if not self._cent_hits_flag:
                self.cprint("Centering time of Hits")
                self._cent_hits_flag = True
            blob["Hits"].time = np.subtract(hits_time, t_first_trigger)

        if self.has_mchits:
            if not self._cent_mchits_flag:
                self.cprint("Centering time of McHits")
                self._cent_mchits_flag = True
            mchits_time = blob["McHits"].time
            blob["McHits"].time = np.subtract(mchits_time, t_first_trigger)

        return blob


class ImageMaker(kp.Module):
    """
    Make a n-d histogram from "Hits" in blob, and store it.

    Attributes
    ----------
    bin_edges_list : List
        List with the names of the fields to bin, and the respective bin edges,
        including the left- and right-most bin edge.
    store_as : str
        Store the images with this name in the blob.
    hit_weights : str, optional
        Use blob["Hits"][hit_weights] as weights for samples in histogram.

    """

    def configure(self):
        self.bin_edges_list = self.require('bin_edges_list')
        self.store_as = self.require('store_as')
        self.hit_weights = self.get('hit_weights')

    def process(self, blob):
        data, bins, name = [], [], ""

        for bin_name, bin_edges in self.bin_edges_list:
            data.append(blob["Hits"][bin_name])
            bins.append(bin_edges)
            name += bin_name + "_"

        if self.hit_weights is not None:
            weights = blob["Hits"][self.hit_weights]
        else:
            weights = None

        histogram = np.histogramdd(data, bins=bins, weights=weights)[0]
        title = name + "event_images"

        hist_one_event = histogram[np.newaxis, ...].astype(np.uint8)
        kp_hist = kp.dataclasses.NDArray(
            hist_one_event, h5loc='x', title=title)

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
        self.bin_plot_freq = self.get("bin_plot_freq", default=1)
        self.res_increase = self.get('res_increase', default=5)

        self.hists = {}
        for bin_name, org_bin_edges in self.bin_edges_list:
            # dont space bin edges for time
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
        bin_edges = np.linspace(
            bin_edges[0], bin_edges[-1], increased_n_bins)

        return bin_edges

    def process(self, blob):
        """
        Extract data from blob for the hist plots.
        """
        if self.i % self.bin_plot_freq == 0:
            for bin_name, hists_data in self.hists.items():
                hist_bin_edges = hists_data["hist_bin_edges"]

                hits = blob["Hits"]
                data = hits[bin_name]
                # get how much is cut off due to these limits
                out_pos = data[data > np.max(hist_bin_edges)].size
                out_neg = data[data < np.min(hist_bin_edges)].size

                # get all hits which are not cut off by other bin edges
                data = hits[bin_name][self._is_in_limits(
                    hits, excluded=bin_name)]
                hist = np.histogram(data, bins=hist_bin_edges)[0]

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

    def _is_in_limits(self, hits, excluded=None):
        """ Get which hits are in the limits defined by ALL bin edges
        (except for given one). """
        inside = None
        for dfield, edges in self.bin_edges_list:
            if dfield == excluded:
                continue
            is_in = np.logical_and(hits[dfield] >= min(edges),
                                   hits[dfield] <= max(edges))
            if inside is None:
                inside = is_in
            else:
                inside = np.logical_and(inside, is_in)
        return inside


class EventSkipper(kp.Module):
    """
    Skip events based on blob content.

    Attributes
    ----------
    event_skipper : callable
        Function that takes the blob as an input, and returns a bool.
        If the bool is true, the blob will be skipped.

    """

    def configure(self):
        self.event_skipper = self.require('event_skipper')
        self._not_skipped = 0
        self._skipped = 0

    def process(self, blob):
        if self.event_skipper(blob):
            self._skipped += 1
            return
        else:
            self._not_skipped += 1
            return blob

    def finish(self):
        tot_events = self._skipped + self._not_skipped
        self.cprint(
            f"Skipped {self._skipped}/{tot_events} events "
            f"({self._skipped/tot_events:.4%})."
        )


class DetApplier(kp.Module):
    """
    Apply calibration to the Hits and McHits with a detx file.

    Attributes
    ----------
    det_file : str
        Path to a .detx detector geometry file.

    """

    def configure(self):
        self.det_file = self.require("det_file")

        self.calib = kp.calib.Calibration(filename=self.det_file)
        self._calib_checked = False

    def process(self, blob):
        if self._calib_checked is False:
            if "pos_x" in blob["Hits"]:
                self.log.warn(
                    "Warning: Using a det file, but pos_x in Hits detected. "
                    "Is the file already calibrated? This might lead to "
                    "errors with t0."
                )
            self._calib_checked = True

        blob = self.calib.process(blob, key="Hits", outkey="Hits")
        if "McHits" in blob:
            blob = self.calib.process(blob, key="McHits", outkey="McHits")

        return blob


class HitRotator(kp.Module):
    """
    Rotates hits by angle theta.

    Attributes
    ----------
    theta : float
        Angle by which hits are rotated (radian).

    """

    def configure(self):
        self.theta = self.require('theta')

    def process(self, blob):
        x = blob['Hits']['x']
        y = blob['Hits']['y']

        rot_matrix = np.array([[np.cos(self.theta), - np.sin(self.theta)],
                               [np.sin(self.theta), np.cos(self.theta)]])

        x_rot = []
        y_rot = []

        for i in range(0, len(x)):
            vec = np.array([[x[i]], [y[i]]])
            rot = np.dot(rot_matrix, vec)
            x_rot.append(rot[0][0])
            y_rot.append(rot[1][0])

        blob['Hits']['x'] = x_rot
        blob['Hits']['y'] = y_rot

        return blob

