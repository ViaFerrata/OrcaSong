"""
Custom km3pipe modules for making nn input files.
"""

import numpy as np
import km3pipe as kp
import km3modules as km
import orcasong.plotting.plot_binstats as plot_binstats

__author__ = 'Stefan Reck'


class McInfoMaker(kp.Module):
    """
    Stores info as float64 in the blob.

    Attributes
    ----------
    extractor : function
        Function to extract the info. Takes the blob as input, outputs
        a dict with the desired mc_infos.
    store_as : str
        Store the mcinfo with this name in the blob.

    """

    def configure(self):
        self.extractor = self.require('extractor')
        self.store_as = self.require('store_as')
        self.to_float64 = self.get("to_float64", default=True)
        self.sort_y = self.get("sort_y", default=True)

    def process(self, blob):
        track = self.extractor(blob)
        if self.sort_y:
            track = {k: track[k] for k in sorted(track)}
        if self.to_float64:
            dtypes = []
            for key, v in track.items():
                if key in ("group_id", "event_id"):
                    dtypes.append((key, type(v)))
                else:
                    dtypes.append((key, np.float64))
        else:
            dtypes = None
        kp_hist = kp.dataclasses.Table(
            track, dtype=dtypes,  h5loc='y', name='event_info')
        if len(kp_hist) != 1:
            self.log.warning(
                "Warning: Extracted mc_info should have len 1, "
                "but it has len {}".format(len(kp_hist))
            )
        blob[self.store_as] = kp_hist
        return blob


class TimePreproc(kp.Module):
    """
    Preprocess the time in the blob in various ways.

    Attributes
    ----------
    add_t0 : bool
        If true, t0 will be added to times of hits.
    center_time : bool
        If true, center hit and mchit times with the time of the first
        triggered hit.

    """

    def configure(self):
        self.add_t0 = self.get('add_t0', default=False)
        self.center_time = self.get('center_time', default=True)

        self._print_flags = set()

    def process(self, blob):
        if self.add_t0:
            blob = self.add_t0_time(blob)
        if self.center_time:
            blob = self.center_hittime(blob)
        return blob

    def add_t0_time(self, blob):
        self._print_once("Adding t0 to hit times")
        blob["Hits"].time = np.add(blob["Hits"].time, blob["Hits"].t0)
        return blob

    def center_hittime(self, blob):
        hits_time = blob["Hits"].time
        hits_triggered = blob["Hits"].triggered
        t_first_trigger = np.min(hits_time[hits_triggered != 0])

        self._print_once("Centering time of Hits with first triggered hit")
        blob["Hits"].time = np.subtract(hits_time, t_first_trigger)

        if "McHits" in blob:
            self._print_once("Centering time of McHits with first triggered hit")
            mchits_time = blob["McHits"].time
            blob["McHits"].time = np.subtract(mchits_time, t_first_trigger)

        return blob

    def _print_once(self, text):
        if text not in self._print_flags:
            self._print_flags.add(text)
            self.cprint(text)


class ImageMaker(kp.Module):
    """
    Make a n-d histogram from "Hits", and store it in the blob as 'samples'.

    Attributes
    ----------
    bin_edges_list : List
        List with the names of the fields to bin, and the respective bin edges,
        including the left- and right-most bin edge.
    hit_weights : str, optional
        Use blob["Hits"][hit_weights] as weights for samples in histogram.

    """

    def configure(self):
        self.bin_edges_list = self.require('bin_edges_list')
        self.hit_weights = self.get('hit_weights')
        self.store_as = "samples"

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

        hist_one_event = histogram[np.newaxis, ...].astype(np.uint8)
        kp_hist = kp.dataclasses.NDArray(
            hist_one_event, h5loc='x', title=name + "event_images")

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
    res_increase : int
        Increase the number of bins by this much in the hists (so that one
        can see if the edges have been placed correctly). Is never used
        for the time binning (field name "time").
    bin_plot_freq : int
        Extract data for the histograms only every given number of blobs
        (reduces time the pipeline takes to complete).

    """

    def configure(self):
        self.bin_edges_list = self.require('bin_edges_list')
        self.res_increase = self.get("res_increase", default=5)
        self.bin_plot_freq = 1

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
        Append the hists, which are the stats of the binning.

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


class PointMaker(kp.Module):
    """
    Store individual hit info from "Hits" in the blob as 'samples'.

    Used for graph networks.

    Attributes
    ----------
    hit_infos : tuple, optional
        Which entries in the '/Hits' Table will be kept. E.g. pos_x, time, ...
        Default: Keep all entries.
    time_window : tuple, optional
        Two ints (start, end). Hits outside of this time window will be cut
        away (based on 'Hits/time'). Default: Keep all hits.
    only_triggered_hits : bool
        If true, use only triggered hits. Otherwise, use all hits (default).
    max_n_hits : int
        Maximum number of hits that gets saved per event. If an event has
        more, some will get cut randomly! Default: Keep all hits.
    fixed_length : bool
        If False (default), save hits of events with variable length as
        2d arrays using km3pipe's indices.
        If True, pad hits of each event with 0s to a fixed length,
        so that they can be stored as 3d arrays like images.
        max_n_hits needs to be given in that case, and a column will be
        added called 'is_valid', which is 0 if the entry is padded,
        and 1 otherwise.
        This is inefficient and will cut off hits, so it should not be used.
    dset_n_hits : str, optional
        If given, store the number of hits that are in the time window
        as a new column called 'n_hits_intime' in the dataset with
        this name (usually this is EventInfo).

    """
    def configure(self):
        self.hit_infos = self.get("hit_infos", default=None)
        self.time_window = self.get("time_window", default=None)
        self.only_triggered_hits = self.get("only_triggered_hits", default=False)
        self.max_n_hits = self.get("max_n_hits", default=None)
        self.fixed_length = self.get("fixed_length", default=False)
        self.dset_n_hits = self.get("dset_n_hits", default=None)
        self.store_as = "samples"

    def process(self, blob):
        if self.fixed_length and self.max_n_hits is None:
            raise ValueError("Have to specify max_n_hits if fixed_length is True")
        if self.hit_infos is None:
            self.hit_infos = blob["Hits"].dtype.names
        points, n_hits = self.get_points(blob)
        blob[self.store_as] = kp.NDArray(points, h5loc="x", title="nodes")
        if self.dset_n_hits:
            blob[self.dset_n_hits] = blob[self.dset_n_hits].append_columns(
                "n_hits_intime", n_hits)
        return blob

    def get_points(self, blob):
        """
        Get the desired hit infos from the blob.

        Returns
        -------
        points : np.array
            The hit infos of this event as a 2d matrix. No of rows are
            fixed to the given max_n_hits. Each of the self.extract_keys,
            is in one column + an additional column which is 1 for
            actual hits, and 0 for if its a padded row.
        n_hits : int
            Number of hits in the given time window.
            Can be stored as n_hits_intime.

        """
        hits = blob["Hits"]
        if self.only_triggered_hits:
            hits = hits[hits.triggered != 0]
        if self.time_window is not None:
            # remove hits outside of time window
            hits = hits[np.logical_and(
                hits["time"] >= self.time_window[0],
                hits["time"] <= self.time_window[1],
            )]

        n_hits = len(hits)
        if self.max_n_hits is not None and n_hits > self.max_n_hits:
            # if there are too many hits, take random ones, but keep order
            indices = np.arange(n_hits)
            np.random.shuffle(indices)
            which = indices[:self.max_n_hits]
            which.sort()
            hits = hits[which]

        if self.fixed_length:
            points = np.zeros(
                (self.max_n_hits, len(self.hit_infos) + 1), dtype="float32")
            for i, which in enumerate(self.hit_infos):
                points[:n_hits, i] = hits[which]
            # last column is whether there was a hit or no
            points[:n_hits, -1] = 1.
            # store along new axis
            points = np.expand_dims(points, 0)
        else:
            points = np.zeros(
                (len(hits), len(self.hit_infos)), dtype="float32")
            for i, which in enumerate(self.hit_infos):
                points[:, i] = hits[which]
        return points, n_hits

    def finish(self):
        columns = tuple(self.hit_infos)
        if self.fixed_length:
            columns += ("is_valid", )
        return {"hit_infos": columns}


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
    Apply detector information to the event data from a detx file, e.g.
    calibrating hits.

    Attributes
    ----------
    det_file : str
        Path to a .detx detector geometry file.
    calib_hits : bool
        Apply calibration to hits. Default: True.
    calib_mchits : bool
        Apply calibration to mchits, if mchits are in the blob. Default: True.
    correct_timeslew : bool
        If true (default), the time slewing of hits depending on their tot
        will be corrected. Only done if calib_hits is True.
    center_hits_to : tuple, optional
        Translate the xyz positions of the hits (and mchits), as if
        the detector was centered at the given position.
        E.g., if its (0, 0, None), the hits and mchits will be
        centered at xy = 00, and z will be left untouched.

    """
    def configure(self):
        self.det_file = self.require("det_file")
        self.correct_timeslew = self.get("correct_timeslew", default=True)
        self.calib_hits = self.get("calib_hits", default=True)
        self.calib_mchits = self.get("calib_mchits", default=True)
        self.center_hits_to = self.get("center_hits_to", default=None)

        self.cprint(f"Calibrating with {self.det_file}")
        self.calib = kp.calib.Calibration(filename=self.det_file)
        self._calib_checked = False

        # dict  dim_name: float
        self._vector_shift = None

        if self.center_hits_to:
            self._cache_shift_center()

    def process(self, blob):
        if (self.calib_hits or self.calib_mchits) and self._calib_checked is False:
            if "pos_x" in blob["Hits"]:
                self.log.warn(
                    "Warning: Using a det file, but pos_x in Hits detected. "
                    "Is the file already calibrated? This might lead to "
                    "errors with t0."
                )
            self._calib_checked = True
        if self.calib_hits:
            blob["Hits"] = self.calib.apply(
                blob["Hits"], correct_slewing=self.correct_timeslew)
        if self.calib_mchits and "McHits" in blob:
            blob["McHits"] = self.calib.apply(blob["McHits"])
        if self.center_hits_to:
            self.shift_hits(blob)
        return blob

    def shift_hits(self, blob):
        """ Translate hits by cached vector. """
        for dim_name in ("pos_x", "pos_y", "pos_z"):
            blob["Hits"][dim_name] += self._vector_shift[dim_name]
            if "McHits" in blob:
                blob["McHits"][dim_name] += self._vector_shift[dim_name]

    def _cache_shift_center(self):
        det_center, shift = {}, {}
        for i, dim_name in enumerate(("pos_x", "pos_y", "pos_z")):
            center = self.calib.detector.dom_table[dim_name].mean()
            det_center[dim_name] = center

            if self.center_hits_to[i] is None:
                shift[dim_name] = 0
            else:
                shift[dim_name] = self.center_hits_to[i] - center

        self._vector_shift = shift
        self.cprint(f"original detector center: {det_center}")
        self.cprint(f"shift for hits: {self._vector_shift}")


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

