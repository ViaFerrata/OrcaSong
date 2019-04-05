"""
Custom km3pipe modules for making nn input files.
"""

import km3pipe as kp
import numpy as np


class McInfoMaker(kp.Module):
    """
    Get the desired mc_info from the blob.
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
    def configure(self):
        self.correct_hits = self.get('correct_hits', default=True)
        self.correct_mchits = self.get('correct_mchits', default=True)

    def process(self, blob):
        blob = time_preproc(blob, self.correct_hits, self.correct_mchits)
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
