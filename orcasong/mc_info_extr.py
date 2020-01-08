"""
Functions that extract info from a blob for the mc_info / y datafield
in the h5 files.

These are made for the specific given runs. They might not be
applicable to other data, and could cause errors or produce unexpected
results when used on data other then the specified.

"""

import warnings
import numpy as np

__author__ = 'Stefan Reck'


def get_real_data(blob):
    """
    Get info present in real data.
    Designed for the 2017 one line runs.

    """
    event_info = blob['EventInfo']

    track = {
        'event_id': event_info.event_id,
        # was .event_id[0] up to km3pipe 8.16.0
        'run_id': event_info.run_id,
        'trigger_mask': event_info.trigger_mask,
    }
    return track


def get_pure_noise(blob):
    """
    For simulated pure noise events, which have particle_type 0.

    """
    event_info = blob['EventInfo']

    track = {
        'event_id': event_info.event_id[0],
        'run_id': event_info.run_id,
        'particle_type': 0
    }
    return track


def get_mupage_mc(blob):
    """
    For mupage muon simulations.

    Will only take into account muons with at least 1 McHit in the active
    line of the detector.

    Designed for the 2017 run by run mupage simulations.
    e.g. mcv5.1_r3.mupage_10G.km3_AAv1.jterbr00002800.5103.root.h5

    Parameters
    ----------
    blob : dict
        The blob from the pipeline.

    Returns
    -------
    track : dict
        The info for mc_info.

    """
    # only one line has hits, but there are two for the mc. This one is active:
    active_du = 2

    track = dict()

    # total number of simulated muons, not all might deposit counts
    n_muons_sim = blob['McTracks'].shape[0]
    track["n_muons_sim"] = n_muons_sim

    # Origin of each mchit (as int) in the active line
    in_active_du = blob["McHits"]["du"] == active_du
    origin = blob["McHits"]["origin"][in_active_du]
    # get how many mchits were produced per muon in the bundle
    origin_dict = dict(zip(*np.unique(origin, return_counts=True)))
    origin_list = []
    for i in range(1, n_muons_sim + 1):
        origin_list.append(origin_dict.get(i, 0))
    # origin_list[i] is num of mc_hits of muon i in active du
    origin_list = np.array(origin_list)

    visible_mc_tracks = blob["McTracks"][origin_list > 0]
    visible_origin_list = origin_list[origin_list > 0]

    # only muons with at least one mchit in active line
    n_muons = len(visible_origin_list)
    if n_muons == 0:
        warnings.warn("No visible muons in blob!")

    track["n_muons"] = n_muons

    track["event_id"] = blob['EventInfo'].event_id[0]
    track["run_id"] = blob["EventInfo"].run_id[0]
    # run_id = blob['Header'].start_run.run_id.astype('float32')

    # take 0: assumed that this is the same for all muons in a bundle
    track["particle_type"] = visible_mc_tracks[0].type if n_muons != 0 else 0

    # always 1 actually
    # track["is_cc"] = blob['McTracks'][0].is_cc

    # always 0 actually
    # track["bjorkeny"] = blob['McTracks'][0].bjorkeny

    # same for all muons in a bundle TODO not?
    track["time_interaction"] = visible_mc_tracks[0].time if n_muons != 0 else 0

    # sum up the energy of all visible muons
    energy = visible_mc_tracks.energy if n_muons != 0 else 0
    track["energy"] = np.sum(energy)

    track["n_mc_hits"] = np.sum(visible_origin_list)
    desc_order = np.argsort(-visible_origin_list)

    # Sort by energy, highest first
    sorted_energy = energy[desc_order] if n_muons != 0 else []
    sorted_mc_hits = visible_origin_list[desc_order] if n_muons != 0 else []

    # Store number of mchits of the 10 highest mchits muons (-1 if it has less)
    for i in range(10):
        field_name = "n_mc_hits_"+str(i)

        if i < len(sorted_mc_hits):
            field_data = sorted_mc_hits[i]
        else:
            field_data = -1

        track[field_name] = field_data

    # Store energy of the 10 highest mchits muons (-1 if it has less)
    for i in range(10):
        field_name = "energy_"+str(i)

        if i < len(sorted_energy):
            field_data = sorted_energy[i]
        else:
            field_data = -1

        track[field_name] = field_data

    # only muons with at least 5 mchits in active line
    track["n_muons_5_mchits"] = len(origin_list[origin_list > 4])
    # only muons with at least 10 mchits in active line
    track["n_muons_10_mchits"] = len(origin_list[origin_list > 9])

    # all muons in a bundle are parallel, so just take dir of first muon
    track["dir_x"] = visible_mc_tracks[0].dir_x if n_muons != 0 else 0
    track["dir_y"] = visible_mc_tracks[0].dir_y if n_muons != 0 else 0
    track["dir_z"] = visible_mc_tracks[0].dir_z if n_muons != 0 else 0

    if n_muons != 0:
        # vertex is the weighted (energy) mean of the individual vertices
        track["vertex_pos_x"] = np.average(visible_mc_tracks[:].pos_x,
                                           weights=visible_mc_tracks[:].energy)
        track["vertex_pos_y"] = np.average(visible_mc_tracks[:].pos_y,
                                           weights=visible_mc_tracks[:].energy)
        track["vertex_pos_z"] = np.average(visible_mc_tracks[:].pos_z,
                                           weights=visible_mc_tracks[:].energy)
    else:
        track["vertex_pos_x"] = 0
        track["vertex_pos_y"] = 0
        track["vertex_pos_z"] = 0

    return track
