"""
Functions that extract info from a blob for the mc_info / y datafield
in the h5 files.
"""

import numpy as np


def get_mc_info_extr(mc_info_extr):
    """
    Get an existing mc info extractor function.

    Attributes
    ----------
    mc_info_extr : function
        Function to extract the info. Takes the blob as input, outputs
        a dict with the desired mc_infos.

    """
    if mc_info_extr == "mupage":
        mc_info_extr = get_mupage_mc

    elif mc_info_extr == "event_and_run_id":
        mc_info_extr = get_event_and_run_id

    else:
        raise ValueError("Unknown mc_info_type " + mc_info_extr)

    return mc_info_extr


def get_event_and_run_id(blob):
    """
    Get event id and run id from event info.
    E.g. for the 2017 one line real data.
    """
    event_id = blob['EventInfo'].event_id[0]
    run_id = blob["EventInfo"].run_id

    track = {'event_id': event_id,
             'run_id': run_id, }
    return track


def get_mupage_mc(blob):
    """
    For mupage muon simulations.

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

    track["event_id"] = blob['EventInfo'].event_id[0]
    track["run_id"] = blob["EventInfo"].run_id[0]
    # run_id = blob['Header'].start_run.run_id.astype('float32')

    # take 0: assumed that this is the same for all muons in a bundle
    track["particle_type"] = blob['McTracks'][0].type

    # always 1 actually
    # track["is_cc"] = blob['McTracks'][0].is_cc

    # always 0 actually
    # track["bjorkeny"] = blob['McTracks'][0].bjorkeny

    # same for all muons in a bundle #TODO not?
    track["time_interaction"] = blob['McTracks'][0].time

    # takes position of time_residual_vertex in 'neutrino' case
    n_muons = blob['McTracks'].shape[0]
    track["n_muons"] = n_muons

    # sum up the energy of all muons
    energy = blob['McTracks'].energy
    track["energy"] = np.sum(energy)

    # Origin of each mchit (as int) in the active line
    in_active_du = blob["McHits"]["du"] == active_du
    origin = blob["McHits"]["origin"][in_active_du]

    # get how many mchits were produced per muon in the bundle
    origin_dict = dict(zip(*np.unique(origin, return_counts=True)))
    origin_list = []
    for i in range(1, n_muons+1):
        origin_list.append(origin_dict.get(i, 0))
    origin_list = np.array(origin_list)
    track["n_mc_hits"] = np.sum(origin_list)
    desc_order = np.argsort(-origin_list)

    # Sort by energy, highest first
    sorted_energy = energy[desc_order]
    sorted_mc_hits = origin_list[desc_order]

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

    # only muons with at least one mchit in active line
    track["n_muons_visible"] = len(origin_list[origin_list > 0])
    # only muons with at least 5 mchits in active line
    track["n_muons_5_mchits"] = len(origin_list[origin_list > 4])
    # only muons with at least 10 mchits in active line
    track["n_muons_10_mchits"] = len(origin_list[origin_list > 9])

    # all muons in a bundle are parallel, so just take dir of first muon
    track["dir_x"] = blob['McTracks'][0].dir_x
    track["dir_y"] = blob['McTracks'][0].dir_y
    track["dir_z"] = blob['McTracks'][0].dir_z

    # vertex is the weighted (energy) mean of the individual vertices
    track["vertex_pos_x"] = np.average(blob['McTracks'][:].pos_x,
                                       weights=blob['McTracks'][:].energy)
    track["vertex_pos_y"] = np.average(blob['McTracks'][:].pos_y,
                                       weights=blob['McTracks'][:].energy)
    track["vertex_pos_z"] = np.average(blob['McTracks'][:].pos_z,
                                       weights=blob['McTracks'][:].energy)

    return track
