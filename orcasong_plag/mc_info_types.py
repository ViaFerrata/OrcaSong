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

    else:
        raise ValueError("Unknown mc_info_type " + mc_info_extr)

    return mc_info_extr


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
    event_id = blob['EventInfo'].event_id[0]
    run_id = blob["EventInfo"].run_id
    # run_id = blob['Header'].start_run.run_id.astype('float32')

    # take 0: assumed that this is the same for all muons in a bundle
    particle_type = blob['McTracks'][0].type
    # always 1 actually
    is_cc = blob['McTracks'][0].is_cc
    # always 0 actually
    bjorkeny = blob['McTracks'][0].bjorkeny
    # same for all muons in a bundle #TODO not?
    time_interaction = blob['McTracks'][0].time
    # takes position of time_residual_vertex in 'neutrino' case
    n_muons = blob['McTracks'].shape[0]

    # sum up the energy of all muons
    energy = np.sum(blob['McTracks'].energy)

    # all muons in a bundle are parallel, so just take dir of first muon
    dir_x = blob['McTracks'][0].dir_x
    dir_y = blob['McTracks'][0].dir_y
    dir_z = blob['McTracks'][0].dir_z

    # vertex is the weighted (energy) mean of the individual vertices
    vertex_pos_x = np.average(blob['McTracks'][:].pos_x,
                              weights=blob['McTracks'][:].energy)
    vertex_pos_y = np.average(blob['McTracks'][:].pos_y,
                              weights=blob['McTracks'][:].energy)
    vertex_pos_z = np.average(blob['McTracks'][:].pos_z,
                              weights=blob['McTracks'][:].energy)

    track = {'event_id': event_id,
             'particle_type': particle_type,
             'energy': energy,
             'is_cc': is_cc,
             'bjorkeny': bjorkeny,
             'dir_x': dir_x,
             'dir_y': dir_y,
             'dir_z': dir_z,
             'time_interaction': time_interaction,
             'run_id': run_id,
             'vertex_pos_x': vertex_pos_x,
             'vertex_pos_y': vertex_pos_y,
             'vertex_pos_z': vertex_pos_z,
             'n_muons': n_muons}

    return track
