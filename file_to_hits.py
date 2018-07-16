#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This utility code contains functions that read the raw MC .h5 files"""

import numpy as np
#from memory_profiler import profile
#import line_profiler # call with kernprof -l -v file.py args


def get_primary_track_index(event_blob):
    """
    Gets the index of the primary (neutrino) track.
    Uses bjorkeny in order to get the primary track, since bjorkeny!=0 for the initial interacting neutrino.
    :param kp.io.HDF5Pump.blob event_blob: HDF5Pump event blob.
    :return: int primary index: Index of the primary track (=neutrino) in the 'McTracks' branch.
    """
    bjorken_y_array = event_blob['McTracks'].bjorkeny
    primary_index = np.where(bjorken_y_array != 0.0)[0][0]
    return primary_index


def get_time_residual_nu_interaction_mean_triggered_hits(time_interaction, hits_time, triggered):
    """

    :param time_interaction:
    :param hits_time:
    :param triggered:
    :return:
    """
    hits_time_triggered = hits_time[triggered == 1]
    t_mean_triggered = np.mean(hits_time_triggered, dtype=np.float64)
    time_residual_vertex = t_mean_triggered - time_interaction

    return time_residual_vertex


def get_event_data(event_blob, geo, do_mc_hits, use_calibrated_file, data_cuts, do4d):
    """
    Reads a km3pipe blob which contains the information for one event.
    Returns a hit array and a track array that contains all relevant information of the event.
    :param kp.io.HDF5Pump.blob event_blob: Event blob of the HDF5Pump which contains all information for one event.
    :param kp.Geometry geo: km3pipe Geometry instance that contains the geometry information of the detector.
                            Only used if the event_blob is from a non-calibrated file!
    :param bool do_mc_hits: tells the function of the hits (mc_hits + BG) or the mc_hits only should be parsed.
                            In the case of mc_hits, the dom_id needs to be calculated thanks to the jpp output.
    :param bool use_calibrated_file: specifies if a calibrated file is used as an input for the event_blob.
                                     If False, the hits of the event_blob are calibrated based on the geo parameter.
    :param dict data_cuts: specifies if cuts should be applied. Contains the keys 'triggered' and 'energy_lower_limit'.
    :param (bool, str) do4d: Tuple that declares if 4D histograms should be created [0] and if yes, what should be used as the 4th dim after xyz.
                             In the case of 'channel_id', this information needs to be included in the event_hits as well.
    :return: ndarray(ndim=2) event_hits: 2D array containing the hit information of the event [pos_xyz time (channel_id)].
    :return: ndarray(ndim=1) event_track: 1D array containing important MC information of the event.
                                          [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
    """
    p = get_primary_track_index(event_blob)

    # parse tracks [event_id, particle_type, energy, isCC, bjorkeny, dir_x/y/z, time]
    event_id = event_blob['EventInfo'].event_id[0]
    run_id = event_blob['RawHeader'][0][0].astype('float32')
    particle_type = event_blob['McTracks'][p].type
    energy = event_blob['McTracks'][p].energy
    is_cc = event_blob['McTracks'][p].is_cc
    bjorkeny = event_blob['McTracks'][p].bjorkeny
    dir_x, dir_y, dir_z = event_blob['McTracks'][p].dir_x, event_blob['McTracks'][p].dir_y, event_blob['McTracks'][p].dir_z
    time_track = event_blob['McTracks'][p].time # actually always 0 for primary neutrino, measured in MC time
    vertex_pos_x, vertex_pos_y, vertex_pos_z = event_blob['McTracks'][p].pos_x, event_blob['McTracks'][p].pos_y, event_blob['McTracks'][p].pos_z
    time_interaction = event_blob['McTracks'][p].time

    # parse hits [x,y,z,time]
    if do_mc_hits is True:
        hits = event_blob["McHits"]
    else:
        hits = event_blob["Hits"]

    if use_calibrated_file is False:
        hits = geo.apply(hits)

    if data_cuts['triggered'] is True:
        hits = hits.__array__[hits.triggered.astype(bool)]
        #hits = hits.triggered_hits # alternative, though it only works for the triggered condition!

    pos_x, pos_y, pos_z = hits.pos_x.astype('float32'), hits.pos_y.astype('float32'), hits.pos_z.astype('float32')
    hits_time = hits.time.astype('float32') # enough for the hit times in KM3NeT
    triggered = hits.triggered.astype('float32')

    time_residual_vertex = get_time_residual_nu_interaction_mean_triggered_hits(time_interaction, hits_time, triggered)

    # save collected information to arrays event_track and event_hits
    event_track = np.array([event_id, particle_type, energy, is_cc, bjorkeny, dir_x, dir_y, dir_z, time_track,
                            run_id, vertex_pos_x, vertex_pos_y, vertex_pos_z, time_residual_vertex], dtype=np.float64)

    ax = np.newaxis
    event_hits = np.concatenate([pos_x[:, ax], pos_y[:, ax], pos_z[:, ax], hits_time[:, ax], triggered[:, ax]], axis=1)

    if do4d[0] is True and do4d[1] == 'channel_id' or do4d[1] == 'xzt-c':
        channel_id = hits.channel_id.astype('float32')
        event_hits = np.concatenate([event_hits, channel_id[:, ax]], axis=1)

    return event_hits, event_track