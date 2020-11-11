"""
Functions that extract info from a blob for the mc_info / y datafield
in the h5 files.

These are made for the specific given runs. They might not be
applicable to other data, and could cause errors or produce unexpected
results when used on data other then the specified.

"""

import warnings
import numpy as np
from km3pipe.io.hdf5 import HDF5Header
from h5py import File

__author__ = "Daniel Guderian"


def get_std_reco(blob):

    """
    Function to extract std reco info. The implemented strategy is the following:
    First, look for whether a rec stag has been reached and only then extract the reconstructed
    paramater from it. If not, set it to a dummy value (for now 0). This means that for an analysis the events with
    exactly zero have to be filtered out!

    The 'best track' is the first (highest lik) while a certain rec stage has to be reached. This might
    have to be adjusted for other recos than JMuonGandalf chain.


    Members of the Tracks:
    dtype([('E', '<f8'), ('JCOPY_Z_M', '<f4'), ('JENERGY_CHI2', '<f4'), ('JENERGY_ENERGY', '<f4'),
     ('JENERGY_MUON_RANGE_METRES', '<f4'), ('JENERGY_NDF', '<f4'), ('JENERGY_NOISE_LIKELIHOOD', '<f4'),
     ('JENERGY_NUMBER_OF_HITS', '<f4'), ('JGANDALF_BETA0_RAD', '<f4'), ('JGANDALF_BETA1_RAD', '<f4'),
     ('JGANDALF_CHI2', '<f4'), ('JGANDALF_LAMBDA', '<f4'), ('JGANDALF_NUMBER_OF_HITS', '<f4'),
     ('JGANDALF_NUMBER_OF_ITERATIONS', '<f4'), ('JSHOWERFIT_ENERGY', '<f4'), ('JSTART_LENGTH_METRES', '<f4'),
     ('JSTART_NPE_MIP', '<f4'), ('JSTART_NPE_MIP_TOTAL', '<f4'), ('JVETO_NPE', '<f4'), ('JVETO_NUMBER_OF_HITS', '<f4'),
     ('dir_x', '<f8'), ('dir_y', '<f8'), ('dir_z', '<f8'), ('id', '<i4'), ('idx', '<i8'), ('length', '<f8'),
     ('likelihood', '<f8'), ('pos_x', '<f8'), ('pos_y', '<f8'), ('pos_z', '<f8'), ('rec_type', '<i4'),
     ('t', '<f8'), ('group_id', '<i8')])


    members of rec stages:
    .idx (corresponding to the track id),
    .rec_stage (rec stage identifier, for JMuonGandalf for example: 1=prefit, 2=simplex, 3=gandalf,
                    4=engery, 5=start),
    .group_id (event id in file)


    Parameters
    ----------
    blob : blob containing the reco info

    Returns
    -------
    std_reco_info : dict
            Dict with the most common std reco params. Can be expanded.

    """

    # use this later to identify not reconstructed events
    dummy_value = 0

    # if there was no std reco at all, this will not exist
    # these are events that stopped at/before prefit
    try:
        rec_stages = blob["RecStages"]
        # get first track only
        rec_stages_best_track = rec_stages.rec_stage[rec_stages.idx == 0]

        # often enough: best track is the first
        best_track = blob["Tracks"][0]

    except KeyError:
        rec_stages_best_track = []
        print(
            "An event didnt have any reco. Setting everything to"
            + str(dummy_value)
            + "."
        )

    # take the direction only if JGanalf was executed
    if 3 in rec_stages_best_track:

        std_dir_x = best_track["dir_x"]
        std_dir_y = best_track["dir_y"]
        std_dir_z = best_track["dir_z"]

        std_beta0 = best_track["JGANDALF_BETA0_RAD"]
        std_lik = best_track["likelihood"]
        std_n_hits_gandalf = best_track["JGANDALF_NUMBER_OF_HITS"]

    else:

        std_dir_x = dummy_value
        std_dir_y = dummy_value
        std_dir_z = dummy_value

        std_beta0 = dummy_value
        std_lik = dummy_value
        std_n_hits_gandalf = dummy_value

    # energy fit from JEnergy
    if 4 in rec_stages_best_track:

        std_energy = best_track["E"]
        lik_energy = best_track["JENERGY_CHI2"]

    else:
        std_energy = dummy_value
        lik_energy = dummy_value

    # vertex and length from JStart
    if 5 in rec_stages_best_track:

        std_pos_x = best_track["pos_x"]
        std_pos_y = best_track["pos_y"]
        std_pos_z = best_track["pos_z"]

        std_length = best_track["JSTART_LENGTH_METRES"]

    else:

        std_pos_x = dummy_value
        std_pos_y = dummy_value
        std_pos_z = dummy_value

        std_length = dummy_value

    std_reco_info = {
        "std_dir_x": std_dir_x,
        "std_dir_y": std_dir_y,
        "std_dir_z": std_dir_z,
        "std_beta0": std_beta0,
        "std_lik": std_lik,
        "std_n_hits_gandalf": std_n_hits_gandalf,
        "std_pos_x": std_pos_x,
        "std_pos_y": std_pos_y,
        "std_pos_z": std_pos_z,
        "std_energy": std_energy,
        "std_lik_energy": lik_energy,
        "std_length": std_length,
    }

    return std_reco_info


def get_real_data_info_extr(input_file):

    """
    Wrapper function that includes the actual mc_info_extr
    for real data. There are no n_gen like in the neutrino case.

    Parameters
    ----------
    input_file : km3net data file
            Can be online or offline format.

    Returns
    -------
    mc_info_extr : function
            The actual mc_info_extr function that holds the extractions.

    """

    # check if std reco is present
    f = File(input_file, "r")
    has_std_reco = "reco" in f.keys()

    def mc_info_extr(blob):

        """
        Processes a blob and creates the y with mc_info and, if existing, std reco.

        For this real data case it is only general event info, like the id.

        Parameters
        ----------
        blob : dict
                The blob from the pipeline.

        Returns
        -------
        track : dict
                Containing all the specified info the y should have.

        """

        event_info = blob["EventInfo"][0]

        # add n_hits info for the cut
        n_hits = len(blob["Hits"])

        track = {
            "event_id": event_info.event_id,
            "run_id": event_info.run_id,
            "trigger_mask": event_info.trigger_mask,
            "n_hits": n_hits,
        }

        # get all the std reco info
        if has_std_reco:

            std_reco_info = get_std_reco(blob)

            track.update(std_reco_info)

        return track

    return mc_info_extr


def get_random_noise_mc_info_extr(input_file):

    """
    Wrapper function that includes the actual mc_info_extr
    for random noise simulations. There are no n_gen like in the neutrino case.

    Parameters
    ----------
    input_file : km3net data file
            Can be online or offline format.

    Returns
    -------
    mc_info_extr : function
            The actual mc_info_extr function that holds the extractions.

    """

    # check if std reco is present
    f = File(input_file, "r")
    has_std_reco = "reco" in f.keys()

    def mc_info_extr(blob):

        """
        Processes a blob and creates the y with mc_info and, if existing, std reco.

        For this random noise case it is only general event info, like the id.

        Parameters
        ----------
        blob : dict
                The blob from the pipeline.

        Returns
        -------
        track : dict
                Containing all the specified info the y should have.

        """
        event_info = blob["EventInfo"]

        track = {
            "event_id": event_info.event_id[0],
            "run_id": event_info.run_id[0],
            "particle_type": 0,
        }

        # get all the std reco info
        if has_std_reco:

            std_reco_info = get_std_reco(blob)

            track.update(std_reco_info)

        return track

    return mc_info_extr


def get_neutrino_mc_info_extr(input_file):

    """
    Wrapper function that includes the actual mc_info_extr
    for neutrino simulations. The n_gen parameter, needed for neutrino weighting
    is extracted from the header of the file.

    Parameters
    ----------
    input_file : km3net data file
            Can be online or offline format.

    Returns
    -------
    mc_info_extr : function
            The actual mc_info_extr function that holds the extractions.

    """

    # check if std reco is present
    f = File(input_file, "r")
    has_std_reco = "reco" in f.keys()

    # get the n_gen
    header = HDF5Header.from_hdf5(input_file)
    n_gen = header.genvol.numberOfEvents

    def mc_info_extr(blob):

        """
        Processes a blob and creates the y with mc_info and, if existing, std reco.

        For this neutrino case it is the full mc info for the primary neutrino; there are the several "McTracks":
        check the simulation which index "p" the neutrino has.

        Parameters
        ----------
        blob : dict
                The blob from the pipeline.

        Returns
        -------
        track : dict
                Containing all the specified info the y should have.

        """

        # get general info about the event
        event_id = blob["EventInfo"].event_id[0]
        run_id = blob["EventInfo"].run_id[0]
        # weights for neutrino analysis
        weight_w1 = blob["EventInfo"].weight_w1[0]
        weight_w2 = blob["EventInfo"].weight_w2[0]
        weight_w3 = blob["EventInfo"].weight_w3[0]

        # first, look for the particular neutrino index of the production
        p = 0  # for ORCA4 (and probably subsequent productions)

        mc_track = blob["McTracks"][p]

        # some track mc truth info
        particle_type = mc_track.type
        energy = mc_track.energy
        is_cc = mc_track.cc
        bjorkeny = mc_track.by
        dir_x, dir_y, dir_z = mc_track.dir_x, mc_track.dir_y, mc_track.dir_z
        time_interaction = (
            mc_track.time
        )  # actually always 0 for primary neutrino, measured in MC time
        vertex_pos_x, vertex_pos_y, vertex_pos_z = (
            mc_track.pos_x,
            mc_track.pos_y,
            mc_track.pos_z,
        )

        # add also the nhits info
        n_hits = len(blob["Hits"])

        track = {
            "event_id": event_id,
            "particle_type": particle_type,
            "energy": energy,
            "is_cc": is_cc,
            "bjorkeny": bjorkeny,
            "dir_x": dir_x,
            "dir_y": dir_y,
            "dir_z": dir_z,
            "time_interaction": time_interaction,
            "run_id": run_id,
            "vertex_pos_x": vertex_pos_x,
            "vertex_pos_y": vertex_pos_y,
            "vertex_pos_z": vertex_pos_z,
            "n_hits": n_hits,
            "weight_w1": weight_w1,
            "weight_w2": weight_w2,
            "weight_w3": weight_w3,
            "n_gen": n_gen,
        }
        # get all the std reco info
        if has_std_reco:

            std_reco_info = get_std_reco(blob)

            track.update(std_reco_info)

        return track

    return mc_info_extr


def get_muon_mc_info_extr(input_file):

    """
    Wrapper function that includes the actual mc_info_extr
    for atm. muon simulations. There are no n_gen like in the neutrino case.

    Parameters
    ----------
    input_file : km3net data file
            Can be online or offline format.

    Returns
    -------
    mc_info_extr : function
            The actual mc_info_extr function that holds the extractions.

    """

    # check if std reco is present
    f = File(input_file, "r")
    has_std_reco = "reco" in f.keys()

    # no n_gen here, but needed for concatenation
    n_gen = 1

    def mc_info_extr(blob):

        """
        Processes a blob and creates the y with mc_info and, if existing, std reco.

        For this atm. muon case it is the full mc info for the primary; there are the several "McTracks":
        check the simulation to understand what "p" you want. Muons come in bundles that have the same direction.
        For energy: sum of all muons in a bundle,
        for vertex: weighted (energy) mean of the individual vertices .

        Parameters
        ----------
        blob : dict
                The blob from the pipeline.

        Returns
        -------
        track : dict
                Containing all the specified info the y should have.

        """

        event_id = blob["EventInfo"].event_id[0]
        run_id = blob["EventInfo"].run_id[0]

        p = 0  # for ORCA4 (and probably subsequent productions)

        mc_track = blob["McTracks"][p]

        particle_type = (
            mc_track.type
        )  # assumed that this is the same for all muons in a bundle
        is_cc = mc_track.cc  # always 0 actually
        bjorkeny = mc_track.by
        time_interaction = mc_track.time  # same for all muons in a bundle

        # sum up the energy of all muons
        energy = np.sum(blob["McTracks"].energy)

        # all muons in a bundle are parallel, so just take dir of first muon
        dir_x, dir_y, dir_z = mc_track.dir_x, mc_track.dir_y, mc_track.dir_z

        # vertex is the weighted (energy) mean of the individual vertices
        vertex_pos_x = np.average(
            blob["McTracks"][p:].pos_x, weights=blob["McTracks"][p:].energy
        )
        vertex_pos_y = np.average(
            blob["McTracks"][p:].pos_y, weights=blob["McTracks"][p:].energy
        )
        vertex_pos_z = np.average(
            blob["McTracks"][p:].pos_z, weights=blob["McTracks"][p:].energy
        )

        # add also the nhits info
        n_hits = len(blob["Hits"])

        # this is only relevant for neutrinos, though dummy info is needed for the concatenation
        weight_w1 = 10
        weight_w2 = 10
        weight_w3 = 10

        track = {
            "event_id": event_id,
            "particle_type": particle_type,
            "energy": energy,
            "is_cc": is_cc,
            "bjorkeny": bjorkeny,
            "dir_x": dir_x,
            "dir_y": dir_y,
            "dir_z": dir_z,
            "time_interaction": time_interaction,
            "run_id": run_id,
            "vertex_pos_x": vertex_pos_x,
            "vertex_pos_y": vertex_pos_y,
            "vertex_pos_z": vertex_pos_z,
            "n_hits": n_hits,
            "weight_w1": weight_w1,
            "weight_w2": weight_w2,
            "weight_w3": weight_w3,
            "n_gen": n_gen,
        }

        # get all the std reco info
        if has_std_reco:

            std_reco_info = get_std_reco(blob)

            track.update(std_reco_info)

        return track

    return mc_info_extr
