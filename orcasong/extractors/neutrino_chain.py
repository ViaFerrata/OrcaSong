"""
Functions that extract info from a blob for the mc_info / y datafield
in the h5 files.

These are made for the specific given runs. They might not be
applicable to other data, and could cause errors or produce unexpected
results when used on data other then the specified. Check for example the 
primary position in the mc_tracks.

"""

import warnings
import numpy as np
from km3pipe.io.hdf5 import HDF5Header
from h5py import File
import os, re

__author__ = "Daniel Guderian"


def get_std_reco(blob, rec_types, rec_parameters_names):

    """
    Function to extract std reco info. This implementation requires h5 files
    to be processed with the option "--best_tracks" which adds the selection
    of best tracks for each reco type to the output using the km3io tools.

    Returns
    -------
    std_reco_info : dict
                    Dict with the std reco info of the best tracks.

    """
    # this dict will be filled up
    std_reco_info = {}

    # all known reco types to iterate over
    reco_type_dict = {
        "BestJmuon": ("jmuon_", "best_jmuon"),
        "BestJshower": ("jshower_", "best_jshower"),
        "BestDusjshower": ("dusjshower_", "best_dusjshower"),
        "BestAashower": ("aashower_", "best_aashower"),
    }

    for name_in_blob, (identifier, best_track_name) in reco_type_dict.items():

        # always write out something for the generally present rec types
        if best_track_name in rec_types:

            # specific names are with the prefix from the rec type
            specific_reco_names = np.core.defchararray.add(
                identifier, rec_parameters_names
            )

            # extract actually present info
            if name_in_blob in blob:

                # get the previously identified best track
                bt = blob[name_in_blob]

                # get all its values
                values = bt.item()
                values_list = list(values)
                # reco_names = bt.dtype.names #in case the fitinf and stuff will be tailored to the reco types
                # at some point, get the names directly like this

            # in case there is no reco for this event but the reco type was done in general
            else:

                # fill all values with nan's
                values_array = np.empty(len(specific_reco_names))
                values_array[:] = np.nan
                values_list = values_array.tolist()

            # create a dict out of them
            keys_list = list(specific_reco_names)

            zip_iterator = zip(keys_list, values_list)
            reco_dict = dict(zip_iterator)

            # add this dict to the complete std reco collection
            std_reco_info.update(reco_dict)

    return std_reco_info


def get_rec_types_in_file(file):

    """
    Checks and returns which rec types are in the file and thus need to be present
    in all best track and their fitinf information later.
    """

    # the known rec types
    rec_type_names = ["best_jmuon", "best_jshower", "best_dusjshower", "best_aashower"]

    # all reco related objects in the file
    reco_objects_in_file = file["reco"].keys()

    # check which ones are in there
    rec_types_in_file = []
    for rec_type in rec_type_names:
        if rec_type in reco_objects_in_file:
            rec_types_in_file.append(rec_type)

            # also get from here the list of dtype names that is share for all recos
            rec_parameters_names = file["reco"][rec_type].dtype.names

    return rec_types_in_file, rec_parameters_names


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

    if has_std_reco:
        # also check, which rec types are present
        rec_types, rec_parameters_names = get_rec_types_in_file(f)

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
        n_trig_hits = np.count_nonzero(blob["Hits"]["triggered"])

        track = {
            "event_id": event_info.event_id,
            "run_id": event_info.run_id,
            "trigger_mask": event_info.trigger_mask,
            "n_hits": n_hits,
            "n_trig_hits": n_trig_hits,
        }

        # get all the std reco info
        if has_std_reco:

            std_reco_info = get_std_reco(blob, rec_types, rec_parameters_names)

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

    if has_std_reco:
        # also check, which rec types are present
        rec_types, rec_parameters_names = get_rec_types_in_file(f)

    # an identifier for what the part of the mc simulation this was
    # this way, events can later be unambiguously identified
    input_filename_string = os.path.basename(input_file)
    part_number = re.findall(r"\d+", input_filename_string)[
        -2
    ]  # second last because of .h5

    # dummy value for concatenation
    tau_topology = 3

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

        # add also the nhits info
        n_hits = len(blob["Hits"])
        n_trig_hits = np.count_nonzero(blob["Hits"]["triggered"])

        track = {
            "event_id": event_info.event_id[0],
            "run_id": event_info.run_id[0],
            "particle_type": 0,
            "part_number": part_number,
            "n_hits": n_hits,
            "n_trig_hits": n_trig_hits,
            "tau_topology": tau_topology,
        }

        # get all the std reco info
        if has_std_reco:

            std_reco_info = get_std_reco(blob, rec_types, rec_parameters_names)

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

    if has_std_reco:
        # also check, which rec types are present
        rec_types, rec_parameters_names = get_rec_types_in_file(f)

    # get the n_gen
    header = HDF5Header.from_hdf5(input_file)
    n_gen = header.genvol.numberOfEvents

    # an identifier for what the part of the mc simulation this was
    # this way, events can later be unambiguously identified
    input_filename_string = os.path.basename(input_file)
    try:
        part_number = re.findall(r"\d+", input_filename_string)[
            -2
        ]  # second last because of .h5 - works only for officially named files
    except IndexError:
        part_number = 0

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
        event_info = blob["EventInfo"]
        event_id = event_info.event_id[0]
        run_id = event_info.run_id[0]

        # weights for neutrino analysis
        weight_w1 = event_info.weight_w1[0]
        weight_w2 = event_info.weight_w2[0]
        weight_w3 = event_info.weight_w3[0]

        is_cc = event_info.W2LIST_GSEAGEN_CC[0]
        bjorkeny = event_info.W2LIST_GSEAGEN_BY[0]

        # first, look for the particular neutrino index of the production
        p = 0  # for ORCA4 (and probably subsequent productions)

        primary_mc_track = blob["McTracks"][p]

        # some track mc truth info
        particle_type = primary_mc_track.pdgid  # sometimes type, sometimes pdgid
        energy = primary_mc_track.energy
        dir_x, dir_y, dir_z = (
            primary_mc_track.dir_x,
            primary_mc_track.dir_y,
            primary_mc_track.dir_z,
        )
        time_interaction = (
            primary_mc_track.time
        )  # actually always 0 for primary neutrino, measured in MC time
        vertex_pos_x, vertex_pos_y, vertex_pos_z = (
            primary_mc_track.pos_x,
            primary_mc_track.pos_y,
            primary_mc_track.pos_z,
        )

        # for (muon) NC interactions, the visible energy is different
        if np.abs(particle_type) == 14 and is_cc == 3:
            visible_energy = energy * bjorkeny
        else:
            visible_energy = energy

        # for tau CC it is not clear what the second interaction is; 1 for shower, 2 for track, 3 for nothing
        tau_topology = 3
        if np.abs(particle_type) == 16:
            if 13 in np.abs(blob["McTracks"].pdgid):
                tau_topology = 2
            else:
                tau_topology = 1

        # add also the nhits info
        n_hits = len(blob["Hits"])
        n_trig_hits = np.count_nonzero(blob["Hits"]["triggered"])

        track = {
            "event_id": event_id,
            "particle_type": particle_type,
            "energy": energy,
            "visible_energy": visible_energy,
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
            "n_trig_hits": n_trig_hits,
            "weight_w1": weight_w1,
            "weight_w2": weight_w2,
            "weight_w3": weight_w3,
            "n_gen": n_gen,
            "part_number": part_number,
            "tau_topology": tau_topology,
        }

        # get all the std reco info
        if has_std_reco:

            std_reco_info = get_std_reco(blob, rec_types, rec_parameters_names)

            track.update(std_reco_info)

        return track

    return mc_info_extr


# function used by Stefan to identify which muons leave how many mc hits in the (active) detector.
def get_mchits_per_muon(blob, inactive_du=None):

    """
    For each muon in McTracks, get the number of McHits.
    Parameters
    ----------
    blob
            The blob.
    inactive_du : int, optional
            McHits in this DU will not be counted.

    Returns
    -------
    np.array
            n_mchits, len = number of muons

    """
    ids = blob["McTracks"]["id"]

    # Origin of each mchit (as int) in the active line
    origin = blob["McHits"]["origin"]

    if inactive_du:
        # only hits in active line
        origin = origin[blob["McHits"]["du"] != inactive_du]

    # get how many mchits were produced per muon in the bundle
    origin_dict = dict(zip(*np.unique(origin, return_counts=True)))

    return np.array([origin_dict.get(i, 0) for i in ids])


def get_muon_mc_info_extr(input_file, inactive_du=None):

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

    if has_std_reco:
        # also check, which rec types are present
        rec_types, rec_parameters_names = get_rec_types_in_file(f)

    # no n_gen here, but needed for concatenation
    n_gen = 1

    # an identifier for what the part of the mc simulation this was
    # this way, events can later be unambiguously identified
    input_filename_string = os.path.basename(input_file)
    part_number = re.findall(r"\d+", input_filename_string)[
        -2
    ]  # second last because of .h5

    # dummy value for concatenation
    tau_topology = 3

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

        primary_mc_track = blob["McTracks"][p]

        particle_type = (
            primary_mc_track.pdgid
        )  # assumed that this is the same for all muons in a bundle, new: pdgid, old: type
        is_cc = 0  # set to 0
        bjorkeny = 0  # set to zero

        time_interaction = primary_mc_track.time  # same for all muons in a bundle

        # sum up the energy from all muons that have at least x mc hits
        n_hits_per_muon = get_mchits_per_muon(
            blob, inactive_du=inactive_du
        )  # DU1 in ORCA4 is in the detx but not powered

        # dont consider muons with less than 15 mc hits
        suficient_hits_mask = n_hits_per_muon >= 15
        energy = np.sum(blob["McTracks"][suficient_hits_mask].energy)
        # instead, assign them a small energy for consistency
        if energy == 0:
            energy = 40

        # also add a visible energy here, which is not further defined
        visible_energy = energy

        # all muons in a bundle are parallel, so just take dir of first muon
        dir_x, dir_y, dir_z = (
            primary_mc_track.dir_x,
            primary_mc_track.dir_y,
            primary_mc_track.dir_z,
        )

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
        n_trig_hits = np.count_nonzero(blob["Hits"]["triggered"])

        # this is only relevant for neutrinos, though dummy info is needed for the concatenation
        weight_w1 = 10
        weight_w2 = 10
        weight_w3 = 10

        track = {
            "event_id": event_id,
            "particle_type": particle_type,
            "energy": energy,
            "visible_energy": visible_energy,
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
            "n_trig_hits": n_trig_hits,
            "weight_w1": weight_w1,
            "weight_w2": weight_w2,
            "weight_w3": weight_w3,
            "n_gen": n_gen,
            "part_number": part_number,
            "tau_topology": tau_topology,
        }

        # get all the std reco info
        if has_std_reco:

            std_reco_info = get_std_reco(blob, rec_types, rec_parameters_names)

            track.update(std_reco_info)

        return track

    return mc_info_extr
