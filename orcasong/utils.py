#!/usr/bin/env python
# coding=utf-8
# Filename: utils.py

"""
Utility code for OrcaSong.
"""

import numpy as np
import km3pipe as kp


class EventSkipper(kp.Module):
    """
    KM3Pipe module that skips events based on some data_cuts.
    """
    def configure(self):
        self.data_cuts = self.require('data_cuts')

    def process(self, blob):
        continue_bool = skip_event(blob['event_track'], self.data_cuts)
        if continue_bool:
            return
        else:
            return blob


def skip_event(event_track, data_cuts):
    """
    Function that checks if an event should be skipped, depending on the data_cuts input.

    Parameters
    ----------
    event_track : ndarray(ndim=1)
        1D array containing important MC information of the event, only the energy of the event (pos_2) is used here.
    data_cuts : dict
        Dictionary that contains information about any possible cuts that should be applied.
        Supports the following cuts: 'triggered', 'energy_lower_limit', 'energy_upper_limit', 'throw_away_prob'.

    Returns
    -------
    continue_bool : bool
        boolean flag to specify, if this event should be skipped or not.

    """
    continue_bool = False

    if data_cuts['energy_lower_limit'] is not None:
        continue_bool = event_track.energy[0] < data_cuts['energy_lower_limit'] # True if E < lower limit

    if data_cuts['energy_upper_limit'] is not None and continue_bool == False:
        continue_bool = event_track.energy[0] > data_cuts['energy_upper_limit'] # True if E > upper limit

    if data_cuts['throw_away_prob'] is not None and continue_bool == False:
        throw_away_prob = data_cuts['throw_away_prob']
        throw_away = np.random.choice([False, True], p=[1 - throw_away_prob, throw_away_prob])
        if throw_away: continue_bool = True

    if data_cuts['custom_skip_function'] is not None:
        continue_bool = use_custom_skip_function(data_cuts['custom_skip_function'], event_track)

    return continue_bool


def use_custom_skip_function(skip_function, event_track):
    """
    User defined, custom skip functions.

    Parameters
    ----------
    skip_function : str
        String that specifies, which custom skip function should be used.
    event_track : ndarray(ndim=1)
        Structured numpy array containing the McTracks info of this event.

    Returns
    -------
    continue_bool : bool
        Bool which specifies, if this event should be skipped or not.

    """
    if skip_function == 'ts_e_flat':
        # cf. orcanet_contrib/utilities/get_func_for_flat_track_shower.py

        particle_type = event_track.particle_type[0]
        if np.abs(particle_type) not in [12, 14]:
            continue_bool = False

        else:
            prob_for_e_bin_arr_path = '/home/woody/capn/mppi033h/Code/OrcaSong/scraps/utilities/arr_fract_for_e_bins.npy'
            # Fraction of tracks compared to showers (n_muon-cc / (n_e-cc + n_e-nc))
            arr_fract_for_e_bins = np.load(prob_for_e_bin_arr_path) # 2d arr, one row e_bins, second row prob
            e_bins = arr_fract_for_e_bins[0, :]
            fract_e = arr_fract_for_e_bins[1, :]

            e_evt = event_track.energy[0]
            idx_e = (np.abs(e_bins - e_evt)).argmin()

            fract = fract_e[idx_e]

            if np.abs(particle_type) == 14: # for muon neutrinos
                if fract <= 1:
                    continue_bool = False
                else:
                    evt_survive_prob = 1/float(fract)
                    continue_bool = np.random.choice([False, True], p=[evt_survive_prob, 1 - evt_survive_prob])

            else:  # for elec neutrinos
                assert np.abs(particle_type) == 12

                if fract >= 1:
                    continue_bool = False
                else:
                    evt_survive_prob = fract
                    continue_bool = np.random.choice([False, True], p=[evt_survive_prob, 1 - evt_survive_prob])

        return continue_bool

    else:
        raise ValueError('The custom skip function "' + str(skip_function) + '" is not known.')


def get_file_particle_type(fname):
    """
    Returns a string that specifies the type of the particles that are contained in the input file.

    Parameters
    ----------
    fname : str
        Filename (full path!) of the input file.

    Returns
    -------
    file_particle_type : str
        String that specifies the type of particles that are contained in the file: ['undefined', 'muon', 'neutrino'].

    """
    event_pump = kp.io.hdf5.HDF5Pump(filename=fname, verbose=False) # TODO suppress print of hdf5pump

    if 'McTracks' not in event_pump[0]:
        file_particle_type = 'undefined'
    else:
        particle_type = event_pump[0]['McTracks'].type

        # if mupage file: first mc_track is an empty neutrino track, second track is the first muon
        if particle_type[0] == 0 and np.abs(particle_type[1]) == 13:
            file_particle_type = 'muon'

        # the first mc_track is the primary neutrino if the input file is containing neutrino events
        elif np.abs(particle_type[0]) in [12, 14, 16]:
            file_particle_type = 'neutrino'
        else:
            raise ValueError('The type of the particles in the "McTracks" folder, <', str(particle_type), '> is not known.')

    event_pump.close_file()

    return file_particle_type


