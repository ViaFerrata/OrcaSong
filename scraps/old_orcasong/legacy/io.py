#!/usr/bin/env python
# coding=utf-8
# Filename: io.py

"""
IO Code for OrcaSong.
"""

import os
import errno
import toml
import warnings


def load_config(config_filepath):
    """
    Loads the config from a .toml file.

    Parameters
    ----------
    config_filepath : str
        Full filepath to a config file. An example can be found in orcasong/default_config.toml

    Returns
    -------
    config : dict
        Dictionary that contains all configuration options of the make_nn_images function.
        An explanation of the config parameters can be found in orcasong/default_config.toml.

    """
    config = toml.load(config_filepath)
    print('Loaded the config file from ' + os.path.abspath(config_filepath))

    return config


def check_user_input(fname, detx_filepath, config):
    """
    Sanity check of the user input.

    Parameters
    ----------
    fname : str
        Full filepath to the input .h5 file.
    detx_filepath : str
        Full filepath to the .detx geometry file that belongs to the fname.
    config : dict
        Dictionary that contains all configuration options of the make_nn_images function.
        An explanation of the config parameters can be found in orcasong/default_config.toml.

    """
    #---- Checks input types ----#

    # Check for options with a single, non-boolean element
    number_args = {'do2d_plots_n': int,  'data_cut_e_low': float, 'data_cut_e_high': float,
                   'data_cut_throw_away': float, 'prod_ident': int}

    for key in number_args:
        expected_arg_type = number_args[key]
        parsed_arg = config[key]

        if parsed_arg in [None, 'None']: # we don't want to check args when there has been no user input
            continue

        if type(parsed_arg) != expected_arg_type:
            try:
                map(expected_arg_type, parsed_arg)
            except ValueError:
                raise TypeError('The argument option ', key, ' only accepts ', str(expected_arg_type),
                                ' values as an input.')

    # Checks the n_bins tuple input
    for dim in config['n_bins']:
        if type(dim) != int:
            raise TypeError('The argument option n_bins only accepts integer values as an input!'
                            ' Your values have the type ' + str(type(dim)))

    # ---- Checks input types ----#

    # ---- Check other things ----#

    if not os.path.isfile(fname):
        raise IOError('The file -' + fname+ '- does not exist.')

    if not os.path.isfile(detx_filepath):
        raise IOError('The file -' + detx_filepath + '- does not exist.')

    if all(do_nd == False for do_nd in [config['do2d'], config['do3d'],config['do4d']]):
        raise ValueError('At least one of do2d, do3d or do4d options must be set to True.')

    if config['do2d'] == False and config['do2d_plots'] == True:
        raise ValueError('The 2D pdf images cannot be created if do2d=False!')

    if config['do2d_plots'] == True and config['do2d_plots_n'] > 100:
        warnings.warn('You declared do2d_pdf=(True, int) with int > 100. This will take more than two minutes.'
                      'Do you really want to create pdfs images for so many events?')


def make_output_dirs(output_dirpath, do2d, do3d, do4d):
    """
    Function that creates all output directories if they don't exist already.

    Parameters
    ----------
    output_dirpath : str
        Full path to the directory, where the orcasong output should be stored.
    do2d : bool
        Declares if 2D histograms, are to be created.
    do3d : bool
        Declares if 3D histograms are to be created.
    do4d : tuple(bool, str)
        Tuple that declares if 4D histograms should be created [0] and if yes, what should be used as the 4th dim after xyz.
        Currently, only 'time' and 'channel_id' are available.

    """
    if do2d:
        projections = ['xy', 'xz', 'yz', 'xt', 'yt', 'zt']
        for proj in projections:
            if not os.path.exists(output_dirpath + '/orcasong_output/4dTo2d/' + proj):
                try:
                    os.makedirs(output_dirpath + '/orcasong_output/4dTo2d/' + proj)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

    if do3d:
        projections = ['xyz', 'xyt', 'xzt', 'yzt', 'rzt']
        for proj in projections:
            if not os.path.exists(output_dirpath + '/orcasong_output/4dTo3d/' + proj):
                try:
                    os.makedirs(output_dirpath + '/orcasong_output/4dTo3d/' + proj)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise

    if do4d[0]:
        proj = 'xyzt' if not do4d[1] == 'channel_id' else 'xyzc'
        if not os.path.exists(output_dirpath + '/orcasong_output/4dTo4d/' + proj):
            try:
                os.makedirs(output_dirpath + '/orcasong_output/4dTo4d/' + proj)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise