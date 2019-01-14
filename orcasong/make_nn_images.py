#!/usr/bin/env python
# coding=utf-8
# Filename: make_nn_images.py
"""
Main OrcaSong code which takes raw simulated .h5 files and the corresponding .detx detector file as input in
order to generate 2D/3D/4D histograms ('images') that can be used for CNNs.

First argument: KM3NeT hdf5 simfile at JTE level.
Second argument: a .detx file that is associated with the hdf5 file.

The input file can be calibrated or not (e.g. contains pos_xyz of the hits).

Usage:
    make_nn_images.py SIMFILE DETXFILE CONFIGFILE
    make_nn_images.py (-h | --help)

Arguments:
    FILENAME    A KM3NeT hdf5 simfile at JTE level.

    DETXFILE    A .detx geometry file that is associated with the hdf5 file.

    CONFIGFILE  A .toml configuration file that contains all configuration options
                of this script. A default config can be found in the OrcaSong repo:
                orcasong/default_config.toml

Options:
    -h --help  Show this screen.

"""

__author__ = 'Michael Moser'
__license__ = 'AGPL'
__version__ = '1.0'
__email__ = 'michael.m.moser@fau.de'
__status__ = 'Prototype'

import os
import errno
import sys
import warnings
import toml
from docopt import docopt
#from memory_profiler import profile # for memory profiling, call with @profile; myfunc()
#import line_profiler # call with kernprof -l -v file.py args
import numpy as np
import km3pipe as kp
import km3modules as km
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

from orcasong.file_to_hits import EventDataExtractor
from orcasong.hits_to_histograms import HistogramMaker


def parse_input():
    """
    Parses and returns all necessary input options for the make_nn_images function.

    Returns
    -------
    fname : str
        Full filepath to the input .h5 file.
    detx_filepath : str
        Full filepath to the .detx geometry file that belongs to the fname.
    config_filepath : str
        Full filepath to a config file. An example can be found in orcasong/default_config.toml

    """
    args = docopt(__doc__)

    fname = args['SIMFILE']
    detx_filepath = args['DETXFILE']
    config_filepath = args['CONFIGFILE']

    return fname, detx_filepath, config_filepath


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


def calculate_bin_edges(n_bins, det_geo, detx_filepath, do4d):
    """
    Calculates the bin edges for the corresponding detector geometry (1 DOM/bin) based on the number of specified bins.

    Used later on for making the event "images" with the in the np.histogramdd funcs in hits_to_histograms.py.
    The bin edges are necessary in order to get the same bin size for each event regardless of the fact if all bins have a hit or not.

    Parameters
    ----------
    n_bins : tuple
        Contains the desired number of bins for each dimension, [n_bins_x, n_bins_y, n_bins_z].
    det_geo : str
        Declares what detector geometry should be used for the binning.
    detx_filepath : str
        Filepath of a .detx detector file which contains the geometry of the detector.
    do4d : tuple(boo, str)
        Tuple that declares if 4D histograms should be created [0] and if yes, what should be used as the 4th dim after xyz.

    Returns
    -------
    x_bin_edges, y_bin_edges, z_bin_edges : ndarray(ndim=1)
        The bin edges for the x,y,z direction.

    """
    # Loads a kp.Geometry object based on the filepath of a .detx file
    print("Reading detector geometry in order to calculate the detector dimensions from file " + detx_filepath)
    geo = kp.calib.Calibration(filename=detx_filepath)

    # derive maximum and minimum x,y,z coordinates of the geometry input [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    dom_position_values = geo.get_detector().dom_positions.values()
    dom_pos_max = np.amax([pos for pos in dom_position_values], axis=0)
    dom_pos_min = np.amin([pos for pos in dom_position_values], axis=0)
    geo_limits = dom_pos_min, dom_pos_max
    print('Detector dimensions [[xmin, ymin, zmin], [xmax, ymax, zmax]]: ' + str(geo_limits))

    if det_geo == 'Orca_115l_23m_h_9m_v' or det_geo == 'Orca_115l_23m_h_?m_v':
        x_bin_edges = np.linspace(geo_limits[0][0] - 9.95, geo_limits[1][0] + 9.95, num=n_bins[0] + 1) #try to get the lines in the bin center 9.95*2 = average x-separation of two lines
        y_bin_edges = np.linspace(geo_limits[0][1] - 9.75, geo_limits[1][1] + 9.75, num=n_bins[1] + 1) # Delta y = 19.483

        # Fitted offsets: x,y,factor: factor*(x+x_off), # Stefan's modifications:
        offset_x, offset_y, scale = [6.19, 0.064, 1.0128]
        x_bin_edges = (x_bin_edges + offset_x) * scale
        y_bin_edges = (y_bin_edges + offset_y) * scale

        if det_geo == 'Orca_115l_23m_h_?m_v':
            # ORCA denser detector study
            z_bin_edges = np.linspace(37.84 - 7.5, 292.84 + 7.5, num=n_bins[2] + 1)  # 15m vertical, 18 DOMs
            # z_bin_edges = np.linspace(37.84 - 6, 241.84 + 6, num=n_bins[2] + 1)  # 12m vertical, 18 DOMs
            # z_bin_edges = np.linspace(37.84 - 4.5, 190.84 + 4.5, num=n_bins[2] + 1)  # 9m vertical, 18 DOMs
            # z_bin_edges = np.linspace(37.84 - 3, 139.84 + 3, num=n_bins[2] + 1)  # 6m vertical, 18 DOMs
            # z_bin_edges = np.linspace(37.84 - 2.25, 114.34 + 2.25, num=n_bins[2] + 1)  # 4.5m vertical, 18 DOMs

        else:
            n_bins_z = n_bins[2] if do4d[1] != 'xzt-c' else n_bins[1] # n_bins = [xyz,t/c] or n_bins = [xzt,c]
            z_bin_edges = np.linspace(geo_limits[0][2] - 4.665, geo_limits[1][2] + 4.665, num=n_bins_z + 1)  # Delta z = 9.329

        # calculate_bin_edges_test(dom_positions, y_bin_edges, z_bin_edges) # test disabled by default. Activate it, if you change the offsets in x/y/z-bin-edges

    else:
        raise ValueError('The specified detector geometry "' + str(det_geo) + '" is not available.')

    return geo, x_bin_edges, y_bin_edges, z_bin_edges


def calculate_bin_edges_test(geo, y_bin_edges, z_bin_edges):
    """
    Tests, if the bins in one direction don't accidentally have more than 'one' OM.

    For the x-direction, an overlapping can not be avoided in an orthogonal space.
    For y and z though, it can!
    For y, every bin should contain the number of lines per y-direction * 18 for 18 doms per line.
    For z, every bin should contain 115 entries, since every z bin contains one storey of the 115 ORCA lines.
    Not 100% accurate, since only the dom positions are used and not the individual pmt positions for a dom.
    """
    dom_positions = np.stack(list(geo.get_detector().dom_positions.values()))
    dom_y = dom_positions[:, 1]
    dom_z = dom_positions[:, 2]
    hist_y = np.histogram(dom_y, bins=y_bin_edges)
    hist_z = np.histogram(dom_z, bins=z_bin_edges)

    print('----------------------------------------------------------------------------------------------')
    print('Y-axis: Bin content: ' + str(hist_y[0]))
    print('It should be:        ' + str(np.array(
        [4 * 18, 7 * 18, 9 * 18, 10 * 18, 9 * 18, 10 * 18, 10 * 18, 10 * 18, 11 * 18, 10 * 18, 9 * 18, 8 * 18, 8 * 18])))
    print('Y-axis: Bin edges: ' + str(hist_y[1]))
    print('..............................................................................................')
    print('Z-axis: Bin content: ' + str(hist_z[0]))
    print('It should have 115 entries everywhere')
    print('Z-axis: Bin edges: ' + str(hist_z[1]))
    print('----------------------------------------------------------------------------------------------')


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

    return continue_bool


def make_nn_images(fname, detx_filepath, config):
    """
    Main code with config parameters. Reads raw .hdf5 files and creates 2D/3D histogram projections that can be used
    for a CNN.

    Parameters
    ----------
    fname : str
        Filename (full path!) of the input file.
    detx_filepath : str
        String with the full filepath to the corresponding .detx file of the input file.
        Used for the binning and for the hits calibration if the input file is not calibrated yet
        (e.g. hits do not contain pos_x/y/z, time, ...).
    config : dict
        Dictionary that contains all configuration options of the make_nn_images function.
        An explanation of the config parameters can be found in orcasong/default_config.toml.

    """
    # Load all parameters from the config
    output_dirpath = config['output_dirpath']
    chunksize, complib, complevel = config['chunksize'], config['complib'], config['complevel']
    flush_freq = config['flush_freq']
    n_bins = tuple(config['n_bins'])
    timecut = (config['timecut_mode'], config['timecut_timespan'])
    do_mc_hits = config['do_mc_hits']
    det_geo = config['det_geo']
    do2d = config['do2d']
    do2d_plots = (config['do2d_plots'], config['do2d_plots_n'])
    do3d = config['do3d']
    do4d = (config['do4d'], config['do4d_mode'])
    prod_ident = config['prod_ident'] if config['prod_ident'] != 'None' else None
    data_cuts = dict()
    data_cuts['triggered'] = config['data_cut_triggered']
    data_cuts['energy_lower_limit'] = config['data_cut_e_low'] if config['data_cut_e_low'] != 'None' else None
    data_cuts['energy_upper_limit'] = config['data_cut_e_high'] if config['data_cut_e_high'] != 'None' else None
    data_cuts['throw_away_prob'] = config['data_cut_throw_away'] if config['data_cut_throw_away'] != 'None' else None

    make_output_dirs(output_dirpath, do2d, do3d, do4d)

    filename = os.path.basename(os.path.splitext(fname)[0])
    filename_output = filename.replace('.','_')

    # set random km3pipe (=numpy) seed
    print('Setting a Global Random State with the seed < 42 >.')
    km.GlobalRandomState(seed=42)

    geo, x_bin_edges, y_bin_edges, z_bin_edges = calculate_bin_edges(n_bins, det_geo, detx_filepath, do4d)
    pdf_2d_plots = PdfPages(output_dirpath + '/orcasong_output/4dTo2d/' + filename_output + '_plots.pdf') if do2d_plots[0] is True else None

    file_particle_type = get_file_particle_type(fname)

    print('Generating histograms from the hits for files based on ' + fname)

    # Initialize OrcaSong Event Pipeline

    pipe = kp.Pipeline() # add timeit=True argument for profiling
    pipe.attach(km.common.StatusBar, every=200)
    pipe.attach(km.common.MemoryObserver, every=400)
    pipe.attach(kp.io.hdf5.HDF5Pump, filename=fname)
    pipe.attach(km.common.Keep, keys=['EventInfo', 'Header', 'RawHeader', 'McTracks', 'Hits', 'McHits'])
    pipe.attach(EventDataExtractor,
                file_particle_type=file_particle_type, geo=geo, do_mc_hits=do_mc_hits,
                data_cuts=data_cuts, do4d=do4d, prod_ident=prod_ident)
    pipe.attach(km.common.Keep, keys=['event_hits', 'event_track'])
    pipe.attach(EventSkipper, data_cuts=data_cuts)
    pipe.attach(HistogramMaker,
                x_bin_edges=x_bin_edges, y_bin_edges=y_bin_edges, z_bin_edges=z_bin_edges,
                n_bins=n_bins, timecut=timecut, do2d=do2d, do2d_plots=do2d_plots, pdf_2d_plots=pdf_2d_plots,
                do3d=do3d, do4d=do4d)
    pipe.attach(km.common.Delete, keys=['event_hits'])

    if do2d:
        for proj in ['xy', 'xz', 'yz', 'xt', 'yt', 'zt']:
            savestr = output_dirpath + '/orcasong_output/4dTo2d/' + proj + '/' + filename_output + '_' + proj + '.h5'
            pipe.attach(kp.io.HDF5Sink, filename=savestr, blob_keys=[proj, 'event_track'], complib=complib,
                        complevel=complevel, chunksize=chunksize, flush_frequency=flush_freq)


    if do3d:
        for proj in ['xyz', 'xyt', 'xzt', 'yzt', 'rzt']:
            savestr = output_dirpath + '/orcasong_output/4dTo3d/' + proj + '/' + filename_output + '_' + proj + '.h5'
            pipe.attach(kp.io.HDF5Sink, filename=savestr, blob_keys=[proj, 'event_track'], complib=complib,
                        complevel=complevel, chunksize=chunksize, flush_frequency=flush_freq)

    if do4d[0]:
        proj = 'xyzt' if not do4d[1] == 'channel_id' else 'xyzc'
        savestr = output_dirpath + '/orcasong_output/4dTo4d/' + proj + '/' + filename_output + '_' + proj + '.h5'
        pipe.attach(kp.io.HDF5Sink, filename=savestr, blob_keys=[proj, 'event_track'], complib=complib,
                    complevel=complevel, chunksize=chunksize, flush_frequency=flush_freq)

    # Execute Pipeline
    pipe.drain()


def main():
    """
    Parses the input to the main make_nn_images function.
    """
    fname, detx_filepath, config_filepath = parse_input()
    config = load_config(config_filepath)
    check_user_input(fname, detx_filepath, config)
    make_nn_images(fname, detx_filepath, config)


if __name__ == '__main__':
    main()










