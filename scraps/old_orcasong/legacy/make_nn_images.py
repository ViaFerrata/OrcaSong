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

import warnings
import os
import sys
#from memory_profiler import profile # for memory profiling, call with @profile; myfunc()
#import line_profiler # call with kernprof -l -v file.py args
import km3pipe as kp
import km3modules as km
import matplotlib as mpl
from docopt import docopt
mpl.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

from legacy.file_to_hits import EventDataExtractor
from legacy.hits_to_histograms import HistogramMaker
from legacy.io import load_config, check_user_input, make_output_dirs
from legacy.geo_binning import calculate_bin_edges
from legacy.utils import get_file_particle_type, EventSkipper


# TODO deprecated
warnings.warn("The original Orcasong is deprecated, and is no longer supported. "
              "Consider switching to the new orcasong.")


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
    # Load all parameters from the config # TODO put everything in a config class, this is horrible
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
    data_cuts['custom_skip_function'] = config['data_cut_custom_func'] if config['data_cut_custom_func'] != 'None' else None

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

    if do2d_plots[0] is True:
        pdf_2d_plots.close()


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










