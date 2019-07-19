#!/usr/bin/env python
# coding=utf-8
# Filename: geo_binning.py

"""
OrcaSong code that handles the spatial binning of the output images.
"""

import numpy as np
import km3pipe as kp


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