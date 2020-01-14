#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Code for computeing 2D/3D/4D histograms ("images") based on the event_hits hit pattern of the file_to_hits.py output"""

import km3pipe as kp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def get_time_parameters(event_hits, mode=('trigger_cluster', 'all'), t_start_margin=0.15, t_end_margin=0.15):
    """
    Gets the fundamental time parameters in one place for cutting a time residual.

    Later on, these parameters cut out a certain time span during an event specified by t_start and t_end.

    Parameters
    ----------
    event_hits : ndarray(ndim=2)
        2D array that contains the hits data for a certain event_id.
    mode : tuple(str, str)
        Type of time cut that is used. Currently available: timeslice_relative and trigger_cluster.
    t_start_margin, t_end_margin : float
        Used in timeslice_relative mode. Defines the start/end time of the selected timespan with t_mean -/+ t_start * t_diff.

    Returns
    -------
    t_start, t_end : float
        Absolute start and end time that will be used for the later timespan cut.
        Events in this timespan are accepted, others are rejected.

    """
    t = event_hits[:, 3:4]

    if mode[0] == 'trigger_cluster':
        triggered = event_hits[:, 4:5]
        t = t[triggered == 1]
        t_mean = np.mean(t, dtype=np.float64)

        if mode[1] == 'tight-0':
            # make a tighter cut, 9.5ns / bin with 100 bins, useful for ORCA 115l mupage events
            t_start = t_mean - 450  # trigger-cluster - 350ns
            t_end = t_mean + 500  # trigger-cluster + 850ns

        elif mode[1] == 'tight-1':
            # make a tighter cut, 12.5ns / bin with 60 bins
            t_start = t_mean - 250  # trigger-cluster - 350ns
            t_end = t_mean + 500  # trigger-cluster + 850ns

        elif mode[1] == 'tight-2':
            # make an even tighter cut, 5.8ns / bin with 60 bins
            t_start = t_mean - 150  # trigger-cluster - 150ns
            t_end = t_mean + 200  # trigger-cluster + 200ns

        else:
            assert mode[1] == 'all' # 'all' refers to the common time range of neutrino events in 115l ORCA
            # include nearly all mc_hits from muon-CC and elec-CC, 20ns / bin with 60 bins
            t_start = t_mean - 350 # trigger-cluster - 350ns
            t_end = t_mean + 850 # trigger-cluster + 850ns

    elif mode[0] == 'timeslice_relative':
        t_min = np.amin(t)
        t_max = np.amax(t)
        t_diff = t_max - t_min
        t_mean = t_min + 0.5 * t_diff

        t_start = t_mean - t_start_margin * t_diff
        t_end = t_mean + t_end_margin * t_diff

    elif mode[0] is None:
        t_start = np.amin(t)
        t_end = np.amax(t)

    else: raise ValueError('Time cut modes other than "first_triggered" or "timeslice_relative" are currently not supported.')

    return t_start, t_end


def compute_4d_to_2d_histograms(event_hits, event_track, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, timecut, do2d_plots, pdf_2d_plots):
    """
    Computes 2D numpy histogram 'images' from the 4D data and appends the 2D histograms to the all_4d_to_2d_hists list,
    [xy, xz, yz, xt, yt, zt].

    Parameters
    ----------
    event_hits : ndarray(ndim=2)
        2D array that contains the hits data for a certain event_id.
    event_track : ndarray(ndim=2)
        Contains the relevant mc_track info for the event in order to get a nice title for the pdf histos.
    x_bin_edges, y_bin_edges, z_bin_edges: ndarray(ndim=1)
        Bin edges for the X/Y/Z-direction.
    n_bins : tuple of int
        Contains the number of bins that should be used for each dimension.
    timecut : tuple(str, str/None)
        Tuple that defines what timecut should be used in hits_to_histograms.
    do2d_plots : bool
        If True, generate 2D matplotlib pdf histograms.
    pdf_2d_plots : mpl.backends.backend_pdf.PdfPages/None
        Either a mpl PdfPages instance or None.

    """
    x, y, z, t = event_hits[:, 0], event_hits[:, 1], event_hits[:, 2], event_hits[:, 3]

    # analyze time
    t_start, t_end = get_time_parameters(event_hits, mode=timecut)

    # create histograms for this event
    hist_xy = np.histogram2d(x, y, bins=(x_bin_edges, y_bin_edges))  # hist[0] = H, hist[1] = xedges, hist[2] = yedges
    hist_xz = np.histogram2d(x, z, bins=(x_bin_edges, z_bin_edges))
    hist_yz = np.histogram2d(y, z, bins=(y_bin_edges, z_bin_edges))

    hist_xt = np.histogram2d(x, t, bins=(x_bin_edges, n_bins[3]), range=((min(x_bin_edges), max(x_bin_edges)), (t_start, t_end)))
    hist_yt = np.histogram2d(y, t, bins=(y_bin_edges, n_bins[3]), range=((min(y_bin_edges), max(y_bin_edges)), (t_start, t_end)))
    hist_zt = np.histogram2d(z, t, bins=(z_bin_edges, n_bins[3]), range=((min(z_bin_edges), max(z_bin_edges)), (t_start, t_end)))

    if do2d_plots:
        hists = [hist_xy, hist_xz, hist_yz, hist_xt, hist_yt, hist_zt]
        convert_2d_numpy_hists_to_pdf_image(hists, t_start, t_end, pdf_2d_plots, event_track=event_track) # slow! takes about 1s per event

    hist_xy = kp.dataclasses.NDArray(hist_xy[0][np.newaxis, ...].astype(np.uint8), h5loc='x', title='XY_Event_Images')
    hist_xz = kp.dataclasses.NDArray(hist_xz[0][np.newaxis, ...].astype(np.uint8), h5loc='x', title='XZ_Event_Images')
    hist_yz = kp.dataclasses.NDArray(hist_yz[0][np.newaxis, ...].astype(np.uint8), h5loc='x', title='YZ_Event_Images')
    hist_xt = kp.dataclasses.NDArray(hist_xt[0][np.newaxis, ...].astype(np.uint8), h5loc='x', title='XT_Event_Images')
    hist_yt = kp.dataclasses.NDArray(hist_yt[0][np.newaxis, ...].astype(np.uint8), h5loc='x', title='YT_Event_Images')
    hist_zt = kp.dataclasses.NDArray(hist_zt[0][np.newaxis, ...].astype(np.uint8), h5loc='x', title='ZT_Event_Images')

    return hist_xy, hist_xz, hist_yz, hist_xt, hist_yt, hist_zt


def convert_2d_numpy_hists_to_pdf_image(hists, t_start, t_end, pdf_2d_plots, event_track=None):
    """
    Creates matplotlib 2D histos based on the numpy histogram2D objects and saves them to a pdf file.

    Parameters
    ----------
    hists : list(ndarray(ndim=2))
        Contains np.histogram2d objects of all projections [xy, xz, yz, xt, yt, zt].
    t_start, t_end : float
        Absolute start/end time of the timespan cut.
    pdf_2d_plots : mpl.backends.backend_pdf.PdfPages/None
        Either a mpl PdfPages instance or None.
    event_track : ndarray(ndim=2)
        Contains the relevant mc_track info for the event in order to get a nice title for the pdf histos.

    """

    fig = plt.figure(figsize=(10, 13))
    if event_track is not None:
        particle_type = {16: 'Tau', -16: 'Anti-Tau', 14: 'Muon', -14: 'Anti-Muon', 12: 'Electron', -12: 'Anti-Electron', 'isCC': ['NC', 'CC']}
        event_info = {'event_id': str(int(event_track.event_id[0])), 'energy': str(event_track.energy[0]),
                      'particle_type': particle_type[int(event_track.particle_type[0])],
                      'interaction_type': particle_type['isCC'][int(event_track.is_cc[0])]}
        title = event_info['particle_type'] + '-' + event_info['interaction_type'] + ', Event ID: ' + event_info['event_id'] + ', Energy: ' + event_info['energy'] + ' GeV'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.suptitle(title, usetex=False, horizontalalignment='center', size='xx-large', bbox=props)

    t_diff = t_end - t_start

    axes_xy = plt.subplot2grid((3, 2), (0, 0), title='XY - projection', xlabel='X Position [m]', ylabel='Y Position [m]', aspect='equal', xlim=(-175, 175), ylim=(-175, 175))
    axes_xz = plt.subplot2grid((3, 2), (0, 1), title='XZ - projection', xlabel='X Position [m]', ylabel='Z Position [m]', aspect='equal', xlim=(-175, 175), ylim=(-57.8, 292.2))
    axes_yz = plt.subplot2grid((3, 2), (1, 0), title='YZ - projection', xlabel='Y Position [m]', ylabel='Z Position [m]', aspect='equal', xlim=(-175, 175), ylim=(-57.8, 292.2))

    axes_xt = plt.subplot2grid((3, 2), (1, 1), title='XT - projection', xlabel='X Position [m]', ylabel='Time [ns]', aspect='auto',
                               xlim=(-175, 175), ylim=(t_start - 0.1*t_diff, t_end + 0.1*t_diff))
    axes_yt = plt.subplot2grid((3, 2), (2, 0), title='YT - projection', xlabel='Y Position [m]', ylabel='Time [ns]', aspect='auto',
                               xlim=(-175, 175), ylim=(t_start - 0.1*t_diff, t_end + 0.1*t_diff))
    axes_zt = plt.subplot2grid((3, 2), (2, 1), title='ZT - projection', xlabel='Z Position [m]', ylabel='Time [ns]', aspect='auto',
                               xlim=(-57.8, 292.2), ylim=(t_start - 0.1*t_diff, t_end + 0.1*t_diff))

    def fill_subplot(hist_ab, axes_ab):
        # Mask hist_ab
        h_ab_masked = np.ma.masked_where(hist_ab[0] == 0, hist_ab[0])

        a, b = np.meshgrid(hist_ab[1], hist_ab[2]) #2,1

        # Format in classical numpy convention: x along first dim (vertical), y along second dim (horizontal)
        # Need to take that into account in convert_2d_numpy_hists_to_pdf_image()
        # transpose to get typical cartesian convention: y along first dim (vertical), x along second dim (horizontal)
        plot_ab = axes_ab.pcolormesh(a, b, h_ab_masked.T)

        the_divider = make_axes_locatable(axes_ab)
        color_axis = the_divider.append_axes("right", size="5%", pad=0.1)

        # add color bar
        cbar_ab = plt.colorbar(plot_ab, cax=color_axis, ax=axes_ab)
        cbar_ab.ax.set_ylabel('Hits [#]')

    fill_subplot(hists[0], axes_xy)
    fill_subplot(hists[1], axes_xz)
    fill_subplot(hists[2], axes_yz)
    fill_subplot(hists[3], axes_xt)
    fill_subplot(hists[4], axes_yt)
    fill_subplot(hists[5], axes_zt)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    pdf_2d_plots.savefig(fig)
    plt.close()


def compute_4d_to_3d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, timecut):
    """
    Computes 3D numpy histogram 'images' from the 4D data and appends the 3D histograms to the all_4d_to_3d_hists list,
    [xyz, xyt, xzt, yzt, rzt].

    Careful: Currently, appending to all_4d_to_3d_hists takes quite a lot of memory (about 200MB for 3500 events).
    In the future, the list should be changed to a numpy ndarray.
    (Which unfortunately would make the code less readable, since an array is needed for each projection...)

    Parameters
    ----------
    event_hits : ndarray(ndim=2)
        2D array that contains the hits data for a certain event_id.
    x_bin_edges, y_bin_edges, z_bin_edges : ndarray(ndim=2)
        Bin edges for the X/Y/Z-direction.
    n_bins : tuple of int
        Contains the number of bins that should be used for each dimension.
    timecut : tuple(str, str/None)
        Tuple that defines what timecut should be used in hits_to_histograms.

    """

    x, y, z, t = event_hits[:, 0:1], event_hits[:, 1:2], event_hits[:, 2:3], event_hits[:, 3:4]

    t_start, t_end = get_time_parameters(event_hits, mode=timecut)

    hist_xyz = np.histogramdd(event_hits[:, 0:3], bins=(x_bin_edges, y_bin_edges, z_bin_edges))

    hist_xyt = np.histogramdd(np.concatenate([x, y, t], axis=1), bins=(x_bin_edges, y_bin_edges, n_bins[3]),
                              range=((min(x_bin_edges), max(x_bin_edges)), (min(y_bin_edges), max(y_bin_edges)), (t_start, t_end)))
    hist_xzt = np.histogramdd(np.concatenate([x, z, t], axis=1), bins=(x_bin_edges, z_bin_edges, n_bins[3]),
                              range=((min(x_bin_edges), max(x_bin_edges)), (min(z_bin_edges), max(z_bin_edges)), (t_start, t_end)))
    hist_yzt = np.histogramdd(event_hits[:, 1:4], bins=(y_bin_edges, z_bin_edges, n_bins[3]),
                              range=((min(y_bin_edges), max(y_bin_edges)), (min(z_bin_edges), max(z_bin_edges)), (t_start, t_end)))

    # add a rotation-symmetric 3d hist
    r = np.sqrt(x * x + y * y)
    rzt = np.concatenate([r, z, t], axis=1)
    hist_rzt = np.histogramdd(rzt, bins=(n_bins[0], n_bins[2], n_bins[3]), range=((np.amin(r), np.amax(r)), (np.amin(z), np.amax(z)), (t_start, t_end)))

    hist_xyz = kp.dataclasses.NDArray(hist_xyz[0][np.newaxis, ...].astype(np.uint8), h5loc='x', title='XYZ_Event_Images')
    hist_xyt = kp.dataclasses.NDArray(hist_xyt[0][np.newaxis, ...].astype(np.uint8), h5loc='x', title='XYT_Event_Images')
    hist_xzt = kp.dataclasses.NDArray(hist_xzt[0][np.newaxis, ...].astype(np.uint8), h5loc='x', title='XZT_Event_Images')
    hist_yzt = kp.dataclasses.NDArray(hist_yzt[0][np.newaxis, ...].astype(np.uint8), h5loc='x', title='YZT_Event_Images')
    hist_rzt = kp.dataclasses.NDArray(hist_rzt[0][np.newaxis, ...].astype(np.uint8), h5loc='x', title='RZT_Event_Images')

    return hist_xyz, hist_xyt, hist_xzt, hist_yzt, hist_rzt


def compute_4d_to_4d_histograms(event_hits, x_bin_edges, y_bin_edges, z_bin_edges, n_bins, timecut, do4d):
    """
    Computes 4D numpy histogram 'images' from the 4D data and appends the 4D histogram to the all_4d_to_4d_hists list,
    [xyzt / xyzc]

    Parameters
    ----------
    event_hits : ndarray(ndim=2)
        2D array that contains the hits data for a certain event_id.
    x_bin_edges, y_bin_edges, z_bin_edges : ndarray(ndim=2)
        Bin edges for the X/Y/Z-direction.
    n_bins : tuple of int
        Contains the number of bins that should be used for each dimension.
    timecut : tuple(str, str/None)
        Tuple that defines what timecut should be used in hits_to_histograms.
    do4d : tuple(bool, str)
        Tuple, where [1] declares what should be used as 4th dimension after xyz.
        Currently, only 'time' and 'channel_id' are available.

    """
    t_start, t_end = get_time_parameters(event_hits, mode=timecut)

    if do4d[1] == 'time':
        hist_4d = np.histogramdd(event_hits[:, 0:4], bins=(x_bin_edges, y_bin_edges, z_bin_edges, n_bins[3]),
                                   range=((min(x_bin_edges),max(x_bin_edges)),(min(y_bin_edges),max(y_bin_edges)),
                                          (min(z_bin_edges),max(z_bin_edges)),(t_start, t_end)))

    elif do4d[1] == 'channel_id':
        time = event_hits[:, 3]
        event_hits = event_hits[np.logical_and(time >= t_start, time <= t_end)]
        channel_id = event_hits[:, 5:6]
        hist_4d = np.histogramdd(np.concatenate([event_hits[:, 0:3], channel_id], axis=1), bins=(x_bin_edges, y_bin_edges, z_bin_edges, 31),
                                   range=((min(x_bin_edges),max(x_bin_edges)),(min(y_bin_edges),max(y_bin_edges)),
                                          (min(z_bin_edges),max(z_bin_edges)),(np.amin(channel_id), np.amax(channel_id))))

    else:
        raise ValueError('The parameter in do4d[1] ' + str(do4d[1]) + ' is not available. Currently, only time and channel_id are supported.')

    proj_name = 'XYZT' if not do4d[1] == 'channel_id' else 'XYZC'
    title = proj_name + '_Event_Images'
    hist_4d = kp.dataclasses.NDArray(hist_4d[0][np.newaxis, ...].astype(np.uint8), h5loc='x', title=title)

    return hist_4d


class HistogramMaker(kp.Module):
    """
    Class that takes a km3pipe blob which contains the information for one event and returns
    a blob with a hit array and a track array that contains all relevant information of the event.
    """
    def configure(self):
        """
        Sets up the input arguments of the EventDataExtractor class.
        """
        self.x_bin_edges = self.require('x_bin_edges')
        self.y_bin_edges = self.require('y_bin_edges')
        self.z_bin_edges = self.require('z_bin_edges')
        self.n_bins = self.require('n_bins')
        self.timecut = self.require('timecut')
        self.do2d = self.require('do2d')
        self.do2d_plots = self.require('do2d_plots')
        self.pdf_2d_plots = self.get('pdf_2d_plots')
        self.do3d = self.require('do3d')
        self.do4d = self.require('do4d')

        self.i = 0

    def process(self, blob):
        """
        Returns a blob (dict), which contains the event_hits array and the event_track array.

        Parameters
        ----------
        blob : dict
            Km3pipe blob which contains all the data from the input file.

        Returns
        -------
        blob : dict
            Dictionary that contains the event_hits array and the event_track array.

        """
        if self.do2d:
            blob['xy'], blob['xz'], blob['yz'], blob['xt'], blob['yt'], blob['zt'] = compute_4d_to_2d_histograms(
                blob['event_hits'], blob['event_track'], self.x_bin_edges, self.y_bin_edges, self.z_bin_edges,
                self.n_bins, self.timecut, self.do2d_plots[0], self.pdf_2d_plots)

            self.i += 1
            if self.pdf_2d_plots is not None and self.i >= self.do2d_plots[1]:
                self.pdf_2d_plots.close()
                raise StopIteration

        if self.do3d:
            blob['xyz'], blob['xyt'], blob['xzt'], blob['yzt'], blob['rzt'] = compute_4d_to_3d_histograms(
                blob['event_hits'], self.x_bin_edges, self.y_bin_edges, self.z_bin_edges,
                self.n_bins, self.timecut)

        if self.do4d[0]:
            proj_key = 'xyzt' if not self.do4d[1] == 'channel_id' else 'xyzc'

            blob[proj_key] = compute_4d_to_4d_histograms(blob['event_hits'], self.x_bin_edges, self.y_bin_edges, self.z_bin_edges,
                                        self.n_bins, self.timecut, self.do4d)

        return blob