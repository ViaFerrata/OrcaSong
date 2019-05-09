#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import km3pipe as kp
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def get_time_array(fname, savestr = ''):
    event_pump = kp.io.hdf5.HDF5Pump(filename=fname)

    time_mean_trigg_all_events = None
    mean_triggered_time_list = []
    for i, event_blob in enumerate(event_pump):
        if i % 200 == 0:
            print('Event No. ' + str(i))

        time = event_blob['Hits'].time
        triggered = event_blob['Hits'].triggered

        mean_triggered_time = np.mean(time[triggered==1])
        mean_triggered_time_list.append(mean_triggered_time)
        time_minus_mean = np.subtract(time, mean_triggered_time)

        ax = np.newaxis
        #time_trigg = np.concatenate([time[:, ax], triggered[:, ax]], axis=1)
        time_minus_mean_trigg = np.concatenate([time_minus_mean[:, ax], triggered[:, ax]], axis=1)

        if i==0:
            time_mean_trigg_all_events = time_minus_mean_trigg
        else:
            time_mean_trigg_all_events = np.concatenate([time_mean_trigg_all_events, time_minus_mean_trigg], axis=0)

        # plotting
        plotting = False
        if plotting:
            time_only_trigg = time[triggered == 1]
            plt.hist(time_only_trigg, bins=50)
            plt.savefig('hist_time_trigg_only.png')

            plt.cla()

            plt.hist(time, bins=50)
            plt.savefig('hist_time.png')

            break

    np.save(savestr + '_time_mean_trigg_all_events.npy', time_mean_trigg_all_events)

    mean_triggered_time_arr = np.array(mean_triggered_time_list, dtype=np.float64)
    np.save(savestr + '_mean_triggered_time_arr_each_event.npy', mean_triggered_time_arr)

    event_pump.close_file()


def get_time_array_mc_hits(fname, savestr='', mean=('', None)):
    event_pump = kp.io.hdf5.HDF5Pump(filename=fname)

    mc_hits_time_mean_all_events = None
    for i, event_blob in enumerate(event_pump):
        if i % 200 == 0:
            print('Event No. ' + str(i))

        time = event_blob['McHits'].time

        if mean[0] == 'trigger':
            mean_triggered_time_arr = mean[1]
            mean_triggered_time_event = mean_triggered_time_arr[i]
            time_minus_mean = np.subtract(time, mean_triggered_time_event)

        else:
            mean_time = np.mean(time)
            time_minus_mean = np.subtract(time, mean_time)

        if i==0:
            mc_hits_time_mean_all_events = time_minus_mean
        else:
            mc_hits_time_mean_all_events = np.concatenate([mc_hits_time_mean_all_events, time_minus_mean], axis=0)

        # plotting
        plotting = False
        if plotting:
            plt.hist(time, bins=50)
            plt.savefig('hist_mc_hits_time.png')
            break

    if mean[0] == 'trigger':
        np.save(savestr + '_mc_hits_time_meantriggered_all_events.npy', mc_hits_time_mean_all_events) # mean centering with triggered hits
    else:
        np.save(savestr + '_mc_hits_time_mean_all_events.npy', mc_hits_time_mean_all_events) # mean centering with mc hits

    event_pump.close_file()


def plot(savestr=''):

    #triggered = time_trigg_all_events[:, 1]
    #time_only_trigg = time_trigg_all_events[triggered == 1]

    time_mean_trigg_all_events = np.load(savestr + '_time_mean_trigg_all_events.npy')

    # plotting
    plt.hist(time_mean_trigg_all_events[:, 0], bins=100)
    plt.grid(True, zorder=0, linestyle='dotted')
    plt.savefig('./plots/' + savestr + '/' + savestr + '_hist_time_minus_mean_all_events.png')
    plt.cla()

    plt.hist(time_mean_trigg_all_events[:, 0], bins=100, range=(-1000, 1000))
    y_lim = plt.ylim() # current one
    plt.ylim((0.6*y_lim[1], y_lim[1]))
    plt.grid(True, zorder=0, linestyle='dotted')
    plt.savefig('./plots/' + savestr + '/' + savestr + '_hist_time_minus_mean_all_events_zoomy_-1000_1000.png')
    plt.cla()

    plt.hist(time_mean_trigg_all_events[:, 0], bins=100, range=(-500, 500))
    plt.grid(True, zorder=0, linestyle='dotted')
    plt.savefig('./plots/' + savestr + '/' + savestr + '_hist_time_minus_mean_all_events_zoomy_-500_500.png')
    plt.cla()

    triggered = time_mean_trigg_all_events[:, 1]
    time_minus_mean_only_trigg_all_events = time_mean_trigg_all_events[triggered==1]

    plt.hist(time_minus_mean_only_trigg_all_events[:, 0], bins=100)
    plt.grid(True, zorder=0, linestyle='dotted')
    plt.savefig('./plots/' + savestr + '/' + savestr + '_hist_time_minus_mean_all_events_trigg_only_.png')


def plot_mc_hits(savestr='', mean='', vert_lines={}, title_ptype=''):
    mc_hits_time_mean_all_events = np.load(savestr + '_mc_hits_time_mean' + mean + '_all_events.npy')

    # plotting

    plt.hist(mc_hits_time_mean_all_events, bins=100)
    plt.grid(True, zorder=0, linestyle='dotted')
    plt.savefig('./plots/' + savestr + '/' + savestr + '_hist_mc_hits_time_minus_mean' + mean + '_all_events.png')

    plt.cla()

    # plt.hist(mc_hits_time_mean_all_events, bins=100, range=(-2500, 2500))
    # plt.savefig(savestr + '_hist_mc_hits_time_minus_mean_all_events_zoom_-2500_2500.png')
    # plt.cla()

    plt.hist(mc_hits_time_mean_all_events, bins=100, range=(-1000, 1500))
    plt.grid(True, zorder=0, linestyle='dotted')

    plt.xlabel('MC-Hit time minus mean time of all triggered hits [ns]')
    plt.ylabel('Number of hits [#]')
    title = plt.title('MC-Hit time pattern for ' + title_ptype + ' events')
    title.set_position([.5, 1.04])

    for tcut_key in vert_lines:
        plt.axvline(x=vert_lines[tcut_key]['xrange'][0], color=vert_lines[tcut_key]['color'], linestyle='--', label=tcut_key)
        plt.axvline(x=vert_lines[tcut_key]['xrange'][1], color=vert_lines[tcut_key]['color'], linestyle='--')

    plt.legend(prop={'size': 12})
    plt.tight_layout()

    plt.savefig('./plots/' + savestr + '/' + savestr + '_hist_mc_hits_time_minus_mean' + mean + '_all_events_zoom_-1000_1500.png')
    plt.savefig('./plots/' + savestr + '/' + savestr + '_hist_mc_hits_time_minus_mean' + mean + '_all_events_zoom_-1000_1500.pdf')
    plt.cla()


if __name__ == '__main__':
    ptypes = {'mupage': {'vert_lines': {'Timecut 0': {'color': 'black', 'xrange': (-450, 500)}}, 'title_ptype': 'atmospheric muon'},
              'muon-CC': {'vert_lines': {'Timecut 1': {'color': 'black', 'xrange': (-250, 500)},
                                         'Timecut 2': {'color': 'firebrick', 'xrange': (-150, 200)}},
                          'title_ptype': r'$\nu_{\mu}-CC$'},
              'elec-CC': {'vert_lines': {'Timecut 1': {'color': 'black', 'xrange': (-250, 500)},
                                         'Timecut 2': {'color': 'firebrick', 'xrange': (-150, 200)}},
                          'title_ptype': r'$\nu_{e}-CC$'},
              'tau-CC': {'vert_lines': {'Timecut 1': {'color': 'black', 'xrange': (-250, 500)},
                                         'Timecut 2': {'color': 'firebrick', 'xrange': (-150, 200)}},
                         'title_ptype': r'$\nu_{\tau}-CC$'},
              }

    for ptype in ptypes:

        if ptype == 'mupage':
            path = '/home/saturn/capn/mppi033h/Data/raw_data/mupage/'
        elif ptype == 'random_noise':
            path = '/home/saturn/capn/mppi033h/Data/raw_data/random_noise/'
        else:
            path = '/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/3-100GeV/' + ptype + '/'

        filenames = {'muon-CC': 'JTE.KM3Sim.gseagen.muon-CC.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016.99.h5',
                     'elec-CC': 'JTE.KM3Sim.gseagen.elec-CC.3-100GeV-1.1E6-1bin-3.0gspec.ORCA115_9m_2016.99.h5',
                     'tau-CC': 'JTE.KM3Sim.gseagen.tau-CC.3.4-100GeV-2.0E8-1bin-3.0gspec.ORCA115_9m_2016.99.h5',
                     'mupage': 'JTE.ph.ph.mupage.ph.ph.ph.ORCA115_9m_2016.99.h5',
                     'random_noise': ''}
        filename_input = path + filenames[ptype]

        # centered with trigg hits mean
        get_time_array(filename_input, savestr=ptype)
        plot(savestr=ptype)

        # centered with mc_hits mean
        get_time_array_mc_hits(filename_input, savestr=ptype)
        plot_mc_hits(savestr=ptype, vert_lines=ptypes[ptype]['vert_lines'], title_ptype=ptypes[ptype]['title_ptype'])

        # centered mc_hits with trigg hits mean for each event
        mean_triggered_time = np.load(ptype + '_mean_triggered_time_arr_each_event.npy')
        get_time_array_mc_hits(filename_input, savestr=ptype, mean=('trigger', mean_triggered_time))
        plot_mc_hits(ptype, mean='triggered', vert_lines=ptypes[ptype]['vert_lines'], title_ptype=ptypes[ptype]['title_ptype'])


