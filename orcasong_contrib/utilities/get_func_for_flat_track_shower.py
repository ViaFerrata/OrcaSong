#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TODO
"""

import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import h5py
import natsort as ns


def get_h5_filepaths(dirpath):
    """
    Returns the filepaths of all .h5 files that are located in a specific directory.

    Parameters
    ----------
    dirpath: str
        Path of the directory where the .h5 files are located.

    Returns
    -------
    filepaths : list
        List with the full filepaths of all .h5 files in the dirpath folder.

    """
    filepaths = []
    for f in os.listdir(dirpath):
        if f.endswith('.h5'):
            filepaths.append(dirpath + '/' + f)

    filepaths = ns.natsorted(filepaths)  # TODO should not be necessary actually!
    return filepaths


def get_energies_for_fpaths(fpath_list, fpath_list_key_ic, cut_e_higher_than_3=False):
    """

    Parameters
    ----------
    fpath_list
    fpath_list_key_ic
    cut_e_higher_than_3

    Returns
    -------

    """

    energy_conc_arr = None
    for i, fpath in enumerate(fpath_list):
        if i % 100 == 0: print('Step ' + str(i))

        f = h5py.File(fpath, 'r')

        tracks = f['mc_tracks']
        tracks_neutr = tracks[tracks['bjorkeny'] != 0]

        assert f['event_info'].shape == tracks_neutr.shape
        energies = tracks_neutr['energy']

        if cut_e_higher_than_3 is True:
            energies = energies[energies <= 3]

        if energy_conc_arr is None:
            energy_conc_arr = energies
        else:
            energy_conc_arr = np.concatenate([energy_conc_arr, energies], axis=0)

        f.close()

    print('Total number of events for ' + fpath_list_key_ic + ' (without 3-5GeV from low_e prod): '
          + str(energy_conc_arr.shape[0]))
    print('Total number of files: ' + str(len(fpath_list)))

    return energy_conc_arr


def save_energies_for_ic(energies_for_ic):

    np.savez('./energies_for_ic.npz',
             muon_cc_3_100=energies_for_ic['muon_cc_3_100'], muon_cc_1_5=energies_for_ic['muon_cc_1_5'],
             elec_cc_3_100=energies_for_ic['elec_cc_3_100'], elec_cc_1_5=energies_for_ic['elec_cc_1_5'],
             elec_nc_3_100=energies_for_ic['elec_nc_3_100'], elec_nc_1_5=energies_for_ic['elec_nc_1_5'])


def load_energies_for_ic():

    data = np.load('./energies_for_ic.npz')

    energies_for_ic = dict()
    energies_for_ic['muon_cc_3_100'] = data['muon_cc_3_100']
    energies_for_ic['muon_cc_1_5'] = data['muon_cc_1_5']
    energies_for_ic['elec_cc_3_100'] = data['elec_cc_3_100']
    energies_for_ic['elec_cc_1_5'] = data['elec_cc_1_5']
    energies_for_ic['elec_nc_3_100'] = data['elec_nc_3_100']
    energies_for_ic['elec_nc_1_5'] = data['elec_nc_1_5']

    return energies_for_ic


def add_low_and_high_e_prods(energies_for_ic):
    """

    Parameters
    ----------
    energies_for_ic

    Returns
    -------

    """

    energies_for_ic['muon_cc'] = np.concatenate([energies_for_ic['muon_cc_3_100'], energies_for_ic['muon_cc_1_5']])
    energies_for_ic['elec_cc'] = np.concatenate([energies_for_ic['elec_cc_3_100'], energies_for_ic['elec_cc_1_5']])
    energies_for_ic['elec_nc'] = np.concatenate([energies_for_ic['elec_nc_3_100'], energies_for_ic['elec_nc_1_5']])
    energies_for_ic['elec_cc_and_nc'] = np.concatenate([energies_for_ic['elec_cc'], energies_for_ic['elec_nc']])


def plot_e_and_make_flat_func(energies_for_ic):
    """

    Parameters
    ----------
    energies_for_ic

    Returns
    -------

    """
    def make_plot_options_and_save(ax, pdfpages, ylabel):
        plt.xlabel('Energy [GeV]')
        plt.ylabel(ylabel)
        x_ticks_major = np.arange(0, 101, 10)
        ax.set_xticks(x_ticks_major)
        ax.grid(True)
        plt.tight_layout()
        pdfpages.savefig(fig)
        plt.cla()


    pdfpages = PdfPages('./e_hist_plots.pdf')
    fig, ax = plt.subplots()

    # plot
    hist_muon_cc = plt.hist(energies_for_ic['muon_cc'], bins=99)
    plt.title('Muon-CC 1-3 + 3-100 GeV for Run 1-2400')
    make_plot_options_and_save(ax, pdfpages, ylabel='Counts [#]')

    hist_shower = plt.hist(energies_for_ic['elec_cc_and_nc'], bins=99)
    plt.title('Shower (elec-CC + elec-NC) 1-3 + 3-100 GeV for 2x Run 1-1200')
    make_plot_options_and_save(ax, pdfpages, ylabel='Counts [#]')

    hist_elec_cc = plt.hist(energies_for_ic['elec_cc'], bins=99)
    plt.title('Elec-CC 1-3 + 3-100 GeV for Run 1-1200')
    make_plot_options_and_save(ax, pdfpages, ylabel='Counts [#]')

    hist_elec_nc = plt.hist(energies_for_ic['elec_nc'], bins=99)
    plt.title('Elec-NC 1-3 + 3-100 GeV for Run 1-1200')
    make_plot_options_and_save(ax, pdfpages, ylabel='Counts [#]')

    # We take 600 muon-CC files and 300 elec-cc and 300 elec_nc files for the split, reduce 1-3GeV bins by 1/2
    hist_shower[0][0] = hist_shower[0][0] / 2 # 1-2GeV
    hist_shower[0][1] = hist_shower[0][1] / 2 # 2-3GeV

    track_div_shower = np.divide(hist_muon_cc[0], hist_shower[0])
    print(hist_muon_cc[0])
    print(hist_shower[0])

    bins=hist_muon_cc[1] # doesnt matter which bins to use
    track_div_shower = np.append(track_div_shower, track_div_shower[-1])
    #track_div_shower = np.concatenate([track_div_shower, np.array(track_div_shower[-1])[:, np.newaxis]], axis=0) # fix for mpl
    print(bins)
    print(track_div_shower)
    ax.step(bins, track_div_shower, linestyle='-', where='post')
    plt.title('Ratio tracks divided by showers')
    make_plot_options_and_save(ax, pdfpages, ylabel='Fraction')

    pdfpages.close()


def main():
    dirs = {
            'muon_cc_3_100': '/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/3-100GeV/muon-CC',
            'muon_cc_1_5': '/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/1-5GeV/muon-CC',
            'elec_cc_3_100': '/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/3-100GeV/elec-CC',
            'elec_cc_1_5': '/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/1-5GeV/elec-CC',
            'elec_nc_3_100': '/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/3-100GeV/elec-NC',
            'elec_nc_1_5': '/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/1-5GeV/elec-NC'
            }

    if os.path.isfile('./energies_for_ic.npz') is True:
        energies_for_ic = load_energies_for_ic()

    else:
        fpaths = dict()
        for dir_ic_key in dirs:
            fpaths[dir_ic_key] = get_h5_filepaths(dirs[dir_ic_key])

        energies_for_ic = dict()
        for fpath_list_key_ic in fpaths:
            print('Getting energies for ' + fpath_list_key_ic)
            cut_flag = True if fpath_list_key_ic in ['muon_cc_1_5', 'elec_cc_1_5', 'elec_nc_1_5'] else False
            fpath_list = fpaths[fpath_list_key_ic]
            energies_for_ic[fpath_list_key_ic] = get_energies_for_fpaths(fpath_list, fpath_list_key_ic, cut_e_higher_than_3=cut_flag)

        save_energies_for_ic(energies_for_ic)

    add_low_and_high_e_prods(energies_for_ic)
    plot_e_and_make_flat_func(energies_for_ic)


if __name__ == '__main__':
    main()