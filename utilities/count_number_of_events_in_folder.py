
import os
import numpy as np
import h5py


def count_number_of_events_in_folder(dirpath):

    fpath_list = []
    for file in os.listdir(dirpath):
        if file.endswith('.h5'):
            fpath = os.path.join(dirpath, file)
            fpath_list.append(fpath)

    n_total = 0
    for fpath in fpath_list:
        f = h5py.File(fpath, 'r')
        n_total += f['event_info'].shape[0]
        f.close()

    return n_total



def main():
    dirpath_rn = '/home/saturn/capn/mppi033h/Data/raw_data/random_noise'
    rn_n_total = count_number_of_events_in_folder(dirpath_rn)
    print('--------------------------------------------------------')
    print('Total number of random_noise events: ' + str(rn_n_total))
    print('--------------------------------------------------------')

    dirpath_mupage = '/home/saturn/capn/mppi033h/Data/raw_data/mupage'
    mupage_n_total = count_number_of_events_in_folder(dirpath_mupage)
    print('--------------------------------------------------------')
    print('Total number of mupage events: ' + str(mupage_n_total))
    print('--------------------------------------------------------')

    dirpath_muon_cc_low_e = '/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/1-5GeV/muon-CC'
    dirpath_muon_cc_high_e = '/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/3-100GeV/muon-CC'
    muon_cc_low_e_n_total = count_number_of_events_in_folder(dirpath_muon_cc_low_e)
    muon_cc_high_e_n_total = count_number_of_events_in_folder(dirpath_muon_cc_high_e)
    print('--------------------------------------------------------')
    print('Total number of muon-CC 1-5GeV events: ' + str(muon_cc_low_e_n_total))
    print('Total number of muon-CC 3-100GeV events: ' + str(muon_cc_high_e_n_total))
    print('--------------------------------------------------------')

    dirpath_elec_cc_low_e = '/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/1-5GeV/elec-CC'
    dirpath_elec_cc_high_e = '/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/3-100GeV/elec-CC'
    elec_cc_low_e_n_total = count_number_of_events_in_folder(dirpath_elec_cc_low_e)
    elec_cc_high_e_n_total = count_number_of_events_in_folder(dirpath_elec_cc_high_e)
    print('--------------------------------------------------------')
    print('Total number of elec-CC 1-5GeV events: ' + str(elec_cc_low_e_n_total))
    print('Total number of elec-CC 3-100GeV events: ' + str(elec_cc_high_e_n_total))
    print('--------------------------------------------------------')

    dirpath_elec_nc_low_e = '/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/1-5GeV/elec-NC'
    dirpath_elec_nc_high_e = '/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/3-100GeV/elec-NC'
    elec_nc_low_e_n_total = count_number_of_events_in_folder(dirpath_elec_nc_low_e)
    elec_nc_high_e_n_total = count_number_of_events_in_folder(dirpath_elec_nc_high_e)
    print('--------------------------------------------------------')
    print('Total number of elec-NC 1-5GeV events: ' + str(elec_nc_low_e_n_total))
    print('Total number of elec-NC 3-100GeV events: ' + str(elec_nc_high_e_n_total))
    print('--------------------------------------------------------')

    dirpath_tau_cc = '/home/saturn/capn/mppi033h/Data/raw_data/ORCA_JTE_NEMOWATER/calibrated/with_jte_times/3-100GeV/tau-CC'
    tau_cc_n_total = count_number_of_events_in_folder(dirpath_tau_cc)
    print('--------------------------------------------------------')
    print('Total number of tau-CC 3-100GeV events: ' + str(tau_cc_n_total))
    print('--------------------------------------------------------')


    n_total_neutr = muon_cc_low_e_n_total + muon_cc_high_e_n_total + elec_cc_low_e_n_total + elec_cc_high_e_n_total\
                    + elec_nc_low_e_n_total + elec_nc_high_e_n_total + tau_cc_n_total
    print('--------------------------------------------------------')
    print('Total number of neutrino events: ' + str(n_total_neutr))
    print('--------------------------------------------------------')


if __name__ == '__main__':
    main()
