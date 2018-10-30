import h5py
import numpy as np

path = '/home/woody/capn/mppi033h/Data/ORCA_JTE_NEMOWATER/ip_images_1-100GeV/4dTo4d/time_-250+500_w_gf_60b'
# JTE_KM3Sim_gseagen_muon-CC_1-5GeV-9_2E5-1bin-1_0gspec_ORCA115_9m_2016_98_xyzt.h5
ptypes = {'muon-CC': 'JTE_KM3Sim_gseagen_muon-CC_1-5GeV-9_2E5-1bin-1_0gspec_ORCA115_9m_2016_',
          'elec-CC': 'JTE_KM3Sim_gseagen_elec-CC_1-5GeV-2_7E5-1bin-1_0gspec_ORCA115_9m_2016_'}

event_id, run_id = None, None
for ptype in ptypes.keys():
    for i in range(601):
        if i % 100 == 0:
            print(i)
        if i == 0: continue

        f = h5py.File(path + '/' + ptypes[ptype] + str(i) + '_xyzt.h5', 'r')
        event_id_f = f['y'][:, 0]
        run_id_f = f['y'][:, 9]

        if event_id is None:
            event_id = event_id_f
            run_id = run_id_f
        else:
            event_id = np.concatenate([event_id, event_id_f], axis=0)
            run_id = np.concatenate([run_id, run_id_f], axis=0)

        f.close()

    ax = np.newaxis
    arr = np.concatenate([run_id[:, ax], event_id[:, ax]], axis=1)
    np.save('/home/woody/capn/mppi033h/Code/OrcaSong/utilities/low_e_prod_surviving_evts_' + ptype + '.npy', arr)
    event_id, run_id = None, None




