#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Code for testing the readout speed of orcasong .hdf5 files."""

import numpy as np
import h5py
import timeit
import cProfile

def generate_batches_from_hdf5_file():
    # 4d
    #filepath = 'JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_9_xyzt_no_compression_chunked.h5' # 4D, (11x13x18x50)), no compression. chunksize=32 --> 1011 ms
    #filepath = 'JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_9_xyzt_lzf.h5' # 4D, (11x13x18x50), lzf --> 2194 ms
    #filepath = 'JTE_KM3Sim_gseagen_muon-CC_3-100GeV-9_1E7-1bin-3_0gspec_ORCA115_9m_2016_9_xyzt_gzip_1.h5' # 4D, (11x13x18x50), gzip, compression_opts=1 --> 1655 ms

    # With new km3pipe structure
    filepath = '/home/woody/capn/mppi033h/orcasong_output/4dTo4d/xyzc/JTE_ph_ph_mupage_ph_ph_ph_ORCA115_9m_2016_9_xyzc.h5'

    print('Testing generator on file ' + filepath)
    batchsize = 32
    dimensions = (batchsize, 11, 13, 18, 31)  # 4D

    f = h5py.File(filepath, "r")
    filesize = len(f['y'])
    print(filesize)

    n_entries = 0
    while n_entries < (filesize - batchsize):
        xs = f['x'][n_entries : n_entries + batchsize]
        xs = np.reshape(xs, dimensions).astype(np.float32)

        y_values = f['y'][n_entries:n_entries+batchsize]
        ys = y_values[['run_id', 'event_id']]

        n_entries += batchsize
        yield (xs, ys)
    f.close()


number = 20
#t = timeit.timeit(generate_batches_from_hdf5_file, number = number)
#t = timeit.Timer(stmt="list(generate_batches_from_hdf5_file())", setup="from __main__ import generate_batches_from_hdf5_file")
#print t.timeit(number) / number
#print str(number) + 'loops, on average ' + str(t.timeit(number) / number *1000) + 'ms'

pr = cProfile.Profile()
pr.enable()

t = timeit.Timer(stmt="list(generate_batches_from_hdf5_file())", setup="from __main__ import generate_batches_from_hdf5_file")
print(str(number) + 'loops, on average ' + str(t.timeit(number) / number *1000) + 'ms')

pr.disable()

pr.print_stats(sort='time')