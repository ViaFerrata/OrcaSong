import os
import time
import datetime
import argparse
import numpy as np
import h5py

from orcasong.tools.postproc import get_filepath_output, copy_used_files
from orcasong.tools.concatenate import copy_attrs

# neu max_ram = 1e9: 2:41 (161s)
# neu max_ram = 6e9: 0:25 (25s)
# alt:  3:38 (218s)


def shuffle_v2(
        input_file,
        datasets=("x", "y"),
        output_file=None,
        max_ram=1e9,
        seed=42):
    """
    Shuffle datasets in a h5file that have the same length.

    Parameters
    ----------
    input_file : str
        Path of the file that will be shuffle.
    datasets : tuple
        Which datasets to include in output.
    output_file : str, optional
        If given, this will be the name of the output file.
        Otherwise, a name is auto generated.
    max_ram : int
        Available ram.
    seed : int
        Sets a fixed random seed for the shuffling.

    Returns
    -------
    output_file : str
        Path to the output file.

    """
    start_time = time.time()
    if output_file is None:
        output_file = get_filepath_output(input_file, shuffle=True)
    if os.path.exists(output_file):
        raise FileExistsError(output_file)

    with h5py.File(input_file, "r") as f_in:
        dset_infos, n_lines = get_dset_infos(f_in, datasets, max_ram)
        print(f"Shuffling datasets {datasets} with {n_lines} lines each")
        np.random.seed(seed)
        indices = np.arange(n_lines)
        np.random.shuffle(indices)

        with h5py.File(output_file, "x") as f_out:
            for dset_info in dset_infos:
                print("Creating dataset", dset_info["name"])
                make_dset(f_out, dset_info, indices)
                print("Done!")

    copy_used_files(input_file, output_file)
    copy_attrs(input_file, output_file)

    print(f"Elapsed time: "
          f"{datetime.timedelta(seconds=int(time.time() - start_time))}")
    return output_file


def get_dset_infos(f, datasets, max_ram):
    """ Check datasets and retrieve relevant infos for each. """
    dset_infos = []
    n_lines = None
    for i, name in enumerate(datasets):
        dset = f[name]
        if i == 0:
            n_lines = len(dset)
        else:
            if len(dset) != n_lines:
                raise ValueError(f"dataset {name} has different length! "
                                 f"{len(dset)} vs {n_lines}")
        # TODO in h5py 3.X, use .nbytes to get uncompressed size
        bytes_per_line = np.asarray(dset[0]).nbytes
        lines_per_batch = int(max_ram / bytes_per_line)
        n_batches = int(np.ceil(n_lines / lines_per_batch))
        dset_infos.append({
            "name": name,
            "dset": dset,
            "bytes_per_line": bytes_per_line,
            "lines_per_batch": lines_per_batch,
            "n_batches": n_batches,
        })
    return dset_infos, n_lines


def get_indices(n_lines, chunksize, chunks_per_batch):
    indices = np.arange(n_lines)
    chunk_starts = indices[::chunksize]

    np.random.shuffle(chunk_starts)
    for batch_no in range(chunks_per_batch)


def make_dset(f_out, dset_info, indices):
    """ Create a shuffled dataset in the output file. """
    for batch_index in range(dset_info["n_batches"]):
        print(f"Processing batch {batch_index+1}/{dset_info['n_batches']}")

        slc = slice(
            batch_index * dset_info["lines_per_batch"],
            (batch_index+1) * dset_info["lines_per_batch"],
        )
        to_read = indices[slc]
        # reading has to be done with linearly increasing index,
        #  so sort -> read -> undo sorting
        sort_ix = np.argsort(to_read)
        unsort_ix = np.argsort(sort_ix)
        data = dset_info["dset"][to_read[sort_ix]][unsort_ix]

        if batch_index == 0:
            in_dset = dset_info["dset"]
            out_dset = f_out.create_dataset(
                dset_info["name"],
                data=data,
                maxshape=in_dset.shape,
                chunks=in_dset.chunks,
                compression=in_dset.compression,
                compression_opts=in_dset.compression_opts,
                shuffle=in_dset.shuffle,
            )
            out_dset.resize(len(in_dset), axis=0)
        else:
            f_out[dset_info["name"]][slc] = data


def h5shuffle():
    parser = argparse.ArgumentParser(description='Shuffle an h5 file using h5py.')
    parser.add_argument('input_file', type=str, help='File to shuffle.')
    parser.add_argument('--output_file', type=str,
                        help='Name of output file. Default: Auto generate name.')
    shuffle_v2(**vars(parser.parse_args()))
