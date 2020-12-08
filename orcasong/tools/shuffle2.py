import os
import time
import datetime
import argparse
import numpy as np
import psutil
import h5py

from orcasong.tools.postproc import get_filepath_output, copy_used_files
from orcasong.tools.concatenate import copy_attrs


__author__ = "Stefan Reck"


def shuffle_v2(
        input_file,
        datasets=("x", "y"),
        output_file=None,
        max_ram=None,
        max_ram_fraction=0.45,
        chunks=False,
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
    max_ram : int, optional
        Available ram in bytes. Default: Use fraction of
        maximum available (see max_ram_fraction).
    max_ram_fraction : float
        in [0, 1]. Fraction of ram to use for reading data when max_ram
        is None. Note: when using chunks, this should be <=0.45, since
        lots of ram is needed for in-memory shuffling.
    chunks : bool
        Use chunk-wise readout. Up to x8 speed boost, but will
        only quasi-randomize order! Needs lots of ram
        to be accurate! (use a node with at least 32gb, the more the better)
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
    if max_ram is None:
        max_ram = max_ram_fraction * psutil.virtual_memory().available
        print(f"Using {max_ram_fraction:.2%} of available ram = {max_ram} bytes")

    with h5py.File(input_file, "r") as f_in:
        dset_infos, n_lines = get_dset_infos(f_in, datasets, max_ram)
        np.random.seed(seed)
        print(f"Shuffling datasets {datasets} with {n_lines} lines each")

        if not chunks:
            indices = np.arange(n_lines)
            np.random.shuffle(indices)

            with h5py.File(output_file, "x") as f_out:
                for dset_info in dset_infos:
                    print("Creating dataset", dset_info["name"])
                    make_dset(f_out, dset_info, indices)
                    print("Done!")
        else:
            indices_chunked = get_indices_largest(dset_infos)

            with h5py.File(output_file, "x") as f_out:
                for dset_info in dset_infos:
                    print("Creating dataset", dset_info["name"])
                    make_dset_chunked(f_out, dset_info, indices_chunked)
                    print("Done!")

    copy_used_files(input_file, output_file)
    copy_attrs(input_file, output_file)

    print(f"Elapsed time: "
          f"{datetime.timedelta(seconds=int(time.time() - start_time))}")
    return output_file


def get_indices_largest(dset_infos):
    largest_dset = np.argmax([v["n_batches_chunkwise"] for v in dset_infos])
    dset_info = dset_infos[largest_dset]

    print(f"Lines per batch: {dset_info['lines_per_batch']}")
    if dset_info['lines_per_batch'] <= 50000:
        print("Warning: Should have more than "
              "50 000 lines per batch to ensure proper shuffling!")
    return get_indices_chunked(
        dset_info["n_batches_chunkwise"],
        dset_info["n_chunks"],
        dset_info["chunksize"],
    )


def get_indices_chunked(n_batches, n_chunks, chunksize):
    """ Return a list with the chunkwise shuffled indices of each batch. """
    chunk_indices = np.arange(n_chunks)
    np.random.shuffle(chunk_indices)
    chunk_batches = np.array_split(chunk_indices, n_batches)

    index_batches = []
    for bat in chunk_batches:
        idx = (bat[:, None]*chunksize + np.arange(chunksize)[None, :]).flatten()
        np.random.shuffle(idx)
        index_batches.append(idx)

    return index_batches


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
        chunksize = dset.chunks[0]
        n_chunks = int(np.ceil(n_lines / chunksize))
        # TODO in h5py 3.X, use .nbytes to get uncompressed size
        bytes_per_line = np.asarray(dset[0]).nbytes
        bytes_per_chunk = bytes_per_line * chunksize

        lines_per_batch = int(np.floor(max_ram / bytes_per_line))
        chunks_per_batch = int(np.floor(max_ram / bytes_per_chunk))

        dset_infos.append({
            "name": name,
            "dset": dset,
            "chunksize": chunksize,
            "n_lines": n_lines,
            "n_chunks": n_chunks,
            "bytes_per_line": bytes_per_line,
            "bytes_per_chunk": bytes_per_chunk,
            "lines_per_batch": lines_per_batch,
            "chunks_per_batch": chunks_per_batch,
            "n_batches_linewise": int(np.ceil(n_lines / lines_per_batch)),
            "n_batches_chunkwise": int(np.ceil(n_chunks / chunks_per_batch)),
        })
    return dset_infos, n_lines


def make_dset(f_out, dset_info, indices):
    """ Create a shuffled dataset in the output file. """
    for batch_index in range(dset_info["n_batches_linewise"]):
        print(f"Processing batch {batch_index+1}/{dset_info['n_batches_linewise']}")

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


def make_dset_chunked(f_out, dset_info, indices_chunked):
    """ Create a shuffled dataset in the output file. """
    start_idx = 0
    for batch_index, to_read in enumerate(indices_chunked):
        print(f"Processing batch {batch_index+1}/{len(indices_chunked)}")

        # remove indices outside of dset
        to_read = to_read[to_read < len(dset_info["dset"])]

        # reading has to be done with linearly increasing index
        #  fancy indexing is super slow
        #  so sort -> turn to slices -> read -> conc -> undo sorting
        sort_ix = np.argsort(to_read)
        unsort_ix = np.argsort(sort_ix)
        fancy_indices = to_read[sort_ix]
        slices = slicify(fancy_indices)
        data = np.concatenate([dset_info["dset"][slc] for slc in slices])
        data = data[unsort_ix]

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
            start_idx = len(data)
        else:
            end_idx = start_idx + len(data)
            f_out[dset_info["name"]][start_idx:end_idx] = data
            start_idx = end_idx

    if start_idx != len(dset_info["dset"]):
        print(f"Warning: last index was {start_idx} not {len(dset_info['dset'])}")


def slicify(fancy_indices):
    """ [0,1,2, 6,7,8] --> [0:3, 6:9] """
    steps = np.diff(fancy_indices) != 1
    slice_starts = np.concatenate([fancy_indices[:1], fancy_indices[1:][steps]])
    slice_ends = np.concatenate([fancy_indices[:-1][steps], fancy_indices[-1:]]) + 1
    return [slice(slice_starts[i], slice_ends[i]) for i in range(len(slice_starts))]


def h5shuffle():
    parser = argparse.ArgumentParser(description='Shuffle an h5 file using h5py.')
    parser.add_argument('input_file', type=str,
                        help='File to shuffle.')
    parser.add_argument('--output_file', type=str,
                        help='Name of output file. Default: Auto generate name.')
    parser.add_argument('--chunks', action='store_true',
                        help="Use chunk-wise readout. Up to 8x speed boost, but will "
                             "only quasi-randomize order! Needs lots of ram "
                             "to be accurate!")
    shuffle_v2(**vars(parser.parse_args()))


if __name__ == '__main__':
    h5shuffle()
