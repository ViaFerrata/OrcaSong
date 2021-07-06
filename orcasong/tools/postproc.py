"""
Scripts for postprocessing h5 files, e.g. shuffling.
"""
import os
import argparse
import warnings

import h5py
import km3pipe as kp
import km3modules as km
from orcasong.modules import EventSkipper
from orcasong.tools.concatenate import get_compopts, copy_attrs


def postproc_file(
        input_file,
        output_file=None,
        shuffle=True,
        event_skipper=None,
        delete=False,
        seed=42,
        statusbar_every=1000):
    """
    Postprocess a file using km3pipe after it has been preprocessed in OrcaSong.

    Parameters
    ----------
    input_file : str
        Path of the file that will be processed.
    output_file : str, optional
        If given, this will be the name of the output file.
        Otherwise, a name is auto generated.
    shuffle : bool
        Shuffle order of events.
    event_skipper : func, optional
        Function that takes the blob as an input, and returns a bool.
        If the bool is true, the event will be skipped.
    delete : bool
        Specifies if the input file should be deleted after processing.
    seed : int
        Sets a fixed random seed for the shuffling.
    statusbar_every : int or None
        After how many line a km3pipe status should be printed.

    Returns
    -------
    output_file : str
        Path to the output file.

    """
    if output_file is None:
        output_file = get_filepath_output(
            input_file, shuffle=shuffle, event_skipper=event_skipper)
    if os.path.exists(output_file):
        raise FileExistsError(output_file)

    print(f'Setting a Global Random State with the seed < {seed} >.')
    km.GlobalRandomState(seed=seed)

    comptopts = get_compopts(input_file)
    # km3pipe uses pytables for saving the shuffled output file,
    # which has the name 'zlib' for the 'gzip' filter
    if comptopts["complib"] == 'gzip':
        comptopts["complib"] = 'zlib'

    pipe = kp.Pipeline()
    if statusbar_every is not None:
        pipe.attach(km.common.StatusBar, every=statusbar_every)
        pipe.attach(km.common.MemoryObserver, every=statusbar_every)
    pipe.attach(
        kp.io.hdf5.HDF5Pump,
        filename=input_file,
        shuffle=shuffle,
        reset_index=True,
    )
    if event_skipper is not None:
        pipe.attach(EventSkipper, event_skipper=event_skipper)
    pipe.attach(
        kp.io.hdf5.HDF5Sink,
        filename=output_file,
        complib=comptopts["complib"],
        complevel=comptopts["complevel"],
        chunksize=comptopts["chunksize"],
        flush_frequency=1000,
    )
    pipe.drain()

    copy_used_files(input_file, output_file)
    copy_attrs(input_file, output_file)
    if delete:
        print("Deleting original file")
        os.remove(input_file)

    print("Done!")
    return output_file


def copy_used_files(source_file, target_file):
    """ Copy the "used_files" dataset from one h5 file to another, if it is present.
    """
    with h5py.File(source_file, "r") as src:
        if "used_files" in src:
            print("Copying used_files dataset")
            with h5py.File(target_file, "a") as trg:
                trg.create_dataset("used_files", data=src["used_files"])


def get_filepath_output(input_file, shuffle=True, event_skipper=None):
    """ Get the filename of the shuffled / rebalanced output file as a str.
    """
    fname_adtn = ''
    if shuffle:
        fname_adtn += '_shuffled'
    if event_skipper is not None:
        fname_adtn += '_reb'
    return f"{os.path.splitext(input_file)[0]}{fname_adtn}.h5"


def h5shuffle():
    # TODO deprecated
    raise NotImplementedError(
        "h5shuffle has been renamed to orcasong h5shuffle")
