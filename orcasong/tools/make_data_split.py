#!/usr/bin/env python
# -*- coding: utf-8 -*-


__author__ = "Michael Moser, Daniel Guderian"

import os
import warnings
import toml
import argparse
import h5py
import random
import numpy as np


def get_parser():
    # TODO deprecated
    raise NotImplementedError(
        "make_data_split has been renamed to orcasong make_data_split")


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "make_data_split",
        description="Create datasets based on the run_id's."
        "Use the config to add input folder and set the ranges."
        "Outputs a list in an txt file that can be used to "
        "concatenate the files specfied"
    )
    parser.add_argument(
        "config", type=str, help="See example config for detailed information"
    )
    parser.set_defaults(func=make_split)


def get_all_ip_group_keys(cfg):
    """
    Gets the keys of all input groups in the config dict.

    The input groups are defined as the dict elements, where the values have the type of a dict.

    Parameters
    ----------
    cfg : dict
        Dict that contains all configuration options and additional information.

    Returns
    -------
    ip_group_keys : list
        List of the input_group keys.

    """
    ip_group_keys = []
    for key in cfg:
        if type(cfg[key]) == dict:
            ip_group_keys.append(key)

    return ip_group_keys


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
        if f.endswith(".h5"):
            filepaths.append(dirpath + "/" + f)

    # randomize order
    random.Random(42).shuffle(filepaths)

    return filepaths


def get_number_of_evts(file, dataset_key="y"):
    """
    Returns the number of events of a file looking at the given dataset.

    Parameters
    ----------
    file : h5 file
        File to read the number of events from.
    dataset_key : str
        String which specifies, which dataset in a h5 file should be used for calculating the number of events.

    Returns
    -------
    n_evts : int
        The number of events in that file.

    """

    f = h5py.File(file, "r")
    dset = f[dataset_key]
    n_evts = dset.shape[0]
    f.close()

    return n_evts


def get_number_of_evts_and_run_ids(
    list_of_files, dataset_key="y", run_id_col_name="run_id"
):
    """
    Gets the number of events and the run_ids for all hdf5 files in the list_of_files.

    The number of events is calculated based on the dataset, which is specified with the dataset_key parameter.

    Parameters
    ----------
    list_of_files : list
        List which contains filepaths to h5 files.
    dataset_key : str
        String which specifies, which dataset in a h5 file should be used for calculating the number of events.
    run_id_col_name : str
        String, which specifies the column name of the 'run_id' column.

    Returns
    -------
    total_number_of_evts : int
        The cumulative (total) number of events.
    mean_number_of_evts_per_file : float
        The mean number of evts per file.
    run_ids : list
        List containing the run_ids of the files in the list_of_files.

    """

    total_number_of_evts = 0
    run_ids = []

    for i, fpath in enumerate(list_of_files):
        f = h5py.File(fpath, "r")

        dset = f[dataset_key]
        n_evts = dset.shape[0]
        total_number_of_evts += n_evts

        run_id = f[dataset_key][0][run_id_col_name]
        run_ids.append(run_id)

        f.close()

    mean_number_of_evts_per_file = total_number_of_evts / len(list_of_files)

    return total_number_of_evts, mean_number_of_evts_per_file, run_ids


def split(a, n):
    """
    Splits a list into n equal sized (if possible! if not, approximately) chunks.

    Parameters
    ----------
    a : list
        A list that should be split.
    n : int
        Number of times the input list should be split.

    Returns
    -------
    a_split : list
        The input list a, which has been split into n chunks.

    """
    # from https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    a_split = list(
        (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))
    )
    return a_split


def print_input_statistics(cfg, ip_group_keys):
    """
    Prints some useful information for each input_group.

    Parameters
    ----------
    cfg : dict
        Dict that contains all configuration options and additional information.
    ip_group_keys : list
        List of the input_group keys.

    """

    print("----------------------------------------------------------------------")
    print("Printing input statistics for your " + cfg["toml_filename"] + " input:")
    print("----------------------------------------------------------------------")

    print(
        "Your input .toml file has the following data input groups: "
        + str(ip_group_keys)
    )
    print("Total number of events: " + str(cfg["n_evts_total"]))

    for key in ip_group_keys:
        print("--------------------------------------------------------------------")
        print("Info for group " + key + ":")
        print("Directory: " + cfg[key]["dir"])
        print("Total number of files: " + str(cfg[key]["n_files"]))
        print("Total number of events: " + str(cfg[key]["n_evts"]))
        print(
            "Mean number of events per file: "
            + str(round(cfg[key]["n_evts_per_file_mean"], 3))
        )
        print("--------------------------------------------------------------------")


def add_fpaths_for_data_split_to_cfg(cfg, key):
    """
    Adds all the filepaths for the output files into a list, and puts them into the cfg['output_dsplit'][key] location
    for all dsplits (train, validate, rest).

    Parameters
    ----------
    cfg : dict
        Dict that contains all configuration options and additional information.
    key : str
        The key of an input_group.

    """

    fpath_lists = {"train": [], "validate": [], "rest": []}
    for i, fpath in enumerate(cfg[key]["fpaths"]):

        run_id = cfg[key]["run_ids"][i]

        for dsplit in ["train", "validate", "rest"]:
            if "run_ids_" + dsplit in cfg[key]:
                if (
                    cfg[key]["run_ids_" + dsplit][0]
                    <= run_id
                    <= cfg[key]["run_ids_" + dsplit][1]
                ):
                    fpath_lists[dsplit].append(fpath)

    for dsplit in ["train", "validate", "rest"]:
        if len(fpath_lists[dsplit]) == 0:
            continue

        n_files_dsplit = cfg["n_files_" + dsplit]
        fpath_lists[dsplit] = split(fpath_lists[dsplit], n_files_dsplit)
        if "output_" + dsplit not in cfg:
            cfg["output_" + dsplit] = dict()
        cfg["output_" + dsplit][key] = fpath_lists[dsplit]


def make_dsplit_list_files(cfg):
    """
    Writes .list files of the datasplits to the disk, with the information in the cfg['output_dsplit'] dict.

    Parameters
    ----------
    cfg : dict
        Dict that contains all configuration options and additional information.

    """
    # check if //conc_list_files folder exists, if not create it.
    if not os.path.exists(cfg["output_file_folder"] + "/conc_list_files"):
        os.makedirs(cfg["output_file_folder"] + "/conc_list_files")

    print()
    print()
    print("In an run-by-run MC the run_id's might not be continuous.")
    print("Here are the actual numbers in the split sets:")
    print("----------------------------------------------")

    # loop over the different specified sets
    for dsplit in ["train", "validate", "rest"]:

        if "output_" + dsplit not in cfg:
            continue

        print(dsplit, "set:")

        first_key = list(cfg["output_" + dsplit].keys())[0]
        n_output_files = len(cfg["output_" + dsplit][first_key])

        # initialize counter of events for all input groups
        imput_groups_dict = cfg["output_" + dsplit]
        final_number_of_events = np.zeros(len(imput_groups_dict))

        # loop over the number of outputfiles for each set
        for i in range(n_output_files):
            fpath_output = (
                cfg["output_file_folder"]
                + "/conc_list_files/"
                + cfg["output_file_name"]
                + "_"
                + dsplit
                + "_"
                + str(i)
                + ".txt"
            )

            # save the txt list
            if "output_lists" not in cfg:
                cfg["output_lists"] = list()
            cfg["output_lists"].append(fpath_output)

            with open(fpath_output, "w") as f_out:
                for j in range(len(imput_groups_dict)):
                    keys = list(imput_groups_dict.keys())

                    for fpath in imput_groups_dict[keys[j]][i]:
                        # also count here the actual sizes
                        final_number_of_events[j] += get_number_of_evts(fpath)
                        f_out.write(fpath + "\n")

        # and then print them
        for i in range(len(imput_groups_dict)):
            print(keys[i], ":", int(final_number_of_events[i]))

        print("----------------------------------------------")


def make_concatenate_and_shuffle_scripts(cfg):
    """
    Function that writes qsub .sh files which concatenates all files inside the list files.

    Parameters
    ----------
    cfg : dict
        Dict that contains all configuration options and additional information.

    """

    dirpath = cfg["output_file_folder"]

    if not os.path.exists(
        dirpath + "/logs"
    ):  # check if /logs folder exists, if not create it.
        os.makedirs(dirpath + "/logs")
    if not os.path.exists(
        dirpath + "/job_scripts"
    ):  # check if /job_scripts folder exists, if not create it.
        os.makedirs(dirpath + "/job_scripts")
    if not os.path.exists(
        dirpath + "/data_split"
    ):  # check if /data_split folder exists, if not create it.
        os.makedirs(dirpath + "/data_split")

    # not available atm...
    # chunksize = '' if cfg['chunksize'] is None else ' --chunksize ' + str(cfg['chunksize'])
    # complib = '' if cfg['complib'] is None else ' --complib ' + str(cfg['complib'])
    # complevel = '' if cfg['complevel'] is None else ' --complevel ' + str(cfg['complevel'])

    # make qsub .sh file for concatenating
    for listfile_fpath in cfg["output_lists"]:
        listfile_fname = os.path.basename(listfile_fpath)
        listfile_fname_wout_ext = os.path.splitext(listfile_fname)[0]
        conc_outputfile_fpath = (
            cfg["output_file_folder"] + "/data_split/" + listfile_fname_wout_ext + ".h5"
        )

        fpath_bash_script = (
            dirpath + "/job_scripts/concatenate_h5_" + listfile_fname_wout_ext + ".sh"
        )

        with open(fpath_bash_script, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write("\n")
            f.write("source " + cfg["venv_path"] + "activate" + "\n")
            f.write("\n")
            f.write("# Concatenate the files in the list\n")

            f.write(
                "concatenate " + listfile_fpath + " --outfile " + conc_outputfile_fpath
            )
            # at the moment it is not possible to set the comp opts like this+ chunksize + complib + complevel

    # make qsub .sh file for shuffling

    for listfile_fpath in cfg["output_lists"]:
        listfile_fname = os.path.basename(listfile_fpath)
        listfile_fname_wout_ext = os.path.splitext(listfile_fname)[0]

        # This is the input for the shuffle tool!
        conc_outputfile_fpath = (
            cfg["output_file_folder"] + "/data_split/" + listfile_fname_wout_ext + ".h5"
        )

        fpath_bash_script = (
            dirpath + "/job_scripts/shuffle_h5_" + listfile_fname_wout_ext + ".sh"
        )

        with open(fpath_bash_script, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write("\n")
            f.write("source " + cfg["venv_path"] + "activate \n")
            f.write("\n")
            f.write("# Shuffle the h5 file \n")

            f.write(
                "h5shuffle2 " + conc_outputfile_fpath + " --max_ram 1000000000 \n"
            )  # fix to 1GB ram; in lyon using a fraction
            # is difficult...
            # time python shuffle/shuffle_h5.py'
            # + delete_flag_shuffle_tool
            # + chunksize + complib + complevel

            if cfg["shuffle_delete"]:
                f.write("\n")
                f.write("rm " + conc_outputfile_fpath + "\n")


def main():
    """
    Main function to make the data split.
    """

    # load the config
    parser = get_parser()
    parsed_args = parser.parse_args()
    make_split(parsed_args.config)


def make_split(config_file):
    # decode config
    cfg = toml.load(config_file)
    cfg["toml_filename"] = config_file

    # set some defaults/Nones - at the moment setting of the com opts is not available!
    # if 'chunksize' not in cfg: cfg['chunksize'] = None
    # if 'complib' not in cfg: cfg['complib'] = None
    # if 'complevel' not in cfg: cfg['complevel'] = None

    # read out all the input groups
    ip_group_keys = get_all_ip_group_keys(cfg)

    # and now loop over input groups extracting info
    n_evts_total = 0
    for key in ip_group_keys:
        print("Collecting information from input group " + key)
        cfg[key]["fpaths"] = get_h5_filepaths(cfg[key]["dir"])
        cfg[key]["n_files"] = len(cfg[key]["fpaths"])
        (
            cfg[key]["n_evts"],
            cfg[key]["n_evts_per_file_mean"],
            cfg[key]["run_ids"],
        ) = get_number_of_evts_and_run_ids(cfg[key]["fpaths"], dataset_key="y")

        n_evts_total += cfg[key]["n_evts"]

    cfg["n_evts_total"] = n_evts_total
    # print the extracted statistics
    print_input_statistics(cfg, ip_group_keys)

    if cfg["print_only"] is True:
        from sys import exit

        exit()

    for key in ip_group_keys:
        add_fpaths_for_data_split_to_cfg(cfg, key)

    # create the list files
    make_dsplit_list_files(cfg)

    # create bash scripts that can be submitted to do the concatenation and shuffle
    if cfg["make_qsub_bash_files"] is True:
        make_concatenate_and_shuffle_scripts(cfg)


if __name__ == "__main__":
    main()
