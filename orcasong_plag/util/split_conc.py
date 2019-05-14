"""
Distribute filepaths among lists, so that they can be concatenated to a
given number of files, and with a train/val split.

Main function: get_split
    The output will be a list with dicts containing a name for the concatenated
    file, and the paths of the individual files that will be concatenated.
    This can be handed over to concatenate_h5 in orcasong.

Example:
    For a folder with 4 files named file_0.h5 to file_3.h5 and a 50:50 split
    between train/val:

    get_split(folder,
              outfile_basestr='test',
              n_train_files=1,
              n_val_files=1,
              train_frac=0.5)

    [
     {
      'file_list': array(['file_2.h5', file_1.h5]),
      'output_filepath': 'test_train_0.h5',
      'is_train': True,
     },
     {
      'file_list': array(['file_4.h5', 'file_0.h5']),
      'output_filepath': 'test_val_0.h5',
      'is_train': False,
     },
    ]

"""
import os
import numpy as np

__author__ = 'Stefan Reck'


def get_files(folder):
    """
    Get pathes of all h5 files in given folder.
    """
    infiles = os.listdir(folder)
    infiles.sort()

    infile_paths = []
    for infile in infiles:
        if infile.endswith(".h5"):
            infile_paths.append(os.path.join(folder, infile))

    return np.array(infile_paths)


def split_path_list(files, train_frac, n_train_files, n_val_files):
    """
    Get train and val files split according to given fraction, and
    distributed over given number of files.

    Parameters
    ----------
    files : List
        The files.
    train_frac : float
        The fraction of files.
    n_train_files : int
        Total number of resulting train files.
    n_val_files : int
        Total number of resulting val files.

    Returns
    -------
    job_files_train : ndarray
        The train files. They are chosen randomly from the files list.
        The total number of files is the given fraction of the input files.
    job_files_val : ndarray
        The val files, similar to the train files.

    """
    if n_train_files < 1 or n_val_files < 1:
        raise ValueError("Need at least 1 file for train and val.")

    order = np.arange(len(files))
    np.random.shuffle(order)

    len_train_files = int(len(files) * train_frac)
    train_files = files[order[:len_train_files]]
    val_files = files[order[len_train_files:]]

    job_files_train = np.array_split(train_files, n_train_files)
    job_files_val = np.array_split(val_files, n_val_files)

    for fs in job_files_train:
        if len(fs) == 0:
            raise ValueError("No files for an output train file!")

    for fs in job_files_val:
        if len(fs) == 0:
            raise ValueError("No files for an output val file!")

    return job_files_train, job_files_val


def get_split(folder, outfile_basestr, n_train_files=1, n_val_files=1,
              train_frac=0.8, seed=64):
    """
    Distribute filepaths among lists (see doc at top of file).

    The files in each train or val file will be drawn randomly from the
    available files. Each train or val files will be created by its own
    seperately submitted job.

    Parameters
    ----------
    folder : str
        Containing the files to concat.
    n_train_files : int
        Total number of resulting train files.
    n_val_files : int
        Total number of resulting val files.
    outfile_basestr : str
        Path and a base for the name. "train"/"val" and a file number will
        get automatically attached to the name.
    train_frac : float
        The fraction of files in the train set.
    seed : int, optional
        Seed for the random stuff.

    Returns
    -------
    jobs : dict
        Contains the arguments for the concatenate script.

    """
    if seed is not None:
        np.random.seed(seed)

    files = get_files(folder)
    job_files_train, job_files_val = split_path_list(files, train_frac, n_train_files, n_val_files)

    jobs = []

    for i, job_files in enumerate(job_files_train):
        output_filepath = "{}_train_{}.h5".format(outfile_basestr, i)
        job_dict = {
            "file_list": job_files,
            "output_filepath": output_filepath,
            "is_train": True,
        }
        jobs.append(job_dict)

    for i, job_files in enumerate(job_files_val):
        output_filepath = "{}_val_{}.h5".format(outfile_basestr, i)
        job_dict = {
            "file_list": job_files,
            "output_filepath": output_filepath,
            "is_train": False,
        }
        jobs.append(job_dict)

    return jobs
