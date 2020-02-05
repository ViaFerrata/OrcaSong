import time
import h5py
import numpy as np
import argparse
import warnings


__author__ = 'Stefan Reck, Michael Moser'


class FileConcatenator:
    """
    For concatenating many small h5 files to a single large one.

    Attributes
    ----------
    input_files : list
        List that contains all filepaths of the input files.
    comptopts : dict
        Options for compression. They are read from the first input file.
        E.g. complib
    cumu_rows : np.array
        The cumulative number of rows (axis_0) of the specified
        input .h5 files (i.e. [0,100,200,300,...] if each file has 100 rows).

    """
    def __init__(self, input_files):
        self.input_files = input_files
        print(f"Checking {len(self.input_files)} files ...")

        self.cumu_rows = self._get_cumu_rows()
        print(f"Total rows:\t{self.cumu_rows[-1]}")

        # Get compression options from first file in the list
        self.comptopts = get_compopts(self.input_files[0])
        print("\n".join([f"{k}:\t{v}" for k, v in self.comptopts.items()]))

        self._append_mc_index = False

    @classmethod
    def from_list(cls, list_file, n_files=None, **kwargs):
        """
        Initialize with a .txt list containing the filepaths.

        Parameters
        ----------
        list_file : str
            Path to a txt file containing the input filepaths, one per line.
        n_files : int, optional
            Only load these many files from the list.

        """
        input_files = []
        with open(list_file) as f:
            for line in f:
                filepath = line.rstrip('\n')
                if filepath != "":
                    input_files.append(filepath)
        if n_files is not None:
            input_files = input_files[:n_files]
        return cls(input_files, **kwargs)

    def concatenate(self, output_filepath):
        """ Concatenate input files and save output to given path. """
        f_out = h5py.File(output_filepath, 'x')
        start_time = time.time()
        for n, input_file in enumerate(self.input_files):
            print(f'Processing file {n+1}/{len(self.input_files)}: {input_file}')
            f_in = h5py.File(input_file, 'r')

            # create metadata
            if n == 0 and 'format_version' in list(f_in.attrs.keys()):
                f_out.attrs['format_version'] = f_in.attrs['format_version']

            for folder_name in f_in:
                if is_folder_ignored(folder_name):
                    # we dont need datasets created by pytables anymore
                    continue

                folder_data = f_in[folder_name][()]
                if n > 0 and folder_name in [
                        'event_info', 'group_info', 'x_indices', 'y']:
                    # we need to add the current number of the group_id / index in the file_output
                    # to the group_ids / indices of the file that is to be appended
                    column_name = 'group_id' if folder_name in [
                        'event_info', 'group_info', 'y'] else 'index'
                    # add 1 because the group_ids / indices start with 0
                    folder_data[column_name] += np.amax(
                        f_out[folder_name][column_name]) + 1

                if self._append_mc_index and folder_name == "event_info":
                    folder_data = self._modify_event_info(input_file, folder_data)

                if n == 0:
                    # first file; create the dummy dataset with no max shape
                    print(f"\tCreating dataset '{folder_name}' with shape "
                          f"{(self.cumu_rows[-1],) + folder_data.shape[1:]}")
                    output_dataset = f_out.create_dataset(
                        folder_name,
                        data=folder_data,
                        maxshape=(None,) + folder_data.shape[1:],
                        chunks=(self.comptopts["chunksize"],) + folder_data.shape[1:],
                        compression=self.comptopts["complib"],
                        compression_opts=self.comptopts["complevel"],
                    )
                    output_dataset.resize(self.cumu_rows[-1], axis=0)

                else:
                    f_out[folder_name][self.cumu_rows[n]:self.cumu_rows[n + 1]] = folder_data
            f_in.close()
            f_out.flush()

        elapsed_time = time.time() - start_time
        # include the used filepaths in the file
        f_out.create_dataset(
            "used_files",
            data=[n.encode("ascii", "ignore") for n in self.input_files]
        )
        f_out.close()
        print(f"\nConcatenation complete!"
              f"\nElapsed time: {elapsed_time/60:.2f} min "
              f"({elapsed_time/len(self.input_files):.2f} s per file)")

    def _modify_event_info(self, input_file, folder_data):
        raise NotImplementedError

    def _get_cumu_rows(self):
        """
        Get the cumulative number of rows of the input_files.
        Also checks if all the files can be safely concatenated to the
        first one.

        """
        # names of datasets that will be in the output; read from first file
        with h5py.File(self.input_files[0], 'r') as f:
            keys_stripped = strip_keys(list(f.keys()))

        rows_per_file = np.zeros(len(self.input_files) + 1, dtype=int)
        for i, file_name in enumerate(self.input_files, start=1):
            with h5py.File(file_name, 'r') as f:
                if not all(k in f.keys() for k in keys_stripped):
                    raise KeyError(
                        f"File {file_name} does not have the "
                        f"keys of the first file! "
                        f"It has {f.keys()} First file: {keys_stripped}")
                # length of each dataset
                rows = [f[k].shape[0] for k in keys_stripped]
                if not all(row == rows[0] for row in rows):
                    raise ValueError(
                        f"Datasets in file {file_name} have varying length! "
                        f"{dict(zip(keys_stripped, rows))}"
                    )
                if not all(k in keys_stripped for k in strip_keys(list(f.keys()))):
                    warnings.warn(
                        f"Additional datasets found in file {file_name} compared "
                        f"to the first file, they wont be in the output! "
                        f"This file: {strip_keys(list(f.keys()))} "
                        f"First file {keys_stripped}"
                    )
                rows_per_file[i] = rows[0]
        return np.cumsum(rows_per_file)


def strip_keys(f_keys):
    """
    Remove pytables folders starting with '_i_', because the shape
    of its first axis does not correspond to the number of events
    in the file. All other folders normally have an axis_0 shape
    that is equal to the number of events in the file.
    Also remove bin_stats.
    """
    return [x for x in f_keys if not is_folder_ignored(x)]


def is_folder_ignored(folder_name):
    """
    Defines pytable folders which should be ignored during concat.
    """
    return '_i_' in folder_name or "bin_stats" in folder_name


def get_compopts(file):
    """
    Extract the following compression options:

    complib : str
        Specifies the compression library that should be used for saving
        the concatenated output files.
        It's read from the first input file.
    complevel : None/int
        Specifies the compression level that should be used for saving
        the concatenated output files.
        A compression level is only available for gzip compression, not lzf!
        It's read from the first input file.
    chunksize : None/int
        Specifies the chunksize for axis_0 in the concatenated output files.
        It's read from the first input file.

    """
    with h5py.File(file, 'r') as f:
        dset = f[strip_keys(list(f.keys()))[0]]
        comptopts = {}
        comptopts["complib"] = dset.compression
        if comptopts["complib"] == 'lzf':
            comptopts["complevel"] = None
        else:
            comptopts["complevel"] = dset.compression_opts
        comptopts["chunksize"] = dset.chunks[0]
    return comptopts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Concatenate many small h5 files to a single large one. '
                    'Compression options and the datasets to be created in '
                    'the new file will be read from the first input file.')
    parser.add_argument(
        'list_file', type=str, help='A txt list of files to concatenate. '
                                    'One absolute filepath per line. ')
    parser.add_argument(
        'output_filepath', type=str, help='The absoulte filepath of the output '
                                          '.h5 file that will be created. ')
    parsed_args = parser.parse_args()

    fc = FileConcatenator.from_list(parsed_args.list_file)
    fc.concatenate(parsed_args.output_filepath)
