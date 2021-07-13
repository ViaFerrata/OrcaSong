import os
import time
import h5py
import warnings


__author__ = 'Stefan Reck, Michael Moser'


class FileConcatenator:
    """
    For concatenating many small h5 files to a single large one in
    km3pipe-compatible format.

    Parameters
    ----------
    input_files : List
        List that contains all filepaths of the input files.
    skip_errors : bool
        If true, ignore files that can't be concatenated.
    comptopts_update : dict, optional
        Overwrite the compression options that get read from the
        first file. E.g. {'chunksize': 10} to get a chunksize of 10.

    Attributes
    ----------
    comptopts : dict
        Options for compression. They are read from the first input file,
        but they can be updated as well during init.
    cumu_rows : dict
        Foe each dataset, the cumulative number of rows (axis_0) of the specified
        input .h5 files (i.e. [0,100,200,300,...] if each file has 100 rows).

    """
    def __init__(self, input_files, skip_errors=False, comptopts_update=None):
        self.skip_errors = skip_errors
        print(f"Checking {len(input_files)} files ...")

        self.input_files, self.cumu_rows = self._get_cumu_rows(input_files)

        # Get compression options from first file in the list
        self.comptopts = get_compopts(self.input_files[0])
        if comptopts_update:
            self.comptopts.update(comptopts_update)
        print("\n".join([f"  {k}:\t{v}" for k, v in self.comptopts.items()]))

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

    def concatenate(self, output_filepath, append_used_files=True):
        """
        Concatenate the input files.

        Parameters
        ----------
        output_filepath : str
            Path of the concatenated output file.
        append_used_files : bool
            If True (default), add a dataset called 'used_files' to the
            output that contains the paths of the input_files.

        """
        print(f"Creating file {output_filepath}")
        with h5py.File(output_filepath, 'x') as f_out:
            start_time = time.time()
            for input_file_nmbr, input_file in enumerate(self.input_files):
                print(f'Processing file {input_file_nmbr+1}/'
                      f'{len(self.input_files)}: {input_file}')
                with h5py.File(input_file, 'r') as f_in:
                    self._conc_file(f_in, f_out, input_file_nmbr)
                f_out.flush()
            elapsed_time = time.time() - start_time

            if append_used_files:
                print("Adding used files to output")
                f_out.create_dataset(
                    "used_files",
                    data=[n.encode("ascii", "ignore") for n in self.input_files]
                )

        copy_attrs(self.input_files[0], output_filepath)

        print(f"\nConcatenation complete!"
              f"\nElapsed time: {elapsed_time/60:.2f} min "
              f"({elapsed_time/len(self.input_files):.2f} s per file)")

    def _conc_file(self, f_in, f_out, input_file_nmbr):
        """ Conc one file to the output. """
        for dset_name in f_in:
            if is_folder_ignored(dset_name):
                # we dont need datasets created by pytables anymore
                continue
            input_dataset = f_in[dset_name]
            folder_data = input_dataset[()]

            if input_file_nmbr > 0:
                # we need to add the current number of the
                # group_id / index in the file_output to the
                # group_ids / indices of the file that is to be appended
                last_index = self.cumu_rows[dset_name][input_file_nmbr] - 1
                if (dset_name.endswith("_indices") and
                        "index" in folder_data.dtype.names):
                    folder_data["index"] += (
                            f_out[dset_name][last_index]["index"] +
                            f_out[dset_name][last_index]["n_items"]
                    )
                elif folder_data.dtype.names and "group_id" in folder_data.dtype.names:
                    # add 1 because the group_ids start with 0
                    folder_data["group_id"] += f_out[dset_name][last_index]["group_id"] + 1

            if input_file_nmbr == 0:
                # first file; create the dataset
                dset_shape = (self.cumu_rows[dset_name][-1],) + folder_data.shape[1:]
                print(f"\tCreating dataset '{dset_name}' with shape {dset_shape}")
                output_dataset = f_out.create_dataset(
                    dset_name,
                    data=folder_data,
                    maxshape=dset_shape,
                    chunks=(self.comptopts["chunksize"],) + folder_data.shape[1:],
                    compression=self.comptopts["complib"],
                    compression_opts=self.comptopts["complevel"],
                    shuffle=self.comptopts["shuffle"],
                )
                output_dataset.resize(dset_shape[0], axis=0)

            else:
                f_out[dset_name][
                    self.cumu_rows[dset_name][input_file_nmbr]:
                    self.cumu_rows[dset_name][input_file_nmbr + 1]
                ] = folder_data

    def _get_cumu_rows(self, input_files):
        """
        Checks if all the files can be safely concatenated to the first one.

        """
        # names of datasets that will be in the output; read from first file
        with h5py.File(input_files[0], 'r') as f:
            target_datasets = strip_keys(list(f.keys()))

        errors, valid_input_files = [], []
        cumu_rows = {k: [0] for k in target_datasets}

        for i, file_name in enumerate(input_files, start=1):
            file_name = os.path.abspath(file_name)
            try:
                rows_this_file = _get_rows(file_name, target_datasets)
            except Exception as e:
                errors.append(e)
                warnings.warn(f"Error during check of file {i}: {file_name}")
                continue
            valid_input_files.append(file_name)
            for k in target_datasets:
                cumu_rows[k].append(cumu_rows[k][-1] + rows_this_file[k])

        if errors:
            print("\n------- Errors -------\n----------------------")
            for error in errors:
                warnings.warn(str(error))
                print("")
            err_str = f"{len(errors)} error(s) during check of files! See above"
            if self.skip_errors:
                warnings.warn(err_str)
            else:
                raise OSError(err_str)

        print(f"Valid input files: {len(valid_input_files)}/{len(input_files)}")
        print("Datasets:\t" + ", ".join(target_datasets))
        return valid_input_files, cumu_rows


def _get_rows(file_name, target_datasets):
    """ Get no of rows from a file and check if its good for conc'ing. """
    with h5py.File(file_name, 'r') as f:
        # check if all target datasets are in the file
        if not all(k in f.keys() for k in target_datasets):
            raise KeyError(
                f"File {file_name} does not have the "
                f"keys of the first file! "
                f"It has {f.keys()} First file: {target_datasets}"
            )
        # check if all target datasets in the file have the same length
        # for datasets that have indices: only check indices
        plain_rows = [
            f[k].shape[0] for k in target_datasets
            if not (f[k].attrs.get("indexed") and f"{k}_indices" in target_datasets)
        ]
        if not all(row == plain_rows[0] for row in plain_rows):
            raise ValueError(
                f"Datasets in file {file_name} have varying length! "
                f"{dict(zip(target_datasets, plain_rows))}"
            )
        # check if the file has additional datasets apart from the target keys
        if not all(k in target_datasets for k in strip_keys(list(f.keys()))):
            warnings.warn(
                f"Additional datasets found in file {file_name} compared "
                f"to the first file, they wont be in the output! "
                f"This file: {strip_keys(list(f.keys()))} "
                f"First file {target_datasets}"
            )
        rows = {k: f[k].shape[0] for k in target_datasets}
    return rows


def strip_keys(f_keys):
    """ Remove unwanted keys from list. """
    return [x for x in f_keys if not is_folder_ignored(x)]


def is_folder_ignored(folder_name):
    """
    Defines datasets which should be ignored during concat.

    Remove pytables folders starting with '_i_', because the shape
    of its first axis does not correspond to the number of events
    in the file. All other folders normally have an axis_0 shape
    that is equal to the number of events in the file.
    Also remove bin_stats.

    """
    return any([s in folder_name for s in (
        '_i_', "bin_stats", "raw_header", "header")])


def get_compopts(file):
    """
    Get the following compression options from a h5 file as a dict:

    complib : str
        Specifies the compression library that should be used for saving
        the concatenated output files.
    complevel : None/int
        Specifies the compression level that should be used for saving
        the concatenated output files.
        A compression level is only available for gzip compression, not lzf!
    chunksize : None/int
        Specifies the chunksize for axis_0 in the concatenated output files.
    shuffle : bool
        Enable shuffle filter for chunks.

    """
    with h5py.File(file, 'r') as f:
        # for reading the comptopts, take first datsets thats not indexed
        dset_names = strip_keys(list(f.keys()))
        for dset_name in dset_names:
            if f"{dset_name}_indices" not in dset_names:
                break
        dset = f[dset_name]
        comptopts = {}
        comptopts["complib"] = dset.compression
        if comptopts["complib"] == 'lzf':
            comptopts["complevel"] = None
        else:
            comptopts["complevel"] = dset.compression_opts
        comptopts["chunksize"] = dset.chunks[0]
        comptopts["shuffle"] = dset.shuffle
    return comptopts


def copy_attrs(source_file, target_file):
    """
    Copy file and dataset attributes from one h5 file to another.
    """
    print("Copying attributes")
    with h5py.File(source_file, "r") as src:
        with h5py.File(target_file, "a") as trg:
            _copy_attrs(src, trg)
            for dset_name, target_dataset in trg.items():
                if dset_name in src:
                    _copy_attrs(src[dset_name], target_dataset)


def _copy_attrs(src_datset, target_dataset):
    for k in src_datset.attrs.keys():
        try:
            if k not in target_dataset.attrs:
                target_dataset.attrs[k] = src_datset.attrs[k]
        except TypeError as e:
            # above can fail if attr is bool and created using pt
            warnings.warn(f"Error: Can not copy attribute {k}: {e}")


def concatenate(file, outfile="concatenated.h5", no_used_files=False, skip_errors=False):
    """ Concatenate wrapped in a function. """
    if len(file) == 1:
        fc = FileConcatenator.from_list(
            file[0],
            skip_errors=skip_errors
        )
    else:
        fc = FileConcatenator(
            input_files=file,
            skip_errors=skip_errors
        )
    fc.concatenate(
        outfile,
        append_used_files=not no_used_files,
    )


def main():
    # TODO deprecated
    raise NotImplementedError(
        "concatenate has been renamed to orcasong concatenate")


if __name__ == '__main__':
    main()
