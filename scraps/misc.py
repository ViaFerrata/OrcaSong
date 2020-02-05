import h5py
import numpy as np


def create_dummy_file(filepath, columns=10, val_array=1, val_recarray=(1, 3)):
    """ Create a dummy h5 file with an array and a recarray in it. """
    with h5py.File(filepath, "w") as f:
        f.create_dataset(
            "numpy_array",
            data=np.ones(shape=(columns, 7, 3))*val_array,
            chunks=(5, 7, 3),
            compression="gzip",
            compression_opts=1
        )
        f.create_dataset(
            "rec_array",
            data=np.array([val_recarray] * columns, dtype=[('x', '<f8'), ('y', '<i8')]),
            chunks=(5,),
            compression="gzip",
            compression_opts=1
        )
