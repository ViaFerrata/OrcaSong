import tempfile
from unittest import TestCase
import numpy as np
import h5py
import orcasong.tools.concatenate as conc

__author__ = 'Stefan Reck'


class TestFileConcatenator(TestCase):
    """
    Test concatenation on pre-generated h5 files. They are in test/data.

    """
    @classmethod
    def setUpClass(cls):
        cls.dummy_file_1 = tempfile.NamedTemporaryFile()
        _create_dummy_file(
            cls.dummy_file_1, columns=10, val_array=1, val_recarray=(1, 3)
        )
        cls.dummy_file_2 = tempfile.NamedTemporaryFile()
        _create_dummy_file(
            cls.dummy_file_2, columns=15, val_array=2, val_recarray=(4, 5)
        )
        cls.dummy_files = (
            cls.dummy_file_1.name,
            cls.dummy_file_2.name,
        )
        cls.compt_opts = {
            'complib': 'gzip', 'complevel': 1, 'chunksize': 5, "shuffle": False
        }

    @classmethod
    def tearDownClass(cls):
        cls.dummy_file_1.close()
        cls.dummy_file_2.close()

    def test_from_list(self):
        with tempfile.NamedTemporaryFile("w+") as tf:
            tf.writelines([f + "\n" for f in self.dummy_files])
            tf.seek(0)
            fc = conc.FileConcatenator.from_list(tf.name)
            self.assertSequenceEqual(self.dummy_files, fc.input_files)

    def test_get_compopts(self):
        comptopts = conc.get_compopts(self.dummy_files[0])
        self.assertDictEqual(comptopts, self.compt_opts)

    def test_fc_get_comptopts(self):
        fc = conc.FileConcatenator(self.dummy_files)
        self.assertDictEqual(fc.comptopts, self.compt_opts)

    def test_fc_get_comptopts_updates(self):
        fc = conc.FileConcatenator(self.dummy_files, comptopts_update={'chunksize': 1})
        target_compt_opts = dict(self.compt_opts)
        target_compt_opts["chunksize"] = 1
        self.assertDictEqual(fc.comptopts, target_compt_opts)

    def test_get_cumu_rows(self):
        fc = conc.FileConcatenator(self.dummy_files)
        np.testing.assert_array_equal(fc.cumu_rows, [0, 10, 25])

    def test_concatenate_used_files(self):
        fc = conc.FileConcatenator(self.dummy_files)
        with tempfile.TemporaryFile() as tf:
            fc.concatenate(tf)
            with h5py.File(tf, "r") as f:
                self.assertSequenceEqual(
                    f["used_files"][()].tolist(),
                    [n.encode("ascii", "ignore") for n in self.dummy_files],
                )

    def test_concatenate_attrs(self):
        fc = conc.FileConcatenator(self.dummy_files)
        with tempfile.TemporaryFile() as tf:
            fc.concatenate(tf)
            with h5py.File(tf, "r") as f:
                target_attrs = dict(f.attrs)
                target_dset_attrs = dict(f["numpy_array"].attrs)
            with h5py.File(self.dummy_files[0], "r") as f:
                source_attrs = dict(f.attrs)
                source_dset_attrs = dict(f["numpy_array"].attrs)

            self.assertDictEqual(source_attrs, target_attrs)
            self.assertDictEqual(source_dset_attrs, target_dset_attrs)

    def test_concatenate_array(self):
        fc = conc.FileConcatenator(self.dummy_files)
        with tempfile.TemporaryFile() as tf:
            fc.concatenate(tf)
            with h5py.File(tf, "r") as f:
                target = np.ones((25, 7, 3))
                target[10:, :, :] = 2.
                np.testing.assert_array_equal(
                    target,
                    f["numpy_array"][()]
                )

    def test_concatenate_recarray_with_groupid(self):
        fc = conc.FileConcatenator(self.dummy_files)
        with tempfile.TemporaryFile() as tf:
            fc.concatenate(tf)
            with h5py.File(tf, "r") as f:
                target = np.array(
                    [(1, 3, 1)] * 25,
                    dtype=[('x', '<f8'), ('y', '<i8'), ("group_id", "<i8")]
                )
                target["x"][10:] = 4.
                target["y"][10:] = 5.
                target["group_id"] = np.arange(25)
                np.testing.assert_array_equal(
                    target,
                    f["rec_array"][()]
                )


def _create_dummy_file(filepath, columns=10, val_array=1, val_recarray=(1, 3)):
    """ Create a dummy h5 file with an array and a recarray in it. """
    with h5py.File(filepath, "w") as f:
        dset = f.create_dataset(
            "numpy_array",
            data=np.ones(shape=(columns, 7, 3))*val_array,
            chunks=(5, 7, 3),
            compression="gzip",
            compression_opts=1
        )
        dset.attrs.create("test_dset", "ok")

        rec_array = np.array(
            [val_recarray + (1, )] * columns,
            dtype=[('x', '<f8'), ('y', '<i8'), ("group_id", "<i8")]
        )
        rec_array["group_id"] = np.arange(columns)
        f.create_dataset(
            "rec_array",
            data=rec_array,
            chunks=(5,),
            compression="gzip",
            compression_opts=1
        )
        f.attrs.create("test_file", "ok")
