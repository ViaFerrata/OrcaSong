import tempfile
from unittest import TestCase
import os
import numpy as np
import h5py
import orcasong.tools.concatenate as conc

__author__ = 'Stefan Reck'


class TestFileConcatenator(TestCase):
    """
    Test concatenation on pre-generated h5 files. They are in test/data.

    create_dummy_file(
        "dummy_file_1.h5", columns=10, val_array=1, val_recarray=(1, 3)
    )
    create_dummy_file(
        "dummy_file_2.h5", columns=15, val_array=2, val_recarray=(4, 5)
    )

    """
    def setUp(self):
        # the files to test on
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        self.dummy_files = (
            os.path.join(data_dir, "dummy_file_1.h5"),  # 10 columns
            os.path.join(data_dir, "dummy_file_2.h5"),  # 15 columns
        )
        # their compression opts
        self.compt_opts = {
            'complib': 'gzip', 'complevel': 1, 'chunksize': 5
        }

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

    def test_get_cumu_rows(self):
        fc = conc.FileConcatenator(self.dummy_files)
        np.testing.assert_array_equal(fc.cumu_rows, [0, 10, 25])

    def test_concatenate_used_files(self):
        fc = conc.FileConcatenator(self.dummy_files)
        with tempfile.TemporaryFile() as tf:
            fc.concatenate(tf)
            with h5py.File(tf) as f:
                self.assertSequenceEqual(
                    f["used_files"][()].tolist(),
                    [n.encode("ascii", "ignore") for n in self.dummy_files],
                )

    def test_concatenate_array(self):
        fc = conc.FileConcatenator(self.dummy_files)
        with tempfile.TemporaryFile() as tf:
            fc.concatenate(tf)
            with h5py.File(tf) as f:
                target = np.ones((25, 7, 3))
                target[10:, :, :] = 2.
                np.testing.assert_array_equal(
                    target,
                    f["numpy_array"][()]
                )

    def test_concatenate_recarray(self):
        fc = conc.FileConcatenator(self.dummy_files)
        with tempfile.TemporaryFile() as tf:
            fc.concatenate(tf)
            with h5py.File(tf) as f:
                target = np.array(
                    [(1, 3)] * 25,
                    dtype=[('x', '<f8'), ('y', '<i8')]
                )
                target["x"][10:] = 4.
                target["y"][10:] = 5.
                np.testing.assert_array_equal(
                    target,
                    f["rec_array"][()]
                )