from unittest import TestCase
import tempfile
import os
import h5py
import numpy as np
import orcasong.tools.postproc as postproc
import orcasong.tools.shuffle2 as shuffle2
from .test_concatenate import BaseTestClass

__author__ = "Stefan Reck"

test_dir = os.path.dirname(os.path.realpath(__file__))
MUPAGE_FILE = os.path.join(test_dir, "data", "mupage.root.h5")


class TestPostproc(TestCase):
    def setUp(self):
        self.output_file = "temp_output.h5"

    def tearDown(self):
        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def test_shuffle(self):
        postproc.postproc_file(
            input_file=MUPAGE_FILE,
            output_file=self.output_file,
            shuffle=True,
            event_skipper=None,
            delete=False,
            seed=13,
        )

        with h5py.File(self.output_file, "r") as f:
            np.testing.assert_equal(f["event_info"]["event_id"], np.array([1, 0, 2]))
            self.assertTrue("origin" in f.attrs.keys())


class TestShuffleV2(TestCase):
    def setUp(self):
        self.temp_input = "temp_input.h5"
        self.temp_output = "temp_output.h5"

        self.x, self.y = _make_shuffle_dummy_file(self.temp_input)
        np.random.seed(42)
        shuffle2.h5shuffle2(
            input_file=self.temp_input,
            output_file=self.temp_output,
            datasets=("x", "y"),
            max_ram=400,  # -> 2 batches
        )

    def tearDown(self):
        for f in (self.temp_input, self.temp_output):
            if os.path.exists(f):
                os.remove(f)

    def test_shuffled_has_same_entries_as_input(self):
        with h5py.File(self.temp_output, "r") as f:
            x_s = f["x"][()]
            np.testing.assert_array_equal(
                self.x[:, 1:], x_s[:, 1:][np.argsort(x_s[:, 0])]
            )

    def test_all_shuffled_datasets_have_same_order(self):
        with h5py.File(self.temp_output, "r") as f:
            np.testing.assert_array_equal(f["x"][:, 0], f["y"][:, 0])

    def test_seed_produces_this_shuffled_order(self):
        target_order = np.array(
            [
                12.0,
                5.0,
                0.0,
                20.0,
                6.0,
                2.0,
                19.0,
                13.0,
                1.0,
                9.0,
                8.0,
                4.0,
                3.0,
                11.0,
                16.0,
                15.0,
                7.0,
                10.0,
                17.0,
                14.0,
                21.0,
                18.0,
            ]
        )
        with h5py.File(self.temp_output, "r") as f:
            np.testing.assert_array_equal(f["x"][:, 0], target_order)

    def test_run_3_iterations(self):
        # just check if it goes through without errors
        fname = "temp_output_triple.h5"
        try:
            shuffle2.h5shuffle2(
                input_file=self.temp_input,
                output_file=fname,
                datasets=("x", "y"),
                iterations=3,
            )
        finally:
            if os.path.exists(fname):
                os.remove(fname)


class TestShuffleIndexed(BaseTestClass.BaseIndexedFile):
    def setUp(self) -> None:
        self.outfile = "temp_out.h5"
        shuffle2.h5shuffle2(
            self.infile.name,
            output_file=self.outfile,
            datasets=("x",),
            seed=2,
        )

    def tearDown(self) -> None:
        if os.path.exists(self.outfile):
            os.remove(self.outfile)

    def test_check_x(self):
        with h5py.File(self.outfile) as f_out:
            np.testing.assert_array_equal(
                f_out["x"],
                np.concatenate([np.arange(17, 20), np.arange(5, 17), np.arange(0, 5)]),
            )

    def test_check_x_indices_n_items(self):
        with h5py.File(self.outfile) as f_out:
            target_n_items = np.array([3, 12, 5])
            np.testing.assert_array_equal(f_out["x_indices"]["n_items"], target_n_items)

    def test_check_x_indices_index(self):
        with h5py.File(self.outfile) as f_out:
            target_index = np.array([0, 3, 15])
            np.testing.assert_array_equal(f_out["x_indices"]["index"], target_index)


class TestShuffle2LargeFile(TestCase):
    def setUp(self):
        self.infile = tempfile.NamedTemporaryFile()
        with h5py.File(self.infile, "w") as f:
            dset_x = f.create_dataset("x", data=np.arange(2000), chunks=(11, ))
            dset_x.attrs.create("indexed", 1)
            n_items = np.ones(100) * 20
            self.index = np.concatenate([[0.], np.cumsum(n_items)[:-1]])
            indices = np.array(
                list(zip(self.index, n_items)),
                dtype=[("index", "<i8"), ("n_items", "<i8")],
            )
            f.create_dataset("x_indices", data=indices, chunks=(14, ))

        self.outfile = "temp_out.h5"
        shuffle2.h5shuffle2(
            self.infile.name,
            output_file=self.outfile,
            datasets=("x",),
            seed=2,
            max_ram=10000,
            iterations=2,
        )

    def test_indices_is_correct(self):
        with h5py.File(self.outfile) as f_out:
            np.testing.assert_array_equal(
                f_out["x_indices"]["index"],
                self.index,
            )


def _make_shuffle_dummy_file(filepath):
    x = np.random.rand(22, 2)
    x[:, 0] = np.arange(22)
    y = np.random.rand(22, 3)
    y[:, 0] = np.arange(22)
    with h5py.File(filepath, "w") as f:
        f.create_dataset("x", data=x, chunks=(5, 2))
        f.create_dataset("y", data=y, chunks=(5, 3))
    return x, y
