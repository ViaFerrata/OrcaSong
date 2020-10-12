from unittest import TestCase
import os
import h5py
import numpy as np
import orcasong.tools.postproc as postproc

__author__ = 'Stefan Reck'

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
