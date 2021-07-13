import os
from unittest import TestCase
import tempfile
import numpy as np
import h5py
import orcasong.core
import orcasong.extractors as extractors
from orcasong.plotting.plot_binstats import read_hists_from_h5file


__author__ = 'Stefan Reck'


test_dir = os.path.dirname(os.path.realpath(__file__))
MUPAGE_FILE = os.path.join(test_dir, "data", "mupage.root.h5")
DET_FILE = os.path.join(test_dir, "data", "KM3NeT_-00000001_20171212.detx")


class TestFileBinner(TestCase):
    """ Assert that the filebinner still produces the same output. """
    @classmethod
    def setUpClass(cls):
        cls.proc = orcasong.core.FileBinner(
            bin_edges_list=[
                ["pos_z", np.linspace(0, 200, 3)],
                ["time", np.linspace(0, 600, 3)],
                ["channel_id", np.linspace(-0.5, 30.5, 3)],
            ],
            extractor=extractors.get_real_data_info_extr(MUPAGE_FILE),
            det_file=DET_FILE,
            add_t0=True,
            keep_event_info=True,
        )
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.outfile = os.path.join(cls.tmpdir.name, "binned.h5")
        cls.proc.run(infile=MUPAGE_FILE, outfile=cls.outfile)
        cls.f = h5py.File(cls.outfile, "r")

    @classmethod
    def tearDownClass(cls):
        cls.f.close()
        cls.tmpdir.cleanup()

    def test_keys(self):
        self.assertSetEqual(set(self.f.keys()), {
            '_i_event_info', '_i_group_info', '_i_y', 'bin_stats',
            'event_info', 'group_info', 'x', 'x_indices', 'y'})

    def test_x(self):
        target = np.array([
            [[[4, 1], [6, 3]],
                [[12, 5], [6, 7]]],
            [[[4, 2], [1, 3]],
                [[5, 7], [8, 5]]],
            [[[3, 3], [2, 4]],
                [[5, 6], [6, 8]]]
        ], dtype=np.uint8)
        np.testing.assert_equal(target, self.f["x"])

    def test_y(self):
        y = self.f["y"][()]
        target = {
            'event_id': np.array([0., 1., 2.]),
            'run_id': np.array([1., 1., 1.]),
            'trigger_mask': np.array([18., 18., 16.]),
            'group_id': np.array([0, 1, 2]),
        }
        for k, v in target.items():
            np.testing.assert_equal(y[k], v)

    def test_bin_stats(self):
        bin_stats = read_hists_from_h5file(self.f)
        target_hists = {
            "channel_id": np.array([20.0, 8.0, 10.0, 8.0, 16.0, 13.0, 11.0, 9.0, 10.0, 11.0]),
            "pos_z": np.array([0.0, 0.0, 0.0, 36.0, 0.0, 19.0, 30.0, 0.0, 31.0, 0.0]),
            "time": np.array([57.0, 59.0]),
        }
        for dim, infos in bin_stats.items():
            np.testing.assert_equal(infos["hist"], target_hists[dim])


class TestFileGraph(TestCase):
    """ Assert that the FileGraph still produces the same output. """
    @classmethod
    def setUpClass(cls):
        # produce test file, once for fixed_length (old format), and once
        # for the new format
        cls.proc_fixed_length, cls.proc = [orcasong.core.FileGraph(
            max_n_hits=3,
            time_window=[0, 50],
            hit_infos=["pos_z", "time", "channel_id"],
            extractor=extractors.get_real_data_info_extr(MUPAGE_FILE),
            det_file=DET_FILE,
            add_t0=False,
            keep_event_info=True,
            correct_timeslew=False,
            fixed_length=fixed_length,
        ) for fixed_length in (True, False)]
        cls.tmpdir = tempfile.TemporaryDirectory()

        cls.outfile_fixed_length = os.path.join(cls.tmpdir.name, "binned_fixed_length.h5")
        cls.proc_fixed_length.run(infile=MUPAGE_FILE, outfile=cls.outfile_fixed_length)
        cls.f_fixed_length = h5py.File(cls.outfile_fixed_length, "r")

        cls.outfile = os.path.join(cls.tmpdir.name, "binned.h5")
        cls.proc.run(infile=MUPAGE_FILE, outfile=cls.outfile)
        cls.f = h5py.File(cls.outfile, "r")

    @classmethod
    def tearDownClass(cls):
        cls.f.close()
        cls.f_fixed_length.close()
        cls.tmpdir.cleanup()

    def test_keys_fixed_length(self):
        self.assertSetEqual(set(self.f_fixed_length.keys()), {
            '_i_event_info', '_i_group_info', '_i_y',
            'event_info', 'group_info', 'x', 'x_indices', 'y'})

    def test_keys(self):
        self.assertSetEqual(set(self.f_fixed_length.keys()), {
            '_i_event_info', '_i_group_info', '_i_y',
            'event_info', 'group_info', 'x', 'x_indices', 'y'})

    def test_x_attrs_fixed_length(self):
        to_check = {
            "hit_info_0": "pos_z",
            "hit_info_1": "time",
            "hit_info_2": "channel_id",
            "hit_info_3": "is_valid",
            "indexed": False,
        }
        attrs = dict(self.f_fixed_length["x"].attrs)
        for k, v in to_check.items():
            self.assertTrue(attrs[k] == v)

    def test_x_attrs(self):
        to_check = {
            "hit_info_0": "pos_z",
            "hit_info_1": "time",
            "hit_info_2": "channel_id",
            "indexed": True,
        }
        attrs = dict(self.f["x"].attrs)
        for k, v in to_check.items():
            self.assertTrue(attrs[k] == v)

    def test_x_fixed_length(self):
        target = np.array([
            [[676.941,  13.,  30.,   1.],
             [461.111,  32.,   9.,   1.],
             [424.941,   1.,  30.,   1.]],
            [[172.83,  32.,  25.,   1.],
             [316.83,   2.,  14.,   1.],
             [461.059,   1.,   3.,   1.]],
            [[496.83,  34.,  25.,   1.],
             [605.111,   9.,   4.,   1.],
             [424.889,  46.,  29.,   1.]]
        ], dtype=np.float32)
        np.testing.assert_equal(target, self.f_fixed_length["x"])

    def test_x(self):
        target = np.array([
            [676.941,  13.,  30.],
            [461.111,  32.,   9.],
            [424.941,   1.,  30.],
            [172.83,  32.,  25.],
            [316.83,   2.,  14.],
            [461.059,   1.,   3.],
            [496.83,  34.,  25.],
            [605.111,   9.,   4.],
            [424.889,  46.,  29.],
        ], dtype=np.float32)
        np.testing.assert_equal(target, self.f["x"])

    def test_y_fixed_length(self):
        y = self.f_fixed_length["y"][()]
        target = {
            'event_id': np.array([0., 1., 2.]),
            'run_id': np.array([1., 1., 1.]),
            'trigger_mask': np.array([18., 18., 16.]),
            'group_id': np.array([0, 1, 2]),
        }
        for k, v in target.items():
            np.testing.assert_equal(y[k], v)

    def test_y(self):
        y = self.f["y"][()]
        target = {
            'event_id': np.array([0., 1., 2.]),
            'run_id': np.array([1., 1., 1.]),
            'trigger_mask': np.array([18., 18., 16.]),
            'group_id': np.array([0, 1, 2]),
        }
        for k, v in target.items():
            np.testing.assert_equal(y[k], v)