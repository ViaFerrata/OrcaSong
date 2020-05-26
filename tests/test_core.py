import os
from unittest import TestCase
import tempfile
import numpy as np
import h5py
import orcasong.core
import orcasong.mc_info_extr
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
            mc_info_extr=orcasong.mc_info_extr.get_real_data,
            det_file=DET_FILE,
            add_t0=True,
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
                [[11, 5], [7, 7]]],
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
            "time": np.array([56.0, 60.0]),
        }
        for dim, infos in bin_stats.items():
            np.testing.assert_equal(infos["hist"], target_hists[dim])


class TestFileGraph(TestCase):
    """ Assert that the FileGraph still produces the same output. """
    @classmethod
    def setUpClass(cls):
        cls.proc = orcasong.core.FileGraph(
            max_n_hits=3,
            time_window=[0, 50],
            hit_infos=["pos_z", "time", "channel_id"],
            mc_info_extr=orcasong.mc_info_extr.get_real_data,
            det_file=DET_FILE,
            add_t0=True,
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
            '_i_event_info', '_i_group_info', '_i_y',
            'event_info', 'group_info', 'x', 'x_indices', 'y'})

    def test_x_title(self):
        self.assertEqual(self.f["x"].attrs["TITLE"].decode(), "pos_z, time, channel_id, is_valid")

    def test_x(self):
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
