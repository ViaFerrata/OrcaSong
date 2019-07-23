from unittest import TestCase
import orcasong.modules as modules
from km3pipe.dataclasses import Table
import numpy as np


__author__ = 'Stefan Reck'


class TestModules(TestCase):
    def test_mc_info_maker(self):
        """ Test the mcinfo maker on some dummy data. """
        def mc_info_extr(blob):
            hits = blob["Hits"]
            return {"dom_id_0": hits.dom_id[0],
                    "time_2": hits.time[2]}

        in_blob = {
            "Hits": Table({
                'dom_id': [2, 3, 3],
                'channel_id': [0, 1, 2],
                'time': [10.1, 11.2, 12.3]
            })
        }
        module = modules.McInfoMaker(
            mc_info_extr=mc_info_extr, store_as="test")
        out_blob = module.process(in_blob)

        self.assertSequenceEqual(list(out_blob.keys()), ["Hits", "test"])
        self.assertSequenceEqual(list(out_blob["test"].dtype.names),
                                 ('dom_id_0', 'time_2'))
        np.testing.assert_array_equal(out_blob["test"]["dom_id_0"],
                                      np.array([2, ]))
        np.testing.assert_array_equal(out_blob["test"]["time_2"],
                                      np.array([12.3, ]))


class TestTimePreproc(TestCase):
    def setUp(self):
        self.in_blob = {
            "Hits": Table({
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            })
        }

        self.in_blob_mc = {
            "Hits": Table({
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            }),
            "McHits": Table({
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            })
        }

    def test_time_preproc_t0(self):
        module = modules.TimePreproc(
            add_t0=True, center_time=False)

        target = {
            "Hits": Table({
                'time': [1.1, 2.2, 3.3],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            })
        }

        out_blob = module.process(self.in_blob)

        self.assertSetEqual(set(out_blob.keys()), set(target.keys()))
        np.testing.assert_array_equal(np.array(out_blob["Hits"]),
                                      np.array(target["Hits"]))

    def test_time_preproc_center(self):
        module = modules.TimePreproc(
            add_t0=False, center_time=True)

        target = {
            "Hits": Table({
                'time': [-1., 0., 1.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            })
        }

        out_blob = module.process(self.in_blob)

        self.assertSetEqual(set(out_blob.keys()), set(target.keys()))
        np.testing.assert_array_equal(np.array(out_blob["Hits"]),
                                      np.array(target["Hits"]))

    def test_time_preproc_t0_and_center(self):
        module = modules.TimePreproc(
            add_t0=True, center_time=True)

        target = {
            "Hits": Table({
                'time': [-1.1, 0., 1.1],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            })
        }

        out_blob = module.process(self.in_blob)

        self.assertSetEqual(set(out_blob.keys()), set(target.keys()))
        np.testing.assert_array_almost_equal(
            np.array(out_blob["Hits"].view("<f8")),
            np.array(target["Hits"].view("<f8")))

    def test_time_preproc_mchits_t0_and_center(self):
        module = modules.TimePreproc(
            add_t0=True, center_time=True)

        target = {
            "Hits": Table({
                'time': [-1.1, 0., 1.1],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            }),
            "McHits": Table({
                'time': [-1.1, 0., 1.1],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            }),
        }
        out_blob = module.process(self.in_blob_mc)

        self.assertSetEqual(set(out_blob.keys()), set(target.keys()))
        np.testing.assert_array_almost_equal(
            np.array(out_blob["McHits"].view("<f8")),
            np.array(target["McHits"].view("<f8")))


class TestImageMaker(TestCase):
    def test_2d_xt_binning(self):
        # (3 x 2) x-t binning
        bin_edges_list = [
            ["x", [3.5, 4.5, 5.5, 6.5]],
            ["time", [0.5, 2, 3.5]]
        ]

        module = modules.ImageMaker(
            bin_edges_list=bin_edges_list, store_as="histogram")
        in_blob = {
            "Hits": Table({
                "x": [4, 5, 6],
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            })
        }

        target = {
            "Hits": Table({
                "x": [4, 5, 6],
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            }),
            "histogram": np.array([[
                [1, 0],
                [0, 1],
                [0, 1],
            ]])
        }

        out_blob = module.process(in_blob)
        self.assertSetEqual(set(out_blob.keys()), set(target.keys()))
        np.testing.assert_array_almost_equal(
            np.array(out_blob["Hits"].view("<f8")),
            np.array(target["Hits"].view("<f8")))
        np.testing.assert_array_almost_equal(
            np.array(out_blob["histogram"]),
            np.array(target["histogram"]))

    def test_unknown_field(self):
        # (3 x 2) x-t binning
        bin_edges_list = [
            ["aggg", [3.5, 4.5, 5.5, 6.5]],
            ["time", [0.5, 2, 3.5]]
        ]

        module = modules.ImageMaker(
            bin_edges_list=bin_edges_list, store_as="histogram")
        in_blob = {
            "Hits": Table({
                "x": [4, 5, 6],
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            })
        }

        with self.assertRaises(ValueError):
            module.process(in_blob)

    def test_1d_binning(self):
        # (1, ) t binning
        bin_edges_list = [
            ["time", [2.5, 3.5]]
        ]

        module = modules.ImageMaker(
            bin_edges_list=bin_edges_list, store_as="histogram")
        in_blob = {
            "Hits": Table({
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            })
        }

        target = {
            "Hits": Table({
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            }),
            "histogram": np.array([
                [1, ],
            ])
        }

        out_blob = module.process(in_blob)
        self.assertSetEqual(set(out_blob.keys()), set(target.keys()))
        np.testing.assert_array_almost_equal(
            np.array(out_blob["Hits"].view("<f8")),
            np.array(target["Hits"].view("<f8")))
        np.testing.assert_array_almost_equal(
            np.array(out_blob["histogram"]),
            np.array(target["histogram"]))

    def test_1d_binning_no_hits(self):
        # (1, ) t binning
        bin_edges_list = [
            ["time", [3.5, 4.5]]
        ]

        module = modules.ImageMaker(
            bin_edges_list=bin_edges_list, store_as="histogram")
        in_blob = {
            "Hits": Table({
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            })
        }

        target = {
            "Hits": Table({
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            }),
            "histogram": np.array([
                [0, ],
            ])
        }

        out_blob = module.process(in_blob)
        self.assertSetEqual(set(out_blob.keys()), set(target.keys()))
        np.testing.assert_array_almost_equal(
            np.array(out_blob["Hits"].view("<f8")),
            np.array(target["Hits"].view("<f8")))
        np.testing.assert_array_almost_equal(
            np.array(out_blob["histogram"]),
            np.array(target["histogram"]))
