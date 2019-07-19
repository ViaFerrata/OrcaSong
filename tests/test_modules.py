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
        np.testing.assert_array_equal(np.array(out_blob["Hits"]),
                                      np.array(target["Hits"]))

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

        out_blob = module.process(self.in_blob)

        self.assertSetEqual(set(out_blob.keys()), set(target.keys()))
        np.testing.assert_array_equal(np.array(out_blob["McHits"]),
                                      np.array(target["McHits"]))
