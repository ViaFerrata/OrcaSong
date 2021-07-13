import os
from unittest import TestCase
import numpy as np
import orcasong.modules as modules
import km3pipe as kp


__author__ = 'Stefan Reck'


test_dir = os.path.dirname(os.path.realpath(__file__))
MUPAGE_FILE = os.path.join(test_dir, "data", "mupage.root.h5")
DET_FILE = os.path.join(test_dir, "data", "KM3NeT_-00000001_20171212.detx")


class TestModules(TestCase):
    def test_mc_info_maker(self):
        """ Test the mcinfo maker on some dummy data. """
        def extractor(blob):
            hits = blob["Hits"]
            return {"dom_id_0": hits.dom_id[0],
                    "time_2": hits.time[2]}

        in_blob = {
            "Hits": kp.Table({
                'dom_id': [2, 3, 3],
                'channel_id': [0, 1, 2],
                'time': [10.1, 11.2, 12.3]
            })
        }
        module = modules.McInfoMaker(
            extractor=extractor, store_as="test")
        out_blob = module.process(in_blob)

        self.assertSequenceEqual(list(out_blob.keys()), ["Hits", "test"])
        self.assertSequenceEqual(list(out_blob["test"].dtype.names),
                                 ('dom_id_0', 'time_2'))
        np.testing.assert_array_equal(out_blob["test"]["dom_id_0"],
                                      np.array([2, ], dtype="float64"))
        np.testing.assert_array_equal(out_blob["test"]["time_2"],
                                      np.array([12.3, ], dtype="float64"))

    def test_mc_info_maker_dtype(self):
        """ Test the mcinfo maker on some dummy data. """
        def extractor(blob):
            hits = blob["Hits"]
            return {"dom_id_0": hits.dom_id[0],
                    "time_2": hits.time[2]}

        in_blob = {
            "Hits": kp.Table({
                'dom_id': np.array([2, 3, 3], dtype="int8"),
                'time': np.array([10.1, 11.2, 12.3], dtype="float32"),
            })
        }
        module = modules.McInfoMaker(
            extractor=extractor, store_as="test", to_float64=False)
        out_blob = module.process(in_blob)

        np.testing.assert_array_equal(
            out_blob["test"]["dom_id_0"], np.array([2, ], dtype="int8"))
        np.testing.assert_array_equal(
            out_blob["test"]["time_2"], np.array([12.3, ], dtype="float32"))

    def test_event_skipper(self):
        def event_skipper(blob):
            # skip if true
            return blob["a"] == 42

        module = modules.EventSkipper(event_skipper=event_skipper)

        self.assertEqual(module.process({"a": 42}), None)
        self.assertEqual(module.process({"a": 25}), {"a": 25})


class TestTimePreproc(TestCase):
    def setUp(self):
        self.in_blob = {
            "Hits": kp.Table({
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            })
        }

        self.in_blob_mc = {
            "Hits": kp.Table({
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            }),
            "McHits": kp.Table({
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            })
        }

    def test_time_preproc_t0(self):
        module = modules.TimePreproc(
            add_t0=True, center_time=False)

        target = {
            "Hits": kp.Table({
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
            "Hits": kp.Table({
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
            "Hits": kp.Table({
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
            "Hits": kp.Table({
                'time': [-1.1, 0., 1.1],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            }),
            "McHits": kp.Table({
                'time': [-1.2, -0.2, 0.8],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            }),
        }
        out_blob = module.process(self.in_blob_mc)

        self.assertSetEqual(set(out_blob.keys()), set(target.keys()))
        np.testing.assert_array_almost_equal(
            np.array(out_blob["McHits"].view("<f8")),
            np.array(target["McHits"].view("<f8")))


class TestPointMaker(TestCase):
    def setUp(self):
        self.input_blob_1 = {
            "Hits": kp.Table({
                "x": [4, 5, 6],
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],}),
            "EventInfo": kp.Table({
                "pad": 1.
            })
        }

    def test_default_settings(self):
        pm = modules.PointMaker(
            max_n_hits=4)
        result = pm.process(self.input_blob_1)["samples"]
        self.assertTupleEqual(
            pm.finish()["hit_infos"], ("t0", "time", "x"))
        target = np.array(
            [[0.1, 1, 4],
             [0.2, 2, 5],
             [0.3, 3, 6]],
            dtype="float32")
        np.testing.assert_array_equal(result, target)

    def test_default_settings_fixed_length(self):
        pm = modules.PointMaker(
            max_n_hits=4, fixed_length=True)
        result = pm.process(self.input_blob_1)["samples"]
        self.assertTupleEqual(
            pm.finish()["hit_infos"], ("t0", "time", "x", "is_valid"))
        target = np.array(
            [[[0.1, 1, 4, 1],
              [0.2, 2, 5, 1],
              [0.3, 3, 6, 1],
              [0,   0, 0, 0]]], dtype="float32")
        np.testing.assert_array_equal(result, target)

    def test_input_blob_1_fixed_length(self):
        pm = modules.PointMaker(
            max_n_hits=4,
            hit_infos=("x", "time"),
            time_window=None,
            dset_n_hits=None,
            fixed_length=True,
        )
        result = pm.process(self.input_blob_1)["samples"]
        self.assertTupleEqual(
            pm.finish()["hit_infos"], ("x", "time", "is_valid"))
        target = np.array(
            [[[4, 1, 1],
              [5, 2, 1],
              [6, 3, 1],
              [0, 0, 0]]], dtype="float32")
        np.testing.assert_array_equal(result, target)

    def test_input_blob_1_max_n_hits_fixed_length(self):
        input_blob_long = {
            "Hits": kp.Table({
                "x": np.random.rand(1000).astype("float32"),
        })}
        result = modules.PointMaker(
            max_n_hits=10,
            hit_infos=("x",),
            time_window=None,
            dset_n_hits=None,
            fixed_length=True,
        ).process(input_blob_long)["samples"]

        self.assertSequenceEqual(result.shape, (1, 10, 2))
        self.assertTrue(all(
            np.isin(result[0, :, 0], input_blob_long["Hits"]["x"])))

    def test_input_blob_time_window_fixed_length(self):
        result = modules.PointMaker(
            max_n_hits=4,
            hit_infos=("x", "time"),
            time_window=[1, 2],
            dset_n_hits=None,
            fixed_length=True,
        ).process(self.input_blob_1)["samples"]
        target = np.array(
            [[[4, 1, 1],
              [5, 2, 1],
              [0, 0, 0],
              [0, 0, 0]]], dtype="float32")
        np.testing.assert_array_equal(result, target)

    def test_input_blob_time_window_nhits_fixed_length(self):
        result = modules.PointMaker(
            max_n_hits=4,
            hit_infos=("x", "time"),
            time_window=[1, 2],
            dset_n_hits="EventInfo",
            fixed_length=True,
        ).process(self.input_blob_1)["EventInfo"]
        print(result)
        self.assertEqual(result["n_hits_intime"], 2)


class TestImageMaker(TestCase):
    def test_2d_xt_binning(self):
        # (3 x 2) x-t binning
        bin_edges_list = [
            ["x", [3.5, 4.5, 5.5, 6.5]],
            ["time", [0.5, 2, 3.5]]
        ]

        module = modules.ImageMaker(bin_edges_list=bin_edges_list)
        in_blob = {
            "Hits": kp.Table({
                "x": [4, 5, 6],
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            })
        }

        target = {
            "Hits": kp.Table({
                "x": [4, 5, 6],
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            }),
            "samples": np.array([[
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
            np.array(out_blob["samples"]),
            np.array(target["samples"]))

    def test_unknown_field(self):
        # (3 x 2) x-t binning
        bin_edges_list = [
            ["aggg", [3.5, 4.5, 5.5, 6.5]],
            ["time", [0.5, 2, 3.5]]
        ]

        module = modules.ImageMaker(
            bin_edges_list=bin_edges_list)
        in_blob = {
            "Hits": kp.Table({
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

        module = modules.ImageMaker(bin_edges_list=bin_edges_list)
        in_blob = {
            "Hits": kp.Table({
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            })
        }

        target = {
            "Hits": kp.Table({
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            }),
            "samples": np.array([
                [1, ],
            ])
        }

        out_blob = module.process(in_blob)
        self.assertSetEqual(set(out_blob.keys()), set(target.keys()))
        np.testing.assert_array_almost_equal(
            np.array(out_blob["Hits"].view("<f8")),
            np.array(target["Hits"].view("<f8")))
        np.testing.assert_array_almost_equal(
            np.array(out_blob["samples"]),
            np.array(target["samples"]))

    def test_1d_binning_no_hits(self):
        # (1, ) t binning
        bin_edges_list = [
            ["time", [3.5, 4.5]]
        ]

        module = modules.ImageMaker(bin_edges_list=bin_edges_list)
        in_blob = {
            "Hits": kp.Table({
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            })
        }

        target = {
            "Hits": kp.Table({
                'time': [1., 2., 3.],
                "t0": [0.1, 0.2, 0.3],
                "triggered": [0, 1, 1],
            }),
            "samples": np.array([
                [0, ],
            ])
        }

        out_blob = module.process(in_blob)
        self.assertSetEqual(set(out_blob.keys()), set(target.keys()))
        np.testing.assert_array_almost_equal(
            np.array(out_blob["Hits"].view("<f8")),
            np.array(target["Hits"].view("<f8")))
        np.testing.assert_array_almost_equal(
            np.array(out_blob["samples"]),
            np.array(target["samples"]))


class TestDetApplier(TestCase):
    def setUp(self):
        self.deta = modules.DetApplier(
            det_file=DET_FILE,
            center_hits_to=(0, 5, None),
        )
        # self.pump = kp.io.HDF5Pump(filename=MUPAGE_FILE)
        # self.blob = self.pump[0]

    def test_cache_center(self):
        target = {"pos_x": 58.75166782379619, "pos_y": -21.5, "pos_z": 0}
        for d in ("pos_x", "pos_y", "pos_z"):
            np.testing.assert_array_almost_equal(target[d], self.deta._vector_shift[d])

    def test_shift_is_applied_to_hits(self):
        blob = {"Hits": {
            "pos_x": np.ones(3),
            "pos_y": np.ones(3)*2,
            "pos_z": np.ones(3)*3,
        }}
        target = {
            "pos_x": np.ones(3) * 59.75166782379619,
            "pos_y": np.ones(3) * -19.5,
            "pos_z": np.ones(3) * 3,
        }
        self.deta.shift_hits(blob)
        for d in ("pos_x", "pos_y", "pos_z"):
            np.testing.assert_array_almost_equal(target[d], blob["Hits"][d])


class TestBinningStatsMaker(TestCase):
    def test_it(self):
        # (3 x 2) x-t binning
        bin_edges_list = [
            ["x", [3.5, 4.5, 5.5, 6.5]],
            ["time", [0.5, 2, 3.5]],
            ["z", [1, 4]]
        ]

        in_blob = {
            "Hits": kp.Table({
                "x": [4, 5, 6, 6],
                'time': [1., 2., 3., 50],
                "z": [0, 3, 4, 5],

                "t0": [0.1, 0.2, 0.3, 0.4],
                "triggered": [0, 1, 1, 1],
            })
        }

        target = {
            'x': {
                'hist': np.array([0., 0., 0., 1., 0., 1.]),
                'hist_bin_edges': np.array([3.5, 4., 4.5, 5., 5.5, 6., 6.5]),
                'bin_edges': [3.5, 4.5, 5.5, 6.5],
                'cut_off': np.array([0., 0.])
            },
            'time': {
                'hist': np.array([0., 2.]),
                'hist_bin_edges': [0.5, 2, 3.5],
                'bin_edges': [0.5, 2, 3.5],
                'cut_off': np.array([0., 1.])
            },
            'z': {
                'hist': np.array([0., 2.]),
                'hist_bin_edges': np.array([1., 2.5, 4.]),
                'bin_edges': [1, 4],
                'cut_off': np.array([1., 1.])
            }
        }

        module = modules.BinningStatsMaker(
            bin_edges_list=bin_edges_list, res_increase=2)
        module.process(in_blob)
        output = module.finish()
        check_dicts_n_ray(output, target)


def check_dicts_n_ray(a, b):
    """ Check if dicts with dicts with ndarrays are equal. """
    if set(a.keys()) != set(b.keys()):
        raise KeyError("{} != {}".format(a.keys(), b.keys()))
    for key in a.keys():
        if set(a[key].keys()) != set(b[key].keys()):
            raise KeyError("{} != {}".format(a[key].keys(), b[key].keys()))
        for skey in a[key].keys():
            np.testing.assert_array_almost_equal(a[key][skey], b[key][skey])
