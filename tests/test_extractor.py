import os
from unittest import TestCase
import tempfile
import numpy as np
import h5py
import orcasong.core
import orcasong.extractors as extractors


__author__ = "Daniel Guderian"


test_dir = os.path.dirname(os.path.realpath(__file__))
NEUTRINO_FILE = os.path.join(test_dir, "data", "neutrino_file.h5")
DET_FILE_NEUTRINO = os.path.join(test_dir, "data", "neutrino_detector_file.detx")


class TestStdRecoExtractor(TestCase):
    """ Assert that the neutrino info is extracted correctly File has 18 events. """

    @classmethod
    def setUpClass(cls):
        cls.proc = orcasong.core.FileGraph(
            max_n_hits=3,
            time_window=[0, 50],
            hit_infos=["pos_z", "time", "channel_id"],
            extractor=extractors.get_neutrino_mc_info_extr(NEUTRINO_FILE),
            det_file=DET_FILE_NEUTRINO,
            add_t0=True,
            keep_event_info=True,
        )
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.outfile = os.path.join(cls.tmpdir.name, "binned.h5")
        cls.proc.run(infile=NEUTRINO_FILE, outfile=cls.outfile)
        cls.f = h5py.File(cls.outfile, "r")

    @classmethod
    def tearDownClass(cls):
        cls.f.close()
        cls.tmpdir.cleanup()

    def test_keys(self):
        self.assertSetEqual(
            set(self.f.keys()),
            {
                "_i_event_info",
                "_i_group_info",
                "_i_y",
                "event_info",
                "group_info",
                "x",
                "x_indices",
                "y",
            },
        )

    def test_y(self):
        y = self.f["y"][()]
        target = {
            "weight_w2": np.array(
                [
                    29650.0,
                    297100.0,
                    41450.0,
                    371400.0,
                    1101000000.0,
                    2757000.0,
                    15280000.0,
                    262800000.0,
                    22590.0,
                    24240.0,
                    80030.0,
                    3018000.0,
                    120600.0,
                    872200.0,
                    50440000.0,
                    21540.0,
                    42170.0,
                    25230.0,
                ]
            ),
            "n_gen": np.array(
                [
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                    60000.0,
                ]
            ),
            "dir_z": np.array(
                [
                    -0.896549,
                    -0.835252,
                    0.300461,
                    0.108997,
                    0.128445,
                    -0.543621,
                    -0.23205,
                    -0.297228,
                    0.694932,
                    0.73835,
                    -0.007682,
                    0.437847,
                    -0.126804,
                    0.153432,
                    -0.263229,
                    0.820217,
                    0.452473,
                    0.294217,
                ]
            ),
            "is_cc": np.array(
                [
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                    2.0,
                ]
            ),
            "std_dir_z": np.array(
                [
                    -0.923199825369434,
                    -0.6422689266782661,
                    0.38853917922036363,
                    -0.16690804339142448,
                    -0.01584853496341109,
                    -0.10151549881670698,
                    -0.0409694104272829,
                    -0.32964369874021787,
                    -0.3294926806601529,
                    0.6524241250799204,
                    -0.3899574246450216,
                    0.27872277417339086,
                    0.0019490791409933206,
                    0.20341370281708737,
                    -0.15739475718286297,
                    0.8040250543935723,
                    0.08772622550043882,
                    -0.7766722433951796,
                ]
            ),
            "std_energy": np.array(
                [
                    4.7187625606210775,
                    4.169818842606011,
                    1.0056373761749966,
                    5.908597073055873,
                    12.409377607517195,
                    7.566695371401163,
                    1.3546775620239864,
                    2.659528737837978,
                    1.0056373761749966,
                    2.1968321463948755,
                    1.4821714294894754,
                    10.135831333340658,
                    2.6003934443336765,
                    1.4492149732348223,
                    71.69167874147956,
                    8.094744120333358,
                    3.148088080484504,
                    1.0056373761749966,
                ]
            ),
        }
        for k, v in target.items():
            np.testing.assert_equal(y[k], v)
