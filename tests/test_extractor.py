import os
from unittest import TestCase
import tempfile
import numpy as np
import h5py
import orcasong.core
import orcasong.extractors as extractors


__author__ = "Daniel Guderian"


test_dir = os.path.dirname(os.path.realpath(__file__))
NEUTRINO_FILE = os.path.join(test_dir, "data", "mcv6_ORCA_test_rc.gsg_tauCClowE-CC_3-50GeV.km3sim.jterbr00008155.jorcarec.jsh.aanet.760.h5")
DET_FILE_NEUTRINO = os.path.join(test_dir, "data", "KM3NeT_00000049_20200707.detx")

NO_COMPLE_RECO_FILE = os.path.join(test_dir, "data", "arca_test_without_some_jmuon_recos.h5")
ARCA_DETX = os.path.join(test_dir, "data", "KM3NeT_-00000001_20171212.detx")


class TestStdRecoExtractor(TestCase):
    """ Assert that the neutrino info is extracted correctly. File has 18 events. """

    @classmethod
    def setUpClass(cls):
        #normal case, with complete recos
        cls.proc = orcasong.core.FileGraph(
            max_n_hits=3,
            time_window=[0, 50],
            hit_infos=["pos_z", "time", "channel_id"],
            extractor=extractors.get_neutrino_mc_info_extr(NEUTRINO_FILE),
            det_file=DET_FILE_NEUTRINO,
            add_t0=True,
            keep_event_info=True,
            fixed_length=True,
        )
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.outfile = os.path.join(cls.tmpdir.name, "binned.h5")
        cls.proc.run(infile=NEUTRINO_FILE, outfile=cls.outfile)
        cls.f = h5py.File(cls.outfile, "r")
        cls.reco_names_in_original_file = ["best_jshower","best_jmuon"]
        cls.reco_names = ["jshower","jmuon"]
        cls.quantities_to_test = ["dir_z","dir_x","pos_x"]
        cls.mc_quantities_to_test = ["dir_z","dir_x","vertex_pos_x"]
        
        #case where a few recos from jmuon are missing
        cls.proc = orcasong.core.FileGraph(
            max_n_hits=3,
            time_window=[0, 50],
            hit_infos=["pos_z", "time", "channel_id"],
            extractor=extractors.get_neutrino_mc_info_extr(NO_COMPLE_RECO_FILE),
            det_file=ARCA_DETX,
            add_t0=True,
            keep_event_info=True,
            fixed_length=True,
        )
        cls.outfile_arca = os.path.join(cls.tmpdir.name, "binned_arca.h5")
        cls.proc.run(infile=NO_COMPLE_RECO_FILE, outfile=cls.outfile_arca)
        cls.f_arca = h5py.File(cls.outfile_arca, "r")

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

    def test_y_contents(self):
    
        #the y from the processed file 
        y = self.f["y"][()]
        
        #the info from the original file
        orig_info = h5py.File(NEUTRINO_FILE,"r")
        
        orig_reco_info = orig_info["reco"]
        orig_mc_info = orig_info["mc_tracks"]
        
        #test a few reco parameter
        for i in range(len(self.reco_names_in_original_file)):
            for j in self.quantities_to_test:
                orig_reco_values = orig_reco_info[self.reco_names_in_original_file[i]][j]
            
                assert np.allclose(orig_reco_values,y[self.reco_names[i]+"_"+j])
        
        #test a few mc truth parameter
        for j in range(len(self.quantities_to_test)):
            orig_mc_values = orig_mc_info[self.quantities_to_test[j]][0] #only take fist, the primary
            
            assert np.allclose(orig_mc_values,y[self.mc_quantities_to_test[j]][0])
    
    def test_incomplete_reco_padding(self): 
        
        #make sure the x and y even in this case have the same length
        x = self.f_arca["x"]
        y = self.f_arca["y"]
        
        assert len(x) == len(y)
    
    
    
    
    
    
    