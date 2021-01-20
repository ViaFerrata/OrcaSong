from unittest import TestCase
import os
import h5py
import numpy as np
from orcasong.tools.make_data_split import *

__author__ = 'Daniel Guderian'

test_dir = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(test_dir, "data")
#these are files that were processed with orcasong
mupage_file = os.path.join(test_data_dir, "processed_data_muon", "processed_graph_muon.h5")
neutrino_file = os.path.join(test_data_dir,"processed_data_neutrino", "processed_graph_neutrino.h5")
#config file containing 2 input groups
config_file = os.path.join(test_data_dir, "test_make_data_split_config.toml")
#the list files that will be created
list_file_dir = os.path.join(test_data_dir, "data_split_test_output", "conc_list_files")
list_output_val = os.path.join(list_file_dir, "test_list_validate_0.txt")
list_output_train = os.path.join(list_file_dir, "test_list_train_0.txt")


class TestMakeDataSplit(TestCase):
	
	''' Runs the make_data_split like in the use case. At the end, the created lists are checked.'''
	
	@classmethod
	def setUpClass(cls):
		#the expected lists to compare to
		cls.input_categories_list = ["neutrino","muon"]
		#include name with linebreak as they will look like this in the final files
		cls.file_path_list = ['processed_data_muon/processed_graph_muon.h5','processed_data_neutrino/processed_graph_neutrino.h5',
							  'processed_data_muon/processed_graph_muon.h5\n','processed_data_neutrino/processed_graph_neutrino.h5\n']
		cls.file_path_list_val = ['processed_data_neutrino/processed_graph_neutrino.h5','processed_data_neutrino/processed_graph_neutrino.h5\n']
		cls.n_events_list = [18,3]
		#create list_file_dir
		if not os.path.exists(list_file_dir):
			os.makedirs(list_file_dir)
	
	@classmethod
	def tearDownClass(cls):
		#remove the lists created
		os.remove(list_output_val)
		os.remove(list_output_train)
		os.removedirs(list_file_dir)

	def test_read_keys_off_config(self):
		#decode config
		self.cfg = toml.load(config_file)
		self.cfg['toml_filename'] = config_file
		#get input groups and compare
		self.ip_group_keys = get_all_ip_group_keys(self.cfg)
		self.assertSequenceEqual(self.ip_group_keys,self.input_categories_list)
		
	def test_get_filepath_and_n_events(self):	
		os.chdir(test_data_dir)
		self.n_evts_total = 0
		for key in self.ip_group_keys:
			print('Collecting information from input group ' + key)
			self.cfg[key]['fpaths'] = get_h5_filepaths(self.cfg[key]['dir'])
			self.cfg[key]['n_files'] = len(self.cfg[key]['fpaths'])
			self.cfg[key]['n_evts'], self.cfg[key]['n_evts_per_file_mean'], self.cfg[key]['run_ids'] = get_number_of_evts_and_run_ids(self.cfg[key]['fpaths'], dataset_key='y')
			self.n_evts_total += self.cfg[key]['n_evts']
		
			self.assertIn(self.cfg[key]['fpaths'][0],self.file_path_list)
			self.assertIn(self.cfg[key]['n_evts'],self.n_events_list)
	
	def test_make_split(self):
		#main
		self.cfg['n_evts_total'] = self.n_evts_total
		print_input_statistics(self.cfg, self.ip_group_keys)
		for key in self.ip_group_keys:
			add_fpaths_for_data_split_to_cfg(self.cfg, key)
		make_dsplit_list_files(self.cfg)
		
		#assert the single output lists
		with open(list_output_val) as f:
			for line in f:
				self.assertIn(line,self.file_path_list_val)
		f.close
		with open(list_output_train) as f2:
			for line in f2:
				self.assertIn(line,self.file_path_list)
		f.close
		
		
