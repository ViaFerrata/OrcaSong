from unittest import TestCase
import os
import h5py
import numpy as np
import toml
import orcasong.tools.make_data_split as mds

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
list_output_train = os.path.join("data_split_test_output", "conc_list_files", "test_list_train_0.txt")
#the scripts outputs
scripts_output_dir = os.path.join(test_data_dir, "data_split_test_output", "job_scripts")
concatenate_bash_script_train = os.path.join(scripts_output_dir, "concatenate_h5_test_list_train_0.sh")
concatenate_bash_script_val = os.path.join(scripts_output_dir, "concatenate_h5_test_list_validate_0.sh")
shuffle_bash_script_train = os.path.join(scripts_output_dir, "shuffle_h5_test_list_train_0.sh")
shuffle_bash_script_val = os.path.join(scripts_output_dir, "shuffle_h5_test_list_validate_0.sh")
#and the files that will be created from these scripts
concatenate_file = os.path.join("data_split_test_output", "data_split", "test_list_train_0.h5")

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
		cls.contents_concatenate_script = ['concatenate ' + list_output_train + ' --outfile ' + concatenate_file]
		cls.contents_shuffle_script = ['h5shuffle2 ' + concatenate_file + ' --max_ram 1000000000 \n']

		
		#create list_file_dir
		if not os.path.exists(list_file_dir):
			os.makedirs(list_file_dir)
	
	@classmethod
	def tearDownClass(cls):
		#remove the lists created
		os.remove(list_output_val)
		os.remove(list_output_train)
		os.remove(concatenate_bash_script_train)
		os.remove(concatenate_bash_script_val)
		os.remove(shuffle_bash_script_train)
		os.remove(shuffle_bash_script_val)
		os.removedirs(scripts_output_dir)
		os.removedirs(list_file_dir)
		os.removedirs(os.path.join(test_data_dir, "data_split_test_output", "logs"))
		os.removedirs(os.path.join(test_data_dir, "data_split_test_output", "data_split"))

		

	def test_read_keys_off_config(self):
		self.cfg = read_config(config_file)
		#get input groups and compare
		self.ip_group_keys = mds.get_all_ip_group_keys(self.cfg)
		self.assertSequenceEqual(self.ip_group_keys,self.input_categories_list)
		
	def test_get_filepath_and_n_events(self):	
		#repeat first 2 steps
		self.cfg = read_config(config_file)
		self.ip_group_keys = mds.get_all_ip_group_keys(self.cfg)
		
		self.cfg,self.n_evts_total = update_cfg(self.cfg)
		
		for key in self.ip_group_keys:
			self.assertIn(self.cfg[key]['fpaths'][0],self.file_path_list)
			self.assertIn(self.cfg[key]['n_evts'],self.n_events_list)
	
	def test_make_split(self):
		#main
		#repeat first 3 steps
		self.cfg = read_config(config_file)
		self.ip_group_keys = mds.get_all_ip_group_keys(self.cfg)
		self.cfg,self.n_evts_total = update_cfg(self.cfg)
		
		self.cfg['n_evts_total'] = self.n_evts_total
		mds.print_input_statistics(self.cfg, self.ip_group_keys)
		for key in self.ip_group_keys:
			mds.add_fpaths_for_data_split_to_cfg(self.cfg, key)
		mds.make_dsplit_list_files(self.cfg)
		
		#assert the single output lists
		assert os.path.exists(list_output_val) == 1
		with open(list_output_val) as f:
			for line in f:
				self.assertIn(line,self.file_path_list_val)
		f.close
		
		assert os.path.exists(list_output_train) == 1
		with open(list_output_train) as f2:
			for line in f2:
				self.assertIn(line,self.file_path_list)
		f2.close
		
	def test_make_concatenate_and_shuffle_scripts(self):
		#main
		#repeat first 4 steps
		self.cfg = read_config(config_file)
		self.ip_group_keys = mds.get_all_ip_group_keys(self.cfg)
		self.cfg,self.n_evts_total = update_cfg(self.cfg)
		
		self.cfg['n_evts_total'] = self.n_evts_total
		mds.print_input_statistics(self.cfg, self.ip_group_keys)
		for key in self.ip_group_keys:
			mds.add_fpaths_for_data_split_to_cfg(self.cfg, key)
		mds.make_dsplit_list_files(self.cfg)
			
		#create the bash job scripts and test their content		
		mds.make_concatenate_and_shuffle_scripts(self.cfg)
		
		assert os.path.exists(concatenate_bash_script_train) == 1
		with open(concatenate_bash_script_train) as f:
			for line in f:
				pass		#yay, awesome style! ^^
			last_line = line
			self.assertIn(last_line,self.contents_concatenate_script)
		f.close

		assert os.path.exists(shuffle_bash_script_train) == 1
		with open(shuffle_bash_script_train) as f2:
			for line in f2:
				pass
			last_line = line
			self.assertIn(last_line,self.contents_shuffle_script)
		f2.close

		
def update_cfg(cfg):
	
	''' Update the cfg with file paths and also return the total number of events'''
	 
	#get input groups and compare
	ip_group_keys = mds.get_all_ip_group_keys(cfg)
	os.chdir(test_data_dir)
	n_evts_total = 0
	for key in ip_group_keys:
		print('Collecting information from input group ' + key)
		cfg[key]['fpaths'] = mds.get_h5_filepaths(cfg[key]['dir'])
		cfg[key]['n_files'] = len(cfg[key]['fpaths'])
		cfg[key]['n_evts'], cfg[key]['n_evts_per_file_mean'], cfg[key]['run_ids'] = mds.get_number_of_evts_and_run_ids(cfg[key]['fpaths'], dataset_key='y')
		n_evts_total += cfg[key]['n_evts']
			
	return cfg,n_evts_total
	
def read_config(config_file):
	#decode config
	cfg = toml.load(config_file)
	cfg['toml_filename'] = config_file
	return cfg
		
