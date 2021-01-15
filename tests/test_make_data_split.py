from unittest import TestCase
import os
import h5py
import numpy as np
from orcasong.tools.make_data_split import *

__author__ = 'Daniel Guderian'

test_dir = os.path.dirname(os.path.realpath(__file__))
mupage = os.path.join(test_dir, "data", "mupage.root.h5")
neutrino_file = os.path.join(test_dir, "data", "neutrino.h5")
config_file = os.path.join(test_dir, "data", "test_make_data_split_config.toml")

#no idea how to tbh...