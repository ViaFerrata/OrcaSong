#!/usr/bin/python -i
from orcasong.core import TriggeredFileGraph
from orcasong.extractors import (
    get_muon_mc_info_extr,
    get_neutrino_mc_info_extr,
    get_real_data_info_extr,
    get_random_noise_mc_info_extr,
)
import numpy as np
import sys
import os


inputfile = str(sys.argv[1])
detectorfile = str(sys.argv[2])
outputfile = str(sys.argv[3])


# fg = FileGraph(max_n_hits=5000,extractor=get_muon_mc_info_extr(inputfile),det_file=detectorfile,keep_event_info = True)
# fg = FileGraph(max_n_hits=5000,extractor=get_muon_mc_info_extr(inputfile),det_file=detectorfile,
#              keep_event_info = True, time_window = [-1000, +7500])


def skip_low_energy(blob):
    return blob["mc_info"]["energy"] < 1e5


fg = TriggeredFileGraph(
    max_n_hits=5000,
    extractor=get_neutrino_mc_info_extr(inputfile),
    det_file=detectorfile,
    keep_event_info=True,
)

fg.run(inputfile, outputfile)
