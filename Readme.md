## Generating DL images based on KM3NeT-ORCA neutrino simulation data 

OrcaSong is a project that produces 2D/3D/4D histograms ('images') for deep neural networks based on raw MC h5 files.
Currently, only ORCA detector simulations are supported, but ARCA geometries can be easily implemented as well.

The main code for generating the images is located in orcanet/h5_data_to_h5_input.py. <br>
If the simulated .h5 files are not calibrated yet, you need to specify the directory of a .detx file in 'h5_data_to_h5_input.py'.

Currently, a bin size of 11x13x18x60 (x/y/z/t) is used for the final ORCA detector layout. 
