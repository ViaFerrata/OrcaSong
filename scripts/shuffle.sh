#!/bin/bash

source /pbs/home/a/adomi/mypython/bin/activate

DIR=/sps/km3net/users/adomi/GNNs

echo "START"

h5shuffle2 --output_file $DIR/training/Muons_vs_Neutrinos_shuffled2.h5 $DIR/training/Muons_vs_Neutrinos_shuffled2.h5
#h5shuffle --output_file $DIR/training/Muons_vs_Neutrinos_shuffled.h5 $DIR/training/Muons_vs_Neutrinos.h5

echo "END"
