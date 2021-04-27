#!/bin/bash 

cd

. mypython/bin/activate

DIR=/sps/km3net/users/adomi/GNNs/

qsub -V \
     -P P_km3net \
     -l ct=72:00:00 \
     -l vmem=30G \
     -l sps=1 \
     -l fsize=100G \
     -e ${DIR}/logs/ \
     -o ${DIR}/logs/ \
     $DIR/OrcaSong/shuffle.sh 
