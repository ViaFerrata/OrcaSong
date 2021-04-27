#!/bin/bash 

DIR=/sps/km3net/users/gvermar/

qsub -V \
     -P P_km3net \
     -l ct=3:00:00 \
     -l vmem=30G \
     -l sps=1 \
     -l fsize=100G \
     -e ${DIR}/logs/ \
     -o ${DIR}/logs/ \
     $DIR/OrcaSong/concatenatetriggered.sh
