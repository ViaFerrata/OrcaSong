#!/bin/bash 

export DIR=/sps/km3net/users/gvermar/
export i=

#set environment

ccenv python 3.8.6



for i in {1..10}
do 
	qsub -V \
	-P P_km3net \
	-l ct=1:00:00 \
	-l vmem=40G \
	-l sps=1 \
	-l fsize=2G \
	-e ${DIR}/logs/ \
	-o ${DIR}/logs/ \
	$DIR/OrcaSong/extract.sh 
done
