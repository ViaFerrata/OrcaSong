#!/bin/bash

source /sps/km3net/users/gvermar/OrcaSong/mypython/bin/activate

detector=/pbs/throng/km3net/detectors/KM3NeT_-00000001_20171212.detx

# inputfile=/sps/km3net/users/gvermar/arca/v6/raw_h5/mcv6.gsg_nutau-CCHEDIS-shower_1e2-1e8GeV.sirene.jte.jchain.aashower.${i}.h5
# outputfile=/sps/km3net/users/gvermar/arca/v6/ml_ready_le1e5cut/ML_mcv6.gsg_nutau-CCHEDIS-shower_1e2-1e8GeV.sirene.jte.jchain.aashower.${i}.h5

inputfile=/sps/km3net/users/gvermar/arca/v6/raw_h5/mcv6.gsg_nue-CCHEDIS_1e2-1e8GeV.sirene.jte.jchain.aashower.${i}.h5
outputfile=/sps/km3net/users/gvermar/arca/v6/ml_ready_le1e5cut/ML_mcv6.gsg_nue-CCHEDIS_1e2-1e8GeV.sirene.jte.jchain.aashower.${i}.h5

python ${DIR}/OrcaSong/extract.py $inputfile $detector $outputfile

