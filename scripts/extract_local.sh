#!/bin/bash
echo "before" $(which python)
source activate orcasong
echo "after" $(which python)
detector=/data/arca/KM3NeT_-00000001_20171212.detx

# inputfile=/data/arca/v6/raw_h5/mcv6.gsg_nutau-CCHEDIS-shower_1e2-1e8GeV.sirene.jte.jchain.aashower.${i}.h5
# outputfile=/data/arca/v6/ml_ready_triggered/ml_ready_triggered/ML_mcv6.gsg_nutau-CCHEDIS-shower_1e2-1e8GeV.sirene.jte.jchain.aashower.${i}.h5

inputfile=/data/arca/v6/raw_h5/mcv6.gsg_nue-CCHEDIS_1e2-1e8GeV.sirene.jte.jchain.aashower.${i}.h5
outputfile=/data/arca/v6/ml_ready_triggered/ML_mcv6.gsg_nue-CCHEDIS_1e2-1e8GeV.sirene.jte.jchain.aashower.${i}.h5

python /data/OrcaSong/scripts/extract.py $inputfile $detector $outputfile

