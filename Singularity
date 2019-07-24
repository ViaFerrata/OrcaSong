# sudo singularity build orcasong.simg Singularity
Bootstrap: docker
From: python:3.6

%files
. /orcasong

%post
cd /orcasong && make install

%runscript
exec /bin/bash "$@"

%startscript
exec /bin/bash "$@"

