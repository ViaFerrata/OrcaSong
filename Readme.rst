OrcaSong: Preprocessing KM3NeT data for DL
==========================================

.. image:: https://badge.fury.io/py/orcasong.svg
    :target: https://badge.fury.io/py/orcasong

.. image:: https://git.km3net.de/ml/OrcaSong/badges/master/pipeline.svg
    :target: https://git.km3net.de/ml/OrcaSong/pipelines

.. image:: https://git.km3net.de/examples/km3badges/-/raw/master/docs-latest-brightgreen.svg
    :target: https://ml.pages.km3net.de/OrcaSong

.. image:: https://git.km3net.de/ml/OrcaSong/badges/master/coverage.svg
    :target: https://ml.pages.km3net.de/OrcaSong/coverage

OrcaSong is a project for preprocessing raw KM3NeT ORCA or ARCA event data
for the use with deep neural networks, making use of km3nets data processing
pipline km3pipe. Two different modes are available:

- For convolutional networks: produce n-dimensional 'images' (histograms)
- For graph networks: produce a list of nodes, each node representing infos about a hit in the detector

The input to Orcasong are offline/aanet root files, and the output are "DL" files
in the hdf5 format, which can e.g. be used by the OrcaNet software.
For more info, read the documentation here https://ml.pages.km3net.de/OrcaSong!

OrcaSong is a part of the Deep Learning efforts of the neutrino telescope KM3NeT.
Find more information about KM3NeT on http://www.km3net.org.

OrcaSong can be installed via pip by running::

    pip install orcasong

You can get a list of all the bash commands in orcasong by typing::

    orcasong --help

Containerization
----------------
The easiest way to run OrcaSong is with singularity.
A Singularity image of the latest stable version of OrcaSong
is automatically uploaded to our sftp server.
Download it e.g. via::

    wget http://pi1139.physik.uni-erlangen.de/singularity/orcasong_v???.sif

where v??? is the version, e.g. orcasong_v4.3.2.sif.
Run it e.g. via::

    singularity shell orcasong_v???.sif


