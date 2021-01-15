OrcaSong: Preprocessing KM3NeT data for DL
==========================================

.. image:: https://badge.fury.io/py/orcasong.svg
    :target: https://badge.fury.io/py/orcasong

.. image:: https://git.km3net.de/ml/OrcaSong/badges/master/pipeline.svg
    :target: https://git.km3net.de/ml/OrcaSong/pipelines

.. image:: https://examples.pages.km3net.de/km3badges/docs-latest-brightgreen.svg
    :target: https://ml.pages.km3net.de/OrcaSong

.. image:: https://git.km3net.de/ml/OrcaSong/badges/master/coverage.svg
    :target: https://ml.pages.km3net.de/OrcaSong/coverage

.. image:: https://api.codacy.com/project/badge/Grade/1591b2d2d20e4c06a66cad99dc6aebe3
    :alt: Codacy Badge
    :target: https://www.codacy.com/app/sreck/OrcaSong?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=StefReck/OrcaSong&amp;utm_campaign=Badge_Grade


The documentation for OrcaSong can be found at https://ml.pages.km3net.de/OrcaSong!

OrcaSong is a part of the Deep Learning efforts of the neutrino telescope KM3NeT.  
Find more information about KM3NeT on http://www.km3net.org.

In this regard, OrcaSong is a project that preprocesses raw KM3NeT detector data
for the use with deep neural networks, making use of km3nets data processing
pipline km3pipe. Two different modes are available:

- For convolutional networks: produce n-dimensional 'images' (histograms)
- For graph networks: produce a list of nodes, each node representing infos about a hit in the detector

Currently, only simulations with a hdf5 data format are supported as an input.

OrcaSong can be installed via pip by running::

    pip install orcasong

A Singularity image of the latest stable version of OrcaSong is also provided.
You can download it from the km3net sftp server ``pi1139.physik.uni-erlangen.de``
in ``singularity/orcasong.sif``.
