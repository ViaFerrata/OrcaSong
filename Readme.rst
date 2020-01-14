OrcaSong: Generating DL images from KM3NeT data
===============================================

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

In this regard, OrcaSong is a project that produces KM3NeT event images based on the raw detector data.
This means that OrcaSong takes a datafile with (neutrino-) events and based on this data, it produces 2D/3D/4D 'images' (histograms).
Currently, only simulations with a hdf5 data format are supported as an input.

These event 'images' are required for some Deep Learning machine learning algorithms, e.g. Convolutional Neural Networks.

OrcaSong can be installed via pip by running::

    pip install orcasong

