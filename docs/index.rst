.. OrcaSong documentation master file
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |vspace| raw:: latex

   \vspace{1cm}

.. image:: _static/orcasong_wide_transparent.png
   :height: 142px

|vspace|

Welcome to OrcaSong's documentation!
====================================

.. image:: https://git.km3net.de/ml/OrcaSong/badges/master/build.svg
    :target: https://git.km3net.de/ml/OrcaSong/pipelines

| OrcaSong is a part of the Deep Learning efforts for the neutrino telescope KM3NeT.
| Find more information about KM3NeT on http://www.km3net.org.

In this regard, OrcaSong is a project that produces KM3NeT event images based on the raw detector data.
This means that OrcaSong takes a datafile with (neutrino-) events and based on this data, it produces 2D/3D/4D 'images' (histograms).
Currently, only simulations with a hdf5 data format are supported as an input.
These event 'images' are required for some Deep Learning machine learning algorithms, e.g. Convolutional Neural Networks.

As of now, only ORCA detector simulations are supported, but ARCA geometries can be easily implemented as well.

The main code for generating the images is located in orcanet/data_to_images.py.
If the simulated hdf5 files are not calibrated yet, you need to specify the directory of a .detx file in 'data_to_images.py'.

As of now, the documentation contains a small introduction to get started and and a complete API documentation.
Please feel free to contact me or just open an issue on Gitlab / Github if you have any suggestions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
