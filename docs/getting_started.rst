Getting started
===============

On this page, you can find a step by step introduction of how to prepare offline/aanet
root files for deep learning.

.. contents:: :local:


Step 1: From root aanet files to h5 aanet files
-----------------------------------------------
Convert offline files (aka aanet files) from root format to h5 format using
the 'h5extractf' command of km3pipe like so::

    h5extractf aanet_file.root

.. note::
    'h5extractf' is still a prototype, please report if there are any issues.
    There is also a (extremely slow) legacy version available called 'h5extract'.


Step 2: From h5 aanet files to h5 DL files
------------------------------------------
Produce DL h5 files from the aanet h5 files using OrcaSong.
You can either produce images or graphs.
It is easiest to use a config file for setting up all the options.
See here https://git.km3net.de/ml/OrcaSong/-/blob/master/examples/orcasong_example.toml for an
example config file.
You can use it via the command line like this::

    orcasong run aanet_file.h5 orcasong_config.toml --detx detector.detx


For some examples of config files you can check out the git repo here
https://git.km3net.de/ml/OrcaSong/-/tree/master/configs .
These can be loaded from the command line by using the prefix
``orcasong:`` before the filename, e.g. ``orcasong:bundle_ORCA4_data_v5-40.toml``.
Alternatively, you can use the python frontend of orcasong.
See :ref:`orcasong_page` for instructions on how to do this.

The resulting DL h5 files can already be used as input for networks!

Step 3: Concatenate
-------------------
Mandatory for training files, recommended for everything else.
Concatenate the dl files of inidividual (mc-) runs into a few, large files.
This makes it easier to use, and allows to shuffle them in step 4.
See :ref:`concatenate` for details.

.. note::
    Make sure that your training dataset is as random as possible.
    E.g., if you have runs from a given time period, don't use the first
    X runs for your training set. Instead, choose runs randomly over
    the whole period.

.. note::
    For mixing e.g. neutrinos and muon, a list with all DL files that should
    go into one specific file
    can be produced with :ref:`make_data_split`.

Step 4: Shuffle
---------------
Only necessary for training files!
Shuffle the order of events in a h5 file on an event by event basis.
See :ref:`shuffle` for details.
