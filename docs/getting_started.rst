Getting started
===============

On this page, you can find a step by step introduction of how to prepare offline/aanet
root files for deep learning.

.. contents:: :local:


Step 1: From root aanet files to h5 aanet files
-----------------------------------------------
Convert offline files (aka aanet files) from root format to h5 format using
the 'h5extract' command of km3pipe like so::

    h5extract filename.root

.. note::
    This has to be done only once for each file. Check if somebody did this
    already and has put it on sps somewhere. If not, consider putting it on sps
    yourself and let people know.


Step 2: From h5 aanet files to DL files
---------------------------------------
Produce DL files from the aanet h5 files using OrcaSong.
You can either produce images or graphs. See :ref:`orcasong_page` for
instructions on how to do this.

The resulting DL files can already be used as input for networks!

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


Step 4: Shuffle
---------------
Only necessary for training files!
Shuffle the order of events in a h5 file on an event by event basis.
See :ref:`shuffle` for details.
