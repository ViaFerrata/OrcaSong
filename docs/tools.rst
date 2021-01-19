OrcaSong Tools
==============

Orcasong comes with some tools to further process data.

.. _make_data_split:

Make_data_split
---------------

Create datasets for different tasks (like classification or regression) from the files resulting from OrcaSong, based on the run_id. This is particularly helpful for a run-by-run data analysis or to generate equally large datasets per class. A toml config is used, in which the directories and ranges of runs to be considered can be specified, as well as the subdivision into training and validation sets. Detailed descriptions for the options available can be found in the example config in the subfolder make_data_split_configs. As output, a list in txt format with the filepaths belonging to one set is created that can be passed to the concatenate for creating one single file out of the many. 

Can be used via the commandline::

    orcasong/tools/make_data_split.py config.toml


.. _concatenate:

Concatenate
-----------

Concatenate files resulting from OrcaSong, i.e. merge some h5 files
into a single, bigger one. The resulting file can still be read in with
km3pipe. The input can also be a txt file like from make_data_split.

Can be used via the commandline like so::

    concatenate --help

or import as:

.. code-block:: python

    from orcasong.tools import FileConcatenator
    
    fc = FileConcatenator.from_list(input_file_list_from_make_data_split.txt)
    
    fc.concatenate(output_filepath_concat)
   
.. _shuffle:

Shuffle
-------

Shuffle an h5 file using km3pipe.

Can be used via the commandline like so::

    h5shuffle --help

or import function for general postprocessing:

.. code-block:: python

    from orcasong.tools.postproc import postproc_file
    
    postproc_file(output_filepath_concat)

There is also a newer, faster version 2:

.. code-block:: python

    from orcasong.tools.shuffle2 import shuffle_v2
    
    shuffle_v2(output_filepath_concat)
    


