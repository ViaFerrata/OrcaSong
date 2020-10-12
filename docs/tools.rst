OrcaSong Tools
==============

Orcasong comes with some tools to further process data.

Concatenate
-----------

Concatenate files resulting from OrcaSong, i.e. merge some h5 files
into a single, bigger one. The resulting file can still be read in with
km3pipe.

Can be used via the commandline like so::

    concatenate --help

or import as:

.. code-block:: python

    from orcasong.tools import FileConcatenator


Shuffle
-------

Shuffle an h5 file using km3pipe.

Can be used via the commandline like so::

    h5shuffle --help

or import function for general postprocessing:

.. code-block:: python

    from orcasong.tools.postproc import postproc_file

