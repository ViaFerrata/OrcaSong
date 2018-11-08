API Reference
=============

.. contents:: :local:

OrcaSong: Main Framework
------------------------

``orcasong.data_to_images``: Main code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: orcasong.data_to_images
  :no-members:
  :no-inherited-members:

.. currentmodule:: orcasong.data_to_images

.. autosummary::
  :toctree: api

  calculate_bin_edges
  calculate_bin_edges_test
  main
  parse_input


``orcasong.file_to_hits``: Extracting event information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: orcasong.file_to_hits
  :no-members:
  :no-inherited-members:

.. currentmodule:: orcasong.file_to_hits

.. autosummary::
  :toctree: api

  get_event_data
  get_primary_track_index
  get_time_residual_nu_interaction_mean_triggered_hits


``orcasong.hits_to_histograms``: Making images based on the event info
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: orcasong.hits_to_histograms
  :no-members:
  :no-inherited-members:

.. currentmodule:: orcasong.hits_to_histograms

.. autosummary::
  :toctree: api

  compute_4d_to_2d_histograms
  compute_4d_to_3d_histograms
  compute_4d_to_4d_histograms
  convert_2d_numpy_hists_to_pdf_image
  get_time_parameters


``orcasong.histograms_to_files``: Saving the images to a h5 file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: orcasong.histograms_to_files
  :no-members:
  :no-inherited-members:


.. currentmodule:: orcasong.histograms_to_files

.. autosummary::
  :toctree: api

  store_histograms_as_hdf5