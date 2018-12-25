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

  parse_input
  parser_check_input
  make_output_dirs
  calculate_bin_edges
  calculate_bin_edges_test
  get_file_particle_type
  EventSkipper
  skip_event
  data_to_images
  main


``orcasong.file_to_hits``: Extracting event information
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: orcasong.file_to_hits
  :no-members:
  :no-inherited-members:

.. currentmodule:: orcasong.file_to_hits

.. autosummary::
  :toctree: api

  get_primary_track_index
  get_time_residual_nu_interaction_mean_triggered_hits
  get_hits
  get_tracks
  EventDataExtractor


``orcasong.hits_to_histograms``: Making images based on the event info
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: orcasong.hits_to_histograms
  :no-members:
  :no-inherited-members:

.. currentmodule:: orcasong.hits_to_histograms

.. autosummary::
  :toctree: api

  get_time_parameters
  compute_4d_to_2d_histograms
  convert_2d_numpy_hists_to_pdf_image
  compute_4d_to_3d_histograms
  compute_4d_to_4d_histograms
  HistogramMaker