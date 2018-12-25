Getting started with OrcaSong
=============================

.. contents:: :local:

Introduction
------------

On this page, you can find a step by step introduction into the usage of OrcaSong.
The guide starts with some exemplary root simulation files made with jpp and ends with hdf5 event 'images' that can be used for deep neural networks.

Preprocessing
-------------

Let's suppose you have some KM3NeT simulation files in the ROOT dataformat, e.g.::

    /sps/km3net/users/kmcprod/JTE_NEMOWATER/withMX/muon-CC/3-100GeV/JTE.KM3Sim.gseagen.muon-CC.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016.99.root

The file above contains simulated charged-current muon neutrinos from the official 2016 23m ORCA production.
Now, we want to produce neutrino event images based on this data using OrcaSong.

Conversion from .root to .h5
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At first, we have to convert the files from the .root dataformat to a more usable one: hdf5.
For this purpose, we can use a tool called :code:`tohdf5` which is contained in the collaboration framework :code:`km3pipe`.
In order to use :code:`tohdf5`, you need to have loaded a jpp version first. A ready to use bash script for doing this can be found at::

    /sps/km3net/users/mmoser/setenvAA_jpp9_cent_os7.sh

Additionally, you need to have a python environment on Lyon, where you have installed km3pipe (e.g. use a pyenv).
Then, the usage of :code:`tohdf5` is quite easy::

    ~$: tohdf5 -o testfile.h5 /sps/km3net/users/kmcprod/JTE_NEMOWATER/withMX/muon-CC/3-100GeV/JTE.KM3Sim.gseagen.muon-CC.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016.99.root
    ++ tohdf5: Converting '/sps/km3net/users/kmcprod/JTE_NEMOWATER/withMX/muon-CC/3-100GeV/JTE.KM3Sim.gseagen.muon-CC.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016.99.root'...
    Pipeline and module initialisation took 0.002s (CPU 0.000s).
    loading root....  /afs/.in2p3.fr/system/amd64_sl7/usr/local/root/v5.34.23/
    loading aalib...  /pbs/throng/km3net/src/Jpp/v9.0.8454//externals/aanet//libaa.so
    ++ km3pipe.io.aanet.AanetPump: Reading metadata using 'JPrintMeta'
    WARNING ++ km3pipe.io.aanet.MetaParser: Empty metadata
    WARNING ++ km3pipe.io.aanet.AanetPump: No metadata found, this means no data provenance!
    --------------------------[ Blob     250 ]---------------------------
    --------------------------[ Blob     500 ]---------------------------
    --------------------------[ Blob     750 ]---------------------------
    --------------------------[ Blob    1000 ]---------------------------
    --------------------------[ Blob    1250 ]---------------------------
    --------------------------[ Blob    1500 ]---------------------------
    --------------------------[ Blob    1750 ]---------------------------
    --------------------------[ Blob    2000 ]---------------------------
    --------------------------[ Blob    2250 ]---------------------------
    --------------------------[ Blob    2500 ]---------------------------
    --------------------------[ Blob    2750 ]---------------------------
    --------------------------[ Blob    3000 ]---------------------------
    --------------------------[ Blob    3250 ]---------------------------
    EventFile io / wall time = 6.27259 / 73.9881 (8.47784 % spent on io.)
    ================================[ . ]================================
    ++ km3pipe.io.hdf5.HDF5Sink: HDF5 file written to: testfile.h5
    ============================================================
    3457 cycles drained in 75.842898s (CPU 70.390000s). Memory peak: 177.71 MB
      wall  mean: 0.021790s  medi: 0.019272s  min: 0.015304s  max: 2.823921s  std: 0.049242s
      CPU   mean: 0.020330s  medi: 0.020000s  min: 0.010000s  max: 1.030000s  std: 0.018179s
    ++ tohdf5: File '/sps/km3net/users/kmcprod/JTE_NEMOWATER/withMX/muon-CC/3-100GeV/JTE.KM3Sim.gseagen.muon-CC.3-100GeV-9.1E7-1bin-3.0gspec.ORCA115_9m_2016.99.root' was converted.

There are also some options that can be used with :code:`tohdf5`::

    ~$: tohdf5 -h
    Convert ROOT and EVT files to HDF5.

    Usage:
        tohdf5 [options] FILE...
        tohdf5 (-h | --help)
        tohdf5 --version

    Options:
        -h --help                       Show this screen.
        --verbose                       Print more output.
        --debug                         Print everything.
        -n EVENTS                       Number of events/runs.
        -o OUTFILE                      Output file (only if one file is converted).
        -j --jppy                       (Jpp): Use jppy (not aanet) for Jpp readout.
        --ignore-hits                   Don't read the hits.
        -e --expected-rows NROWS        Approximate number of events.  Providing a
                                        rough estimate for this (100, 1000000, ...)
                                        will greatly improve reading/writing speed
                                        and memory usage.
                                        Strongly recommended if the table/array
                                        size is >= 100 MB. [default: 10000]
        -t --conv-times-to-jte          Converts all MC times in the file to JTE

For now though, we will just stick to the standard conversion without any options.

After this conversion, you can investigate the data structure of the hdf5 file with the command :code:`ptdump`::

    ptdump -v testfile.h5
    / (RootGroup) 'KM3NeT'
    /event_info (Table(3457,), fletcher32, shuffle, zlib(5)) 'EventInfo'
      description := {
      "weight_w4": Float64Col(shape=(), dflt=0.0, pos=0),
      "weight_w3": Float64Col(shape=(), dflt=0.0, pos=1),
      "weight_w2": Float64Col(shape=(), dflt=0.0, pos=2),
      "weight_w1": Float64Col(shape=(), dflt=0.0, pos=3),
      "run_id": Int64Col(shape=(), dflt=0, pos=4),
      "timestamp": Int64Col(shape=(), dflt=0, pos=5),
      "nanoseconds": Int64Col(shape=(), dflt=0, pos=6),
      "mc_time": Float64Col(shape=(), dflt=0.0, pos=7),
      "event_id": Int64Col(shape=(), dflt=0, pos=8),
      "mc_id": Int64Col(shape=(), dflt=0, pos=9),
      "group_id": Int64Col(shape=(), dflt=0, pos=10)}
    ...

Hdf5 files are structured into "folders", in example the folder that is shown above is called "event_info".
The event_info is just a two dimensional numpy recarray with the shape (3457, 11), where for each event
important information is stored, e.g. the event_id or the run_id.

There is also a folder called "hits", which contains the photon hits of the detector for all events.
If you dig a little bit into the subfolders you can see that a lot of information is contained about these hits,
e.g. the hit time, but there is no XYZ position of the hits. The only information that you have is the dom_id and the
channel_id of a hit.

Calibrating the .h5 file
~~~~~~~~~~~~~~~~~~~~~~~~

In order to fix this, we can run another tool, :code:`calibrate`, that will add the pos_xyz information to the hdf5 datafile::

    calibrate /sps/km3net/users/mmoser/det_files/orca_115strings_av23min20mhorizontal_18OMs_alt9mvertical_v1.detx testfile.h5

As you can see, you need a .detx geometry file for this "calibration". Typically, you can find the path of this detx
file on the wiki page of the simulation production that you are using. This calibration step is optional, since OrcaSong
can also do it on the fly, using a .detx file.

At this point, we are now ready to start using OrcaSong for the generation of event images.


Usage of OrcaSong
-----------------

In order to use OrcaSong, you can just install it with :code:`pip`::

    ~/$: pip install orcasong

Before you can start to use OrcaSong, you need a .detx detector geometry file that corresponds to your input files.
OrcaSong is currently producing event "images" based on a 1 DOM / XYZ-bin assumption. This image generation is done
automatically, based on the number of bins (n_bins) for each dimension XYZ that you supply as an input and based on the
.detx file which contains the DOM positions.

If your .detx file is not contained in the OrcaSong/detx_files folder, please add it to the repository!
Currently, only the 115l ORCA 2016 detx file is available.

At this point, you're finally ready to use OrcaSong.
OrcaSong can be called from every directory by using the :code:`make_nn_images` command::

    ~/$: make_nn_images testfile.h5 geofile.detx

OrcaSong will then generate a hdf5 file with images that will be put in a "Results" folder at your current path.

The configuration options of OrcaSong can be found by calling the help::

    ~/$: make_nn_images -h
    Main OrcaSong code which takes raw simulated .h5 files and the corresponding .detx detector file as input in
    order to generate 2D/3D/4D histograms ('images') that can be used for CNNs.

    First argument: KM3NeT hdf5 simfile at JTE level.
    Second argument: a .detx file that is associated with the hdf5 file.

    The input file can be calibrated or not (e.g. contains pos_xyz of the hits) and the OrcaSong output is written
    to the current folder by default (otherwise use --o option).
    Makes only 4D histograms ('images') by default.

    Usage:
        data_to_images.py [options] FILENAME DETXFILE
        data_to_images.py (-h | --help)

    Options:
        -h --help                       Show this screen.

        -c CONFIGFILE                   Load all options from a config file (.toml format).

        --o OUTPUTPATH                  Path for the directory, where the OrcaSong output should be stored. [default: ./]

        --chunksize CHUNKSIZE           Chunksize (axis_0) that should be used for the hdf5 output of OrcaSong. [default: 32]

        --complib COMPLIB               Compression library that should be used for the OrcaSong output.
                                        All PyTables compression filters are available. [default: zlib]

        --complevel COMPLEVEL           Compression level that should be used for the OrcaSong output. [default: 1]

        --n_bins N_BINS                 Number of bins that are used in the image making for each dimension, e.g. (x,y,z,t).
                                        [default: 11,13,18,60]

        --det_geo DET_GEO               Which detector geometry to use for the binning, e.g. 'Orca_115l_23m_h_9m_v'.
                                        [default: Orca_115l_23m_h_9m_v]

        --do2d                          If 2D histograms, 'images', should be created.

        --do2d_plots                    If 2D pdf plot visualizations of the 2D histograms should be created, cannot be called if do2d=False.

        --do2d_plots_n N                For how many events the 2D plot visualizations should be made.
                                        OrcaSong will exit after reaching N events. [default: 10]

        --do3d                          If 3D histograms, 'images', should be created.

        --dont_do4d                     If 4D histograms, 'images', should NOT be created.

        --do4d_mode MODE                What dimension should be used in the 4D histograms as the 4th dim.
                                        Available: 'time', 'channel_id'. [default: time]

        --timecut_mode MODE             Defines what timecut mode should be used in hits_to_histograms.py.
                                        At the moment, these cuts are only optimized for ORCA 115l neutrino events!
                                        Currently available:
                                        'timeslice_relative': Cuts out the central 30% of the snapshot.
                                        'trigger_cluster': Cuts based on the mean of the triggered hits.
                                        The timespan for this cut can be chosen in --timecut_timespan.
                                        'None': No timecut.
                                        [default: trigger_cluster]

        --timecut_timespan TIMESPAN     Only used with timecut_mode 'trigger_cluster'.
                                        Defines the timespan of the trigger_cluster cut.
                                        Currently available:
                                        'all': [-350ns, 850ns] -> 20ns / bin (60 bins)
                                        'tight-1': [-250ns, 500ns] -> 12.5ns / bin
                                        'tight-2': [-150ns, 200ns] -> 5.8ns / bin
                                        [default: tight-1]

        --do_mc_hits                    If only the mc_hits (no BG hits!) should be used for the image processing.

        --data_cut_triggered            If non-triggered hits should be thrown away for the images.

        --data_cut_e_low E_LOW          Cut events that are lower than the specified threshold value in GeV.

        --data_cut_e_high E_HIGH        Cut events that are higher than the specified threshold value in GeV.

        --data_cut_throw_away FRACTION  Throw away a random fraction (percentage) of events. [default: 0.00]

        --prod_ident PROD_IDENT         Optional int identifier for the used mc production.
                                        This is useful, if you use events from two different mc productions,
                                        e.g. the 1-5GeV & 3-100GeV Orca 2016 MC. The prod_ident int will be saved in
                                        the 'y' dataset of the output file of OrcaSong. [default: 1]


Alternatively, they can also be found in the docs of the :code:`data_to_images()` function:

.. currentmodule:: orcasong.data_to_images
.. autosummary::
    data_to_images

Other than parsing every information to orcasong via the console, you can also load a .toml config file::

    ~/$: make_nn_images -c config.toml testfile.h5 geofile.detx

Please checkout the config.toml file in the main folder of the OrcaSong repo in order to get an idea about
the structure of the config file.

If anything is still unclear after this introduction just tell me in the deep_learning channel on chat.km3net.de or
write me an email at michael.m.moser@fau.de, such that I can improve this guide!




