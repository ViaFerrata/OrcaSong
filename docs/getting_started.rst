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

    ~/$: make_nn_images testfile.h5 geofile.detx configfile.toml

OrcaSong will then generate a hdf5 file with images that will be put in a "Results" folder at the path that
you've specified in the configfile current path.
Please checkout the default_config.toml file in the orcasong folder of the OrcaSong repo in order to get an idea about
the structure of the config files.

All available configuration options of OrcaSong can be found in /orcasong/default_config::

    --- Documentation for every config parameter that is available ---

    None arguments should be written as string: 'None'

    Parameters
    ----------
    output_dirpath : str
        Full path to the directory, where the orcasong output should be stored.
    chunksize : int
        Chunksize (along axis_0) that is used for saving the OrcaSong output to a .h5 file.
    complib : str
        Compression library that is used for saving the OrcaSong output to a .h5 file.
        All PyTables compression filters are available, e.g. 'zlib', 'lzf', 'blosc', ... .
    complevel : int
        Compression level for the compression filter that is used for saving the OrcaSong output to a .h5 file.
    n_bins : tuple of int
        Declares the number of bins that should be used for each dimension, e.g. (x,y,z,t).
        The option should be written as string, e.g. '11,13,18,60'.
    det_geo : str
        Declares what detector geometry should be used for the binning. E.g. 'Orca_115l_23m_h_9m_v'.
    do2d : bool
        Declares if 2D histograms, 'images', should be created.
    do2d_plots : bool
        Declares if pdf visualizations of the 2D histograms should be created, cannot be called if do2d=False.
    do2d_plots_n: int
        After how many events the event loop will be stopped (making the 2d plots in do2d_plots takes long time).
    do3d : bool
        Declares if 3D histograms should be created.
    do4d : bool
        Declares if 4D histograms should be created.
    do4d_mode : str
        If do4d is True, what should be used as the 4th dim after xyz.
        Currently, only 'time' and 'channel_id' are available.
    prod_ident : int
        Optional int identifier for the used mc production.
        This is e.g. useful, if you use events from two different mc productions, e.g. the 1-5GeV & 3-100GeV Orca 2016 MC.
        In this case, the events are not fully distinguishable with only the run_id and the event_id!
        In order to keep a separation, an integer can be set in the event_track for all events, such that they stay distinguishable.
    timecut_mode : str
        Defines what timecut should be used in hits_to_histograms.py.
        Currently available:
        'timeslice_relative': Cuts out the central 30% of the snapshot. The value of timecut_timespan doesn't matter in this case.
        'trigger_cluster': Cuts based on the mean of the triggered hits.
        'None': No timecut. The value of timecut_timespan doesn't matter in this case.
    timecut_timespan : str/None
        Defines what timespan should be used if a timecut is applied. Only relevant for timecut_mode = 'trigger_cluster'.
        Currently available:
        'all': [-350ns, 850ns] -> 20ns / bin (if e.g. 60 timebins)
        'tight-0': [-450ns, 500ns] -> 15.8ns / bin (if e.g. 60 timebins)
        'tight-1': [-250ns, 500ns] -> 12.5ns / bin (if e.g. 60 timebins)
        'tight-2': [-150ns, 200ns] -> 5.8ns / bin (if e.g. 60 timebins)
    do_mc_hits : bool
        Declares if hits (False, mc_hits + BG) or mc_hits (True) should be processed.
    data_cut_triggered : bool
        Cuts away hits that haven't been triggered.
    data_cut_e_low : float
        Cuts away events that have an energy lower than data_cut_e_low.
    data_cut_e_high : float
        Cuts away events that have an energy higher than data_cut_e_high.
    data_cut_throw_away : float
        Cuts away random events with a certain probability (1: 100%, 0: 0%).
    flush_freq : int
        After how many events the accumulated output should be flushed to the harddisk.
        A larger value leads to a faster orcasong execution, but it increases the RAM usage as well.

    --- Documentation for every config parameter that is available ---





If anything is still unclear after this introduction just tell me in the deep_learning channel on chat.km3net.de or
write me an email at michael.m.moser@fau.de, such that I can improve this guide!




