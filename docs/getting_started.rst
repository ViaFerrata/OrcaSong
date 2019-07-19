Getting started
===============

.. contents:: :local:

Introduction
------------

On this page, you can find a step by step introduction of how to prepare
root files for OrcaSong.
The guide starts with some exemplary root simulation files made with jpp and
ends with hdf5 files ready for the use with OrcaSong.

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

In order to fix this, the data needs to be calibrated.
This can be done in two ways: You can either:

- calibrate the files on the fly by providing the detx file to orcasong (recommended),
- or use a seperate tool from km3pipe called :code:`calibrate`, that will add the pos_xyz information to the hdf5 datafile.

While the first method is the recommended one in principal, the second one can be useful for determining the proper bin edges by looking
at single files. It can be used like this::

    calibrate /sps/km3net/users/mmoser/det_files/orca_115strings_av23min20mhorizontal_18OMs_alt9mvertical_v1.detx testfile.h5

As you can see, you need a .detx geometry file for this "calibration". Typically, you can find the path of this detx
file on the wiki page of the simulation production that you are using.

At this point, we are now ready to start using OrcaSong for the generation of event images.
See the page :ref:`orcasong_page` for instructions on how to use it.




