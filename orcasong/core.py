import os
from abc import abstractmethod
import h5py
import km3pipe as kp
import km3modules as km

import orcasong
import orcasong.modules as modules
import orcasong.plotting.plot_binstats as plot_binstats


__author__ = 'Stefan Reck'


class BaseProcessor:
    """
    Preprocess km3net/antares events for neural networks.

    This serves as a baseclass, which handles things like reading
    events, calibrating, generating labels and saving the output.

    Parameters
    ----------
    extractor : function, optional
        Function that extracts desired info from a blob, which is then
        stored as the "y" datafield in the .h5 file.
        The function takes the km3pipe blob as an input, and returns
        a dict mapping str to floats.
        Examples can be found in orcasong.extractors.
    det_file : str, optional
        Path to a .detx detector geometry file, which can be used to
        calibrate the hits.
    correct_mc_time : bool
        Converts MC hit times to JTE times.
    center_time : bool
        Subtract time of first triggered hit from all hit times. Will
        also be done for McHits if they are in the blob [default: True].
    calib_hits : bool
        Apply calibration to hits if det file is given. Default: True.
    calib_mchits : bool
        Apply calibration to mchits if det file is given and mchits are
        found in the blob. Default: True.
    correct_timeslew : bool
        If true (default), the time slewing of hits depending on their tot
        will be corrected during calibration.
        Only done if det file is given and calib_hits is True.
    center_hits_to : tuple, optional
        Translate the xyz positions of the hits (and mchits), as if
        the detector was centered at the given position.
        E.g., if its (0, 0, None), the hits and mchits will be
        centered at xy = 00, and z will be left untouched.
        Can only be used when a detx file is given.
    add_t0 : bool
        If true, add t0 to the time of hits and mchits. If using a
        det_file, this will already have been done automatically
        [default: False].
    event_skipper : func, optional
        Function that takes the blob as an input, and returns a bool.
        If the bool is true, the blob will be skipped.
        This is placed after the binning and mc_info extractor.
    chunksize : int
        Chunksize (along axis_0) used for saving the output
        to a .h5 file [default: 32].
    keep_event_info : bool
        If True, will keep the "event_info" table [default: False].
    overwrite : bool
        If True, overwrite the output file if it exists already.
        If False, throw an error instead.
    sort_y : bool
        Sort the columns in the y dataset alphabetically.
    y_to_float64 : bool
        Convert everything in the y dataset to float 64 (Default: True).
        Hint: Not all other dtypes can store nan!

    Attributes
    ----------
    n_statusbar : int or None
        Print a statusbar every n blobs.
    n_memory_observer : int or None
        Print memory usage every n blobs.
    complib : str
        Compression library used for saving the output to a .h5 file.
        All PyTables compression filters are available, e.g. 'zlib',
        'lzf', 'blosc', ... .
    complevel : int
        Compression level for the compression filter that is used for
        saving the output to a .h5 file.
    flush_frequency : int
        After how many events the accumulated output should be flushed to
        the harddisk.
        A larger value leads to a faster orcasong execution,
        but it increases the RAM usage as well.
    seed : int, optional
        Makes all random (numpy) actions reproducable. Set at the start of
        each pipeline.

    """
    def __init__(self, extractor=None,
                 det_file=None,
                 correct_mc_time=True,
                 center_time=True,
                 calib_hits=True,
                 calib_mchits=True,
                 add_t0=False,
                 correct_timeslew=True,
                 center_hits_to=None,
                 event_skipper=None,
                 chunksize=32,
                 keep_event_info=False,
                 overwrite=True,
                 sort_y=True,
                 y_to_float64=True):
        if center_hits_to is not None and det_file is None:
            raise ValueError("det_file has to be given when using center_hits_to")

        self.extractor = extractor
        self.det_file = det_file
        self.correct_mc_time = correct_mc_time
        self.center_time = center_time
        self.calib_hits = calib_hits
        self.calib_mchits = calib_mchits
        self.add_t0 = add_t0
        self.correct_timeslew = correct_timeslew
        self.center_hits_to = center_hits_to
        self.event_skipper = event_skipper
        self.chunksize = chunksize
        self.keep_event_info = keep_event_info
        self.overwrite = overwrite
        self.sort_y = sort_y
        self.y_to_float64 = y_to_float64

        self.n_statusbar = 1000
        self.n_memory_observer = 1000
        self.complib = 'zlib'
        self.complevel = 1
        self.flush_frequency = 1000
        self.seed = 42

    def run(self, infile, outfile=None):
        """
        Process the events from the infile, and save them to the outfile.

        Parameters
        ----------
        infile : str
            Path to the input file.
        outfile : str, optional
            Path to the output file (will be created). If none is given,
            will auto generate the name and save it in the cwd.

        """
        if outfile is None:
            outfile = os.path.join(os.getcwd(), "{}_dl.h5".format(
                os.path.splitext(os.path.basename(infile))[0]))
        if not self.overwrite:
            if os.path.isfile(outfile):
                raise FileExistsError(f"File exists: {outfile}")
        if self.seed:
            km.GlobalRandomState(seed=self.seed)
        pipe = self.build_pipe(infile, outfile)
        summary = pipe.drain()
        with h5py.File(outfile, "a") as f:
            self.finish_file(f, summary)

    def run_multi(self, infiles, outfolder):
        """
        Process multiple files into their own output files each.
        The output file names will be generated automatically.

        Parameters
        ----------
        infiles : List
            The path to infiles as str.
        outfolder : str
            The output folder to place them in.

        """
        outfiles = []
        for infile in infiles:
            outfile = os.path.join(
                outfolder,
                f"{os.path.splitext(os.path.basename(infile))[0]}_dl.h5")
            outfiles.append(outfile)
            self.run(infile, outfile)
        return outfiles

    def build_pipe(self, infile, outfile, timeit=True):
        """ Initialize and connect the modules from the different stages. """
        components = [
            *self.get_cmpts_pre(infile=infile),
            *self.get_cmpts_main(),
            *self.get_cmpts_post(outfile=outfile),
        ]
        pipe = kp.Pipeline(timeit=timeit)
        if self.n_statusbar is not None:
            pipe.attach(km.common.StatusBar, every=self.n_statusbar)
        if self.n_memory_observer is not None:
            pipe.attach(km.common.MemoryObserver, every=self.n_memory_observer)
        for cmpt, kwargs in components:
            pipe.attach(cmpt, **kwargs)
        return pipe

    def get_cmpts_pre(self, infile):
        """ Modules that read and calibrate the events. """
        cmpts = [(kp.io.hdf5.HDF5Pump, {"filename": infile})]

        if self.correct_mc_time:
            cmpts.append((km.mc.MCTimeCorrector, {}))
        if self.det_file:
            cmpts.append((modules.DetApplier, {
                "det_file": self.det_file,
                "correct_timeslew": self.correct_timeslew,
                "center_hits_to": self.center_hits_to,
                "calib_hits": self.calib_hits,
                "calib_mchits": self.calib_mchits,
            }))

        if any((self.center_time, self.add_t0)):
            cmpts.append((modules.TimePreproc, {
                "add_t0": self.add_t0,
                "center_time": self.center_time}))
        return cmpts

    @abstractmethod
    def get_cmpts_main(self):
        """  Produce and store the samples as 'samples' in the blob. """
        raise NotImplementedError

    def get_cmpts_post(self, outfile):
        """ Modules that postproc and save the events. """
        cmpts = []
        if self.extractor is not None:
            cmpts.append((modules.McInfoMaker, {
                "extractor": self.extractor,
                "to_float64": self.y_to_float64,
                "sort_y": self.sort_y,
                "store_as": "mc_info"}))

        if self.event_skipper is not None:
            cmpts.append((modules.EventSkipper, {
                "event_skipper": self.event_skipper}))

        keys_keep = ['samples', 'mc_info', "header", "raw_header"]
        if self.keep_event_info:
            keys_keep.append('EventInfo')
        cmpts.append((km.common.Keep, {"keys": keys_keep}))

        cmpts.append((kp.io.HDF5Sink, {
            "filename": outfile,
            "complib": self.complib,
            "complevel": self.complevel,
            "chunksize": self.chunksize,
            "flush_frequency": self.flush_frequency}))
        return cmpts

    def finish_file(self, f, summary):
        """
        Work with the output file after the pipe has finished.

        Parameters
        ----------
        f : h5py.File
            The opened output file.
        summary : km3pipe.Blob
            The output from pipe.drain().

        """
        # Add current orcasong version to h5 file
        f.attrs.create("orcasong", orcasong.__version__)


class FileBinner(BaseProcessor):
    """
    For making binned images and mc_infos, which can be used for conv nets.

    Can also add statistics of the binning to the h5 files, which can
    be plotted to show the distribution of hits among the bins and how
    many hits were cut off.

    Parameters
    ----------
    bin_edges_list : List
        List with the names of the fields to bin, and the respective bin
        edges, including the left- and right-most bin edge.
        Example: For 10 bins in the z direction, and 100 bins in time:
            bin_edges_list = [
                ["pos_z", np.linspace(0, 10, 11)],
                ["time", np.linspace(-50, 550, 101)],
            ]
        Some examples can be found in orcasong.bin_edges.
    add_bin_stats : bool
        Add statistics of the binning to the output file. They can be
        plotted with util/bin_stats_plot.py [default: True].
    hit_weights : str, optional
        Use blob["Hits"][hit_weights] as weights for samples in histogram.
    kwargs
        Options of the BaseProcessor.

    """
    def __init__(self, bin_edges_list,
                 add_bin_stats=True,
                 hit_weights=None,
                 **kwargs):
        self.bin_edges_list = bin_edges_list
        self.add_bin_stats = add_bin_stats
        self.hit_weights = hit_weights
        super().__init__(**kwargs)

    def get_cmpts_main(self):
        """ Generate nD images. """
        cmpts = []
        if self.add_bin_stats:
            cmpts.append((modules.BinningStatsMaker, {
                "bin_edges_list": self.bin_edges_list}))
        cmpts.append((modules.ImageMaker, {
            "bin_edges_list": self.bin_edges_list,
            "hit_weights": self.hit_weights}))
        return cmpts

    def finish_file(self, f, summary):
        super().finish_file(f, summary)
        if self.add_bin_stats:
            plot_binstats.add_hists_to_h5file(summary["BinningStatsMaker"], f)

    def run_multi(self, infiles, outfolder, save_plot=False):
        """
        Bin multiple files into their own output files each.
        The output file names will be generated automatically.

        Parameters
        ----------
        infiles : List
            The path to infiles as str.
        outfolder : str
            The output folder to place them in.
        save_plot : bool
            Save the binning hists as a pdf. Only possible if add_bin_stats
            is True.

        """
        if save_plot and not self.add_bin_stats:
            raise ValueError("Can not make plot when add_bin_stats is False")

        name, shape = self.get_names_and_shape()
        print("Generating {} images with shape {}".format(name, shape))

        outfiles = super().run_multi(infiles=infiles, outfolder=outfolder)

        if save_plot:
            plot_binstats.plot_hist_of_files(
                files=outfiles, save_as=outfolder+"binning_hist.pdf")
        return outfiles

    def get_names_and_shape(self):
        """
        Get names and shape of the resulting x data,
        e.g. (pos_z, time), (18, 50).
        """
        names, shape = [], []
        for bin_name, bin_edges in self.bin_edges_list:
            names.append(bin_name)
            shape.append(len(bin_edges) - 1)
        return tuple(names), tuple(shape)

    def __repr__(self):
        return "<FileBinner: {} {}>".format(*self.get_names_and_shape())


class FileGraph(BaseProcessor):
    """
    Turn km3 events to graph data.

    The resulting file will have a dataset "x" of shape
    (total n_hits, len(hit_infos)).
    The column names of the last axis (i.e. hit_infos) are saved
    as attributes of the dataset (f["x"].attrs).

    Parameters
    ----------
    hit_infos : tuple, optional
        Which entries in the '/Hits' Table will be kept. E.g. pos_x, time, ...
        Default: Keep all entries.
    time_window : tuple, optional
        Two ints (start, end). Hits outside of this time window will be cut
        away (based on 'Hits/time'). Default: Keep all hits.
    only_triggered_hits : bool
        If true, use only triggered hits. Otherwise, use all hits (default).
    max_n_hits : int
        Maximum number of hits that gets saved per event. If an event has
        more, some will get cut randomly! Default: Keep all hits.
    fixed_length : bool
        If False (default), save hits of events with variable length as
        2d arrays using km3pipe's indices.
        If True, pad hits of each event with 0s to a fixed length,
        so that they can be stored as 3d arrays like images.
        max_n_hits needs to be given in that case, and a column will be
        added called 'is_valid', which is 0 if the entry is padded,
        and 1 otherwise.
        This is inefficient and will cut off hits, so it should not be used.
    kwargs
        Options of the BaseProcessor.

    """
    def __init__(self, max_n_hits=None,
                 time_window=None,
                 hit_infos=None,
                 only_triggered_hits=False,
                 fixed_length=False,
                 **kwargs):
        self.max_n_hits = max_n_hits
        self.fixed_length = fixed_length
        self.time_window = time_window
        self.hit_infos = hit_infos
        self.only_triggered_hits = only_triggered_hits
        super().__init__(**kwargs)

    def get_cmpts_main(self):
        return [((modules.PointMaker, {
            "max_n_hits": self.max_n_hits,
            "fixed_length": self.fixed_length,
            "time_window": self.time_window,
            "hit_infos": self.hit_infos,
            "dset_n_hits": "EventInfo",
            "only_triggered_hits": self.only_triggered_hits,
        }))]

    def finish_file(self, f, summary):
        super().finish_file(f, summary)
        for i, hit_info in enumerate(summary["PointMaker"]["hit_infos"]):
            f["x"].attrs.create(f"hit_info_{i}", hit_info)
        f["x"].attrs.create("indexed", not self.fixed_length)
