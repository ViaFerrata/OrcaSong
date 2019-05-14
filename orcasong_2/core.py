import os
import km3pipe as kp
import km3modules as km

from orcasong_2.modules import (TimePreproc,
                                ImageMaker,
                                McInfoMaker,
                                BinningStatsMaker,
                                EventSkipper)
from orcasong_2.mc_info_types import get_mc_info_extr
from orcasong_2.util.bin_stats_plot import (plot_hists,
                                            add_hists_to_h5file,
                                            plot_hist_of_files)

__author__ = 'Stefan Reck'


class FileBinner:
    """
    For making binned images and mc_infos, which can be used for conv. nets.

    Can also add statistics of the binning to the h5 files, which can
    be plotted to show the distribution of hits among the bins and how
    many hits were cut off.

    Attributes
    ----------
    bin_edges_list : List
        List with the names of the fields to bin, and the respective bin edges,
        including the left- and right-most bin edge.
        Example: For 10 bins in the z direction, and 100 bins in time:
            bin_edges_list = [
                ["pos_z", np.linspace(0, 10, 11)],
                ["time", np.linspace(-50, 550, 101)],
            ]
    mc_info_extr : function or string, optional
        Function that extracts desired mc_info from a blob, which is then
        stored as the "y" datafield in the .h5 file.
        Can also give a str identifier for an existing extractor.
    event_skipper : func, optional
        Function that takes the blob as an input, and returns a bool.
        If the bool is true, the blob will be skipped.
    bin_plot_freq : int or None
        If int is given, defines after how many blobs data for an overview
        histogram is extracted.
        It shows the distribution of hits, the bin edges, and how many hits
        were cut off for each field name in bin_edges_list.
        It will be saved to the same path as the outfile in run.
    keep_event_info : bool
        If True, will keep the "event_info" table.
    keep_mc_tracks : bool
        If True, will keep the "McTracks" table.
    n_statusbar : int, optional
        Print a statusbar every n blobs.
    n_memory_observer : int, optional
        Print memory usage every n blobs.
    do_time_preproc : bool
        Do time preprocessing, i.e. add t0 only to real data, and subtract time
        of first triggered hit.
        Will also be done for McHits if they are in the blob.
    chunksize : int
        Chunksize (along axis_0) used for saving the output to a .h5 file.
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

    """
    def __init__(self, bin_edges_list, mc_info_extr=None,
                 event_skipper=None, add_bin_stats=True):

        self.bin_edges_list = bin_edges_list
        self.mc_info_extr = mc_info_extr
        self.event_skipper = event_skipper

        if add_bin_stats:
            self.bin_plot_freq = 1
        else:
            self.bin_plot_freq = None

        self.keep_event_info = True
        self.keep_mc_tracks = False

        self.n_statusbar = 1000
        self.n_memory_observer = 1000
        self.do_time_preproc = True

        self.chunksize = 32
        self.complib = 'zlib'
        self.complevel = 1
        self.flush_frequency = 1000

    def run(self, infile, outfile, save_plot=False):
        """
        Make images for a file.

        Parameters
        ----------
        infile : str
            Path to the input file.
        outfile : str
            Path to the output file (will be created).
        save_plot : bool
            Save the binning hists as a pdf. Only possible if bin_plot_freq
            is not None.

        """
        if save_plot and self.bin_plot_freq is None:
            raise ValueError("Can not make plot when add_bin_stats is False")

        name, shape = self.get_names_and_shape()
        print("Generating {} images with shape {}".format(name, shape))

        pipe = self.build_pipe(infile, outfile)
        smry = pipe.drain()

        if self.bin_plot_freq is not None:
            hists = smry["BinningStatsMaker"]
            add_hists_to_h5file(hists, outfile)

            if save_plot:
                save_as = os.path.splitext(outfile)[0] + "_hists.pdf"
                plot_hists(hists, save_as)

    def run_multi(self, infiles, outfolder, save_plot=False):
        """
        Bin multiple files into their own output files each.

        Parameters
        ----------
        infiles : List
            The path to infiles as str.
        outfolder : str
            The output folder to place them in. The output file name will
            be generated automatically.
        save_plot : bool
            Save the binning hists as a pdf. Only possible if bin_plot_freq
            is not None.

        """
        if save_plot and self.bin_plot_freq is None:
            raise ValueError("Can not make plot when add_bin_stats is False")

        outfiles = []
        for infile in infiles:
            outfile_name = os.path.splitext(os.path.basename(infile))[0] \
                           + "_hist.h5"
            outfile = outfolder + outfile_name
            outfiles.append(outfile)

            self.run(infile, outfile, save_plot=False)

        if save_plot:
            plot_hist_of_files(outfiles, save_as=outfolder+"binning_hist.pdf")

    def build_pipe(self, infile, outfile):
        """
        Build the pipe to generate images and mc_info for a file.
        """

        pipe = kp.Pipeline()

        if self.n_statusbar is not None:
            pipe.attach(km.common.StatusBar, every=self.n_statusbar)
        if self.n_memory_observer is not None:
            pipe.attach(km.common.MemoryObserver, every=self.n_memory_observer)

        pipe.attach(kp.io.hdf5.HDF5Pump, filename=infile)

        pipe.attach(km.common.Keep, keys=['EventInfo', 'Header', 'RawHeader',
                                          'McTracks', 'Hits', 'McHits'])
        if self.do_time_preproc:
            pipe.attach(TimePreproc)

        if self.event_skipper is not None:
            pipe.attach(EventSkipper, event_skipper=self.event_skipper)

        if self.bin_plot_freq is not None:
            pipe.attach(BinningStatsMaker,
                        bin_plot_freq=self.bin_plot_freq,
                        bin_edges_list=self.bin_edges_list)

        pipe.attach(ImageMaker,
                    bin_edges_list=self.bin_edges_list,
                    store_as="histogram")

        if self.mc_info_extr is not None:
            if isinstance(self.mc_info_extr, str):
                mc_info_extr = get_mc_info_extr(self.mc_info_extr)
            else:
                mc_info_extr = self.mc_info_extr

            pipe.attach(McInfoMaker,
                        mc_info_extr=mc_info_extr,
                        store_as="mc_info")

        keys_keep = ['histogram', 'mc_info']
        if self.keep_event_info:
            keys_keep.append('EventInfo')
        if self.keep_mc_tracks:
            keys_keep.append('McTracks')
        pipe.attach(km.common.Keep, keys=keys_keep)

        pipe.attach(kp.io.HDF5Sink,
                    filename=outfile,
                    complib=self.complib,
                    complevel=self.complevel,
                    chunksize=self.chunksize,
                    flush_frequency=self.flush_frequency)
        return pipe

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
        name, shape = self.get_names_and_shape()
        return "<FileBinner: {} {}>".format(name, shape)
