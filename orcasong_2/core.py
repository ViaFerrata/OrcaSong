import os
import km3pipe as kp
import km3modules as km

import orcasong_2.modules as modules
import orcasong_2.util.bin_stats_plot as bs_plot
from orcasong_2.mc_info_types import get_mc_info_extr


__author__ = 'Stefan Reck'


class FileBinner:
    """
    For making binned images and mc_infos, which can be used for conv. nets.

    Can also add statistics of the binning to the h5 files, which can
    be plotted to show the distribution of hits among the bins and how
    many hits were cut off.

    Attributes
    ----------
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
    def __init__(self,
                 bin_edges_list,
                 mc_info_extr=None,
                 det_file=None,
                 add_t0=False,
                 center_time=True,
                 event_skipper=None,
                 add_bin_stats=True):
        """
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
        mc_info_extr : function or string, optional
            Function that extracts desired mc_info from a blob, which is then
            stored as the "y" datafield in the .h5 file.
            Can also give a str identifier for an existing extractor.
        det_file : str, optional
            Path to a .detx detector geometry file, which can be used to
            calibrate the hits.
        add_t0 : bool
            If true, add t0 to the time of hits. If using a det_file,
            this will already have been done automatically.
        center_time : bool
            Subtract time of first triggered hit from all hit times.
            Will also be done for McHits if they are in the blob.
        event_skipper : func, optional
            Function that takes the blob as an input, and returns a bool.
            If the bool is true, the blob will be skipped.
        add_bin_stats : bool
            Add statistics of the binning to the output file. They can be
            plotted with util/bin_stats_plot.py.

        """
        self.bin_edges_list = bin_edges_list
        self.mc_info_extr = mc_info_extr
        self.det_file = det_file
        self.add_t0 = add_t0
        self.center_time = center_time
        self.event_skipper = event_skipper

        if add_bin_stats:
            self.bin_plot_freq = 1
        else:
            self.bin_plot_freq = None

        self.keep_event_info = True
        self.keep_mc_tracks = False

        self.n_statusbar = 1000
        self.n_memory_observer = 1000
        self.chunksize = 32
        self.complib = 'zlib'
        self.complevel = 1
        self.flush_frequency = 1000

    def run(self, infile, outfile=None, save_plot=False):
        """
        Make images for a file.

        Parameters
        ----------
        infile : str
            Path to the input file.
        outfile : str, optional
            Path to the output file (will be created). If none is given,
            will auto generate the name and save it in the cwd.
        save_plot : bool
            Save the binning hists as a pdf. Only possible if add_bin_stats
            is True.

        """
        if save_plot and self.bin_plot_freq is None:
            raise ValueError("Can not make plot when add_bin_stats is False")

        name, shape = self.get_names_and_shape()
        print("Generating {} images with shape {}".format(name, shape))

        if outfile is None:
            infile_basename = os.path.basename(infile)
            outfile_name = os.path.splitext(infile_basename)[0] + "_binned.h5"
            outfile = os.path.join(os.getcwd(), outfile_name)

        pipe = self.build_pipe(infile, outfile)
        smry = pipe.drain()

        if self.bin_plot_freq is not None:
            hists = smry["BinningStatsMaker"]
            bs_plot.add_hists_to_h5file(hists, outfile)

            if save_plot:
                save_as = os.path.splitext(outfile)[0] + "_hists.pdf"
                bs_plot.plot_hists(hists, save_as)

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
            Save the binning hists as a pdf. Only possible if add_bin_stats
            is True.

        """
        if save_plot and self.bin_plot_freq is None:
            raise ValueError("Can not make plot when add_bin_stats is False")

        outfiles = []
        for infile in infiles:
            outfile_name = os.path.splitext(os.path.basename(infile))[0] \
                           + "_hist.h5"
            outfile = os.path.join(outfolder, outfile_name)
            outfiles.append(outfile)

            self.run(infile, outfile, save_plot=False)

        if save_plot:
            bs_plot.plot_hist_of_files(files=outfiles,
                                       save_as=outfolder+"binning_hist.pdf")

    def build_pipe(self, infile, outfile):
        """
        Build the pipeline to generate images and mc_info for a file.
        """

        pipe = kp.Pipeline()

        if self.n_statusbar is not None:
            pipe.attach(km.common.StatusBar, every=self.n_statusbar)
        if self.n_memory_observer is not None:
            pipe.attach(km.common.MemoryObserver, every=self.n_memory_observer)

        pipe.attach(kp.io.hdf5.HDF5Pump, filename=infile)

        pipe.attach(km.common.Keep, keys=['EventInfo', 'Header', 'RawHeader',
                                          'McTracks', 'Hits', 'McHits'])

        if self.det_file:
            pipe.attach(modules.DetApplier, det_file=self.det_file)

        if self.center_time or self.add_t0:
            pipe.attach(modules.TimePreproc,
                        add_t0=self.add_t0,
                        center_time=self.center_time)

        if self.event_skipper is not None:
            pipe.attach(modules.EventSkipper, event_skipper=self.event_skipper)

        if self.bin_plot_freq is not None:
            pipe.attach(modules.BinningStatsMaker,
                        bin_plot_freq=self.bin_plot_freq,
                        bin_edges_list=self.bin_edges_list)

        pipe.attach(modules.ImageMaker,
                    bin_edges_list=self.bin_edges_list,
                    store_as="histogram")

        if self.mc_info_extr is not None:
            if isinstance(self.mc_info_extr, str):
                mc_info_extr = get_mc_info_extr(self.mc_info_extr)
            else:
                mc_info_extr = self.mc_info_extr

            pipe.attach(modules.McInfoMaker,
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
