import km3pipe as kp
import km3modules as km

from orcasong_plag.modules import TimePreproc, ImageMaker, McInfoMaker
from orcasong_plag.mc_info_types import get_mc_info_extr


class FileBinner:
    """
    For making binned images.

    Attributes
    ----------
    bin_edges_list : List
        List with the names of the fields to bin, and the respective bin edges,
        including the left- and right-most bin edge.
        Example:
            bin_edges_list = [
                ["pos_z", np.linspace(0, 10, 11)],
                ["time", np.linspace(-50, 550, 101)],
            ]
    mc_info_extr : function or string, optional
        Function that extracts desired mc_info from a blob, which is then
        stored as the "y" datafield in the .h5 file.
        Can also give a str identifier for an existing extractor.
    n_statusbar : int, optional
        Print a statusbar every n blobs.
    n_memory_observer : int, optional
        Print memory usage every n blobs.
    do_time_preproc : bool
        Do time preprocessing, i.e. add t0 to real data, subtract time
        of first triggered hit.
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
    def __init__(self, bin_edges_list, mc_info_extr=None):
        self.bin_edges_list = bin_edges_list
        self.mc_info_extr = mc_info_extr

        self.n_statusbar = 200
        self.n_memory_observer = 400
        self.do_time_preproc = True
        # self.data_cuts = None

        self.chunksize = 32
        self.complib = 'zlib'
        self.complevel = 1
        self.flush_frequency = 1000

    def run(self, infile, outfile):
        """
        Build the pipeline to make images for the given file.

        Parameters
        ----------
        infile : str or List
            Path to the input file(s).
        outfile : str
            Path to the output file.

        """
        name, shape = self.get_name_and_shape()
        print("Generating {} images with shape {}".format(name, shape))

        pipe = kp.Pipeline()

        if self.n_statusbar is not None:
            pipe.attach(km.common.StatusBar, every=self.n_statusbar)
        if self.n_memory_observer is not None:
            pipe.attach(km.common.MemoryObserver, every=400)

        if not isinstance(infile, list):
            infile = [infile]

        pipe.attach(kp.io.hdf5.HDF5Pump, filenames=infile)

        self.attach_binning_modules(pipe)

        pipe.attach(kp.io.HDF5Sink,
                    filename=outfile,
                    complib=self.complib,
                    complevel=self.complevel,
                    chunksize=self.chunksize,
                    flush_frequency=self.flush_frequency)

        pipe.drain()

    def attach_binning_modules(self, pipe):
        """
        Attach modules to transform a blob to images and mc_info to a km3pipe.

        """
        pipe.attach(km.common.Keep, keys=['EventInfo', 'Header', 'RawHeader',
                                          'McTracks', 'Hits', 'McHits'])
        if self.do_time_preproc:
            pipe.attach(TimePreproc)

        # if self.data_cuts is not None:
        #     from orcasong.utils import EventSkipper
        #     pipe.attach(EventSkipper, data_cuts=self.data_cuts)

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

        pipe.attach(km.common.Keep, keys=['histogram', 'mc_info'])

    def get_name_and_shape(self):
        """
        Get name and shape of the resulting x data, e.g. "pos_z_time", (18, 50).
        """
        res_names, shape = [], []
        for bin_name, bin_edges in self.bin_edges_list:
            res_names.append(bin_name)
            shape.append(len(bin_edges) - 1)

        name = "_".join(res_names)
        shape = tuple(shape)

        return name, shape

    def __repr__(self):
        name, shape = self.get_name_and_shape()
        return "<FileBinner: {} {}>".format(name, shape)
