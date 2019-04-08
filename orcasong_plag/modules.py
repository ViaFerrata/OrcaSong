"""
Custom km3pipe modules for making nn input files.
"""

import km3pipe as kp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker


class McInfoMaker(kp.Module):
    """
    Get the desired mc_info from the blob.

    Attributes
    ----------
    mc_info_extr : function
        Function to extract the info. Takes the blob as input, outputs
        a dict with the desired mc_infos.
    store_as : str
        Store the mcinfo with this name in the blob.

    """
    def configure(self):
        self.mc_info_extr = self.require('mc_info_extr')
        self.store_as = self.require('store_as')

    def process(self, blob):
        track = self.mc_info_extr(blob)
        dtypes = [(key, np.float64) for key in track.keys()]
        kp_hist = kp.dataclasses.Table(track,
                                       dtype=dtypes,
                                       h5loc='y',
                                       name='event_info')

        blob[self.store_as] = kp_hist
        return blob


class TimePreproc(kp.Module):
    """
    Preprocess the time in the blob.

    t0 will be added to the time for real data, but not simulations.
    Time hits and mchits will be shifted by the time of the first
    triggered hit.

    """
    def process(self, blob):
        correct_mchits = "McHits" in blob
        blob = time_preproc(blob,
                            correct_hits=True,
                            correct_mchits=correct_mchits)
        return blob


def time_preproc(blob, correct_hits=True, correct_mchits=True):
    """
    Preprocess the time in the blob.

    t0 will be added to the time for real data, but not simulations.
    Time hits and mchits will be shifted by the time of the first
    triggered hit.

    """
    hits_time = blob["Hits"].time

    if "McHits" not in blob:
        # add t0 only for real data, not sims
        hits_t0 = blob["Hits"].t0
        hits_time = np.add(hits_time, hits_t0)

    hits_triggered = blob["Hits"].triggered
    t_first_trigger = np.min(hits_time[hits_triggered == 1])

    if correct_hits:
        blob["Hits"].time = np.subtract(hits_time, t_first_trigger)

    if correct_mchits:
        mchits_time = blob["McHits"].time
        blob["McHits"].time = np.subtract(mchits_time, t_first_trigger)

    return blob


class ImageMaker(kp.Module):
    """
    Make a n-d histogram from the blob.

    Attributes
    ----------
    bin_edges_list : List
        List with the names of the fields to bin, and the respective bin edges,
        including the left- and right-most bin edge.
    store_as : str
        Store the images with this name in the blob.

    """
    def configure(self):
        self.bin_edges_list = self.require('bin_edges_list')
        self.store_as = self.require('store_as')

    def process(self, blob):
        data, bins, name = [], [], ""

        for bin_name, bin_edges in self.bin_edges_list:
            data.append(blob["Hits"][bin_name])
            bins.append(bin_edges)
            name += bin_name + "_"

        histogram = np.histogramdd(data, bins=bins)[0]
        title = name + "event_images"

        hist_one_event = histogram[np.newaxis, ...].astype(np.uint8)
        kp_hist = kp.dataclasses.NDArray(hist_one_event, h5loc='x', title=title)

        blob[self.store_as] = kp_hist
        return blob


class BinningPlotter(kp.Module):
    """
    Save a histogram of the number of hits for each binning field name.

    E.g. if the bin_edges_list contains "pos_z", this will save a histogram
    of #Hits vs. "pos_z" as a pdf, together with how many hits were outside
    of the bin edges in both directions.

    Per default, the resolution of the histogram (width of bins) will be
    higher then the given bin edges, and the edges will be plotted as horizontal
    lines. The time is the exception: The plotted bins have exactly the
    given bin edges.

    Attributes
    ----------
    bin_edges_list : List
        List with the names of the fields to bin, and the respective bin edges,
        including the left- and right-most bin edge.
    pdf_path : str
        Where to save the hists to. This pdf will contain all the field names
        on their own page each.
    bin_plot_freq : int
        Extract data for the hitograms only every given number of blobs
        (reduces time the pipeline takes to complete).
    res_increase : int
        Increase the number of bins by this much in the plot (so that one
        can see if the edges have been placed correctly). Is never used
        for the time binning (field name "time").
    plot_bin_edges : bool
        If true, will plot the bin edges as horizontal lines. Is never used
        for the time binning (field name "time").

    """
    def configure(self):
        self.bin_edges_list = self.require('bin_edges_list')
        self.pdf_path = self.require('pdf_path')
        self.bin_plot_freq = self.get("bin_plot_freq", default=20)
        self.res_increase = self.get('res_increase', default=5)
        self.plot_bin_edges = self.get('plot_bin_edges', default=True)

        self.hists = {}
        for bin_name, bin_edges in self._yield_spaced_bin_edges():
            self.hists[bin_name] = {
                "hist": np.zeros(len(bin_edges) - 1),
                "out_pos": 0,
                "out_neg": 0,
            }

        self.i = 0

    def _space_bin_edges(self, bin_edges):
        """
        Increase resolution of given binning.
        """
        increased_n_bins = (len(bin_edges) - 1) * self.res_increase + 1
        bin_edges = np.linspace(bin_edges[0], bin_edges[-1],
                                increased_n_bins)

        return bin_edges

    def _yield_spaced_bin_edges(self):
        for bin_name, bin_edges in self.bin_edges_list:
            if bin_name != "time":
                bin_edges = self._space_bin_edges(bin_edges)
            yield bin_name, bin_edges

    def process(self, blob):
        """
        Extract data from blob for the hist plots.
        """
        if self.i % self.bin_plot_freq == 0:
            for bin_name, bin_edges in self._yield_spaced_bin_edges():
                data = blob["Hits"][bin_name]
                hist = np.histogram(data, bins=bin_edges)[0]

                out_pos = data[data > np.max(bin_edges)].size
                out_neg = data[data < np.min(bin_edges)].size

                self.hists[bin_name]["hist"] += hist
                self.hists[bin_name]["out_pos"] += out_pos
                self.hists[bin_name]["out_neg"] += out_neg

        self.i += 1
        return blob

    def finish(self):
        """
        Make and save the histograms to pdf.
        """
        with PdfPages(self.pdf_path) as pdf_file:
            for bin_name, org_bin_edges in self.bin_edges_list:
                hist = self.hists[bin_name]["hist"]
                out_pos = self.hists[bin_name]["out_pos"]
                out_neg = self.hists[bin_name]["out_neg"]
                hist_frac = hist / (np.sum(hist) + out_pos + out_neg)

                if bin_name != "time":
                    bin_edges = self._space_bin_edges(org_bin_edges)
                else:
                    bin_edges = org_bin_edges

                bin_spacing = bin_edges[1] - bin_edges[0]
                fig, ax = plt.subplots()
                plt.bar(bin_edges[:-1],
                        hist_frac,
                        align="edge",
                        width=0.9*bin_spacing,
                        )
                ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

                if self.plot_bin_edges and bin_name != "time":
                    for bin_edge in org_bin_edges:
                        plt.axvline(x=bin_edge, color='grey', linestyle='-',
                                    linewidth=1, alpha=0.9)

                # place a text box in upper left in axes coords
                out_pos_rel = out_pos / np.sum(hist)
                out_neg_rel = out_neg / np.sum(hist)
                textstr = "Hits cut off:\n Left: {:.1%}\n" \
                          " Right: {:.1%}".format(out_neg_rel, out_pos_rel)
                props = dict(boxstyle='round', facecolor='white', alpha=0.9)
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
                        verticalalignment='top', bbox=props)

                plt.xlabel(bin_name)
                plt.ylabel("Fraction of hits")

                pdf_file.savefig(fig)
                print(bin_name, out_neg, out_pos)
        print("Saved binning plot to " + self.pdf_path)
