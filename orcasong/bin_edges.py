"""
Binnings used for some existing detector configurations.

These are made for the specific given runs. They might not be
applicable to other data, and could cause errors or produce unexpected
results when used on data other then the specified.
"""

import numpy as np


def get_edges_2017_ztc():
    """
    Designed for the 2017 runs with the one line detector.

    Will produce (18, 100, 31) 3d data, with dimensions ztc.

    Z binning: 9.45 meters each
    Time binning: 6 ns each
    Channel id binning: 1 DOM per bin

    """
    bin_edges_list = [
        ["pos_z", np.linspace(26, 198, 18 + 1)],
        ["time", np.linspace(-50, 550, 100 + 1)],
        ["channel_id", np.linspace(-0.5, 30.5, 31 + 1)],
    ]
    return bin_edges_list

