"""
Binnings used for some existing detector configurations.

These are made for the specific given runs. They might not be
applicable to other data, and could cause errors or produce unexpected
results when used on data other then the specified.
"""

import numpy as np


def get_4line_bin_edges_list(time_resolution="low"):
    """
    Designed for four line detector.

    XYZ/PM : 1 dom per bin
    Time:
        low:
            100 bins, 6 ns/bin
            mchits cut off (left/right): 1.34%, 4.20%
        high:
            300 bins, 3.33 ns/bin
            mchits cut off (left/right): 0.6%, 0.73%

    """
    time_resolutions = {
        "low": np.linspace(-50, 550, 100 + 1),
        "high": np.linspace(-100, 900, 300 + 1),
    }

    return [
        ["pos_x", np.array([-2 - 30, -2, -2 + 30])],
        ["pos_y", np.array([3.7 - 30, 3.7, 3.7 + 30])],
        ["pos_z", np.linspace(24, 198, 19)],
        ["time", time_resolutions[time_resolution]],
        ["channel_id", np.linspace(-0.5, 30.5, 31 + 1)],
    ]


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
