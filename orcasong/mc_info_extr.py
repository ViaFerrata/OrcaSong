"""
Functions that extract info from a blob for the mc_info / y datafield
in the h5 files.

These are made for the specific given runs. They might not be
applicable to other data, and could cause errors or produce unexpected
results when used on data other then the specified.

"""

import warnings
import numpy as np

__author__ = 'Stefan Reck'


def get_real_data(blob):
    """
    Get info present in real data.
    Designed for the 2017 one line runs.

    """
    event_info = blob['EventInfo']

    track = {
        'event_id': event_info.event_id,
        # was .event_id[0] up to km3pipe 8.16.0
        'run_id': event_info.run_id,
        'trigger_mask': event_info.trigger_mask,
    }
    return track


def get_pure_noise(blob):
    """
    For simulated pure noise events, which have particle_type 0.

    """
    event_info = blob['EventInfo']

    track = {
        'event_id': event_info.event_id[0],
        'run_id': event_info.run_id,
        'particle_type': 0
    }
    return track
