import inspect
from unittest import TestCase
import orcasong.bin_edges
from orcasong.core import FileBinner


__author__ = 'Stefan Reck'


class TestEdges(TestCase):
    """
    Just call all functions in the bin_edges module and see if they work
    with the filebinner.
    """
    def test_them(self):
        funcs = [memb[1] for memb in inspect.getmembers(orcasong.bin_edges)
                 if inspect.isfunction(memb[1])]

        for func in funcs:
            fb = FileBinner(func())
            fb.get_names_and_shape()
