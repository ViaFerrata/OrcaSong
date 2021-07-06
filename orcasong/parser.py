"""
Run OrcaSong functionalities from command line.

"""
import argparse
from orcasong.tools.concatenate import concatenate
from orcasong.tools.postproc import postproc_file
from orcasong.tools.shuffle2 import h5shuffle2
import orcasong.from_toml as from_toml
import orcasong.plotting.plot_binstats as plot_binstats
import orcasong.tools.make_data_split as make_data_split


def _add_parser_concatenate(subparsers):
    parser = subparsers.add_parser(
        "concatenate",
        description='Concatenate many small h5 files to a single large one '
                    'in a km3pipe compatible format. This is intended for '
                    'files that get generated by orcasong, i.e. all datsets '
                    'should have the same length, with one row per '
                    'blob. '
                    'Compression options and the datasets to be created in '
                    'the new file will be read from the first input file.')
    parser.add_argument(
        'file', type=str, nargs="*",
        help="Define the files to concatenate. If it's one argument: A txt list "
             "with pathes of h5 files to concatenate (one path per line). "
             "If it's multiple arguments: "
             "The pathes of h5 files to concatenate.")
    parser.add_argument(
        '--outfile', type=str, default="concatenated.h5",
        help='The absoulte filepath of the output .h5 file that will be created. ')
    parser.add_argument(
        '--no_used_files', action='store_true',
        help="Per default, the paths of the input files are added "
             "as their own datagroup in the output file. Use this flag to "
             "disable. ")
    parser.add_argument(
        '--skip_errors', action='store_true',
        help="If true, ignore files that can't be concatenated. ")
    parser.set_defaults(func=concatenate)


def _add_parser_h5shuffle(subparsers):
    parser = subparsers.add_parser(
        "h5shuffle",
        description='Shuffle an h5 file using km3pipe.',
    )
    parser.add_argument('input_file', type=str, help='File to shuffle.')
    parser.add_argument('--output_file', type=str,
                        help='Name of output file. Default: Auto generate name.')
    parser.add_argument('--delete', action="store_true",
                        help='Delete original file afterwards.')
    parser.set_defaults(func=postproc_file)


def _add_parser_h5shuffle2(subparsers):
    parser = subparsers.add_parser(
        "h5shuffle2",
        description="Shuffle datasets in a h5file that have the same length. "
        "Uses chunkwise readout for speed-up."
    )
    parser.add_argument(
        "input_file", type=str, help="Path of the file that will be shuffled."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="If given, this will be the name of the output file. "
        "Default: input_file + suffix.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=("x", "y"),
        help="Which datasets to include in output. Default: x, y",
    )
    parser.add_argument(
        "--max_ram_fraction",
        type=float,
        default=0.25,
        help="in [0, 1]. Fraction of all available ram to use for reading one batch of data "
        "Note: this should "
        "be <=~0.25 or so, since lots of ram is needed for in-memory shuffling. "
        "Default: 0.25",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Shuffle the file this many times. Default: Auto choose best number.",
    )
    parser.add_argument(
        "--max_ram",
        type=int,
        default=None,
        help="Available ram in bytes. Default: Use fraction of maximum "
             "available instead (see max_ram_fraction).",
    )
    parser.set_defaults(func=h5shuffle2)


def _add_parser_version(subparsers):
    def show_version():
        from orcasong import version
        print(version)

    parser = subparsers.add_parser(
        "version",
        description="Show installed orcanet version.",
    )
    parser.set_defaults(func=show_version)


def main():
    parser = argparse.ArgumentParser(
        prog="orcasong",
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers()

    from_toml.add_parser_run(subparsers)
    _add_parser_concatenate(subparsers)
    _add_parser_h5shuffle(subparsers)
    _add_parser_h5shuffle2(subparsers)
    plot_binstats.add_parser(subparsers)
    make_data_split.add_parser(subparsers)
    _add_parser_version(subparsers)

    kwargs = vars(parser.parse_args())
    func = kwargs.pop("func")
    func(**kwargs)