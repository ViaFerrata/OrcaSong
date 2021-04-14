import os
import toml
from orcasong.core import FileGraph
import orcasong.extractors as extractors


EXTRACTORS = {
    "neutrino_mc": extractors.get_neutrino_mc_info_extr,
    "neutrino_data": extractors.get_real_data_info_extr,
}


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "graph",
        description='Produce graph dl file from aanet file.')
    parser.add_argument('infile', type=str)
    parser.add_argument('toml_file', type=str)
    parser.add_argument('--detx_file', type=str, default=None)
    parser.add_argument('--outfile', type=str, default=None)
    return parser.parse_args()


def make_graph(infile, toml_file, detx_file=None, outfile=None):
    if outfile is None:
        outfile = f"{os.path.splitext(os.path.basename(infile))[0]}_dl.h5"

    cfg = toml.load(toml_file)
    if "detx_file" in cfg:
        if detx_file is not None:
            raise ValueError
        detx_file = cfg.pop("detx_file")

    extractor_name = cfg.pop("extractor")
    if "extractor_config" in cfg:
        extractor_cfg = cfg.pop("extractor_config")
    else:
        extractor_cfg = {}

    extractor = EXTRACTORS[extractor_name](infile, **extractor_cfg)

    FileGraph(
        det_file=detx_file,
        extractor=extractor,
        **cfg,
    ).run(infile=infile, outfile=outfile)
