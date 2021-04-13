import os
import toml
from orcasong.core import FileGraph
from orcasong.extractors import get_neutrino_mc_info_extr


EXTRACTORS = {
    "neutrino": get_neutrino_mc_info_extr,
}


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "graph",
        description='Produce graph dl file from aanet file.')
    parser.add_argument('infile', type=str)
    parser.add_argument('config', type=str)
    parser.add_argument('detx_file', type=str)
    parser.add_argument('--outfile', type=str, default=None)
    return parser.parse_args()


def make_graph(infile, config, detx_file, outfile=None):
    if outfile is None:
        outfile = f"{os.path.splitext(os.path.basename(infile))[0]}_dl.h5"

    cfg = toml.load(config)
    extractor_cfg = cfg.pop("extractor")
    extractor_name = extractor_cfg.pop("name")
    inps = {"infile": infile, "config": config, "detx_file": detx_file, "outfile": outfile}
    extractor = EXTRACTORS[extractor_name](inps, **extractor_cfg)

    FileGraph(
        det_file=detx_file,
        extractor=extractor,
        **cfg,
    ).run(infile=infile, outfile=outfile)
