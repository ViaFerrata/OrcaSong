import os
import toml
import orcasong.core
import orcasong.extractors as extractors

# available extractors. First argument has to be the input filename
EXTRACTORS = {
    "neutrino_mc": extractors.get_neutrino_mc_info_extr,
    "neutrino_data": extractors.get_real_data_info_extr,
}


def _add_args(parser):
    parser.add_argument('infile', type=str, help="Aanet file in h5 format.")
    parser.add_argument('toml_file', type=str, help="Orcasong configuration in toml format.")
    parser.add_argument('--detx_file', type=str, default=None, help=(
        "Optional detx file to calibrate on the fly. Can not be used if a "
        "detx_file has also been given in the toml file."))
    parser.add_argument('--outfile', type=str, default=None, help=(
        "Path to output file. Default: Save with auto ogenerated name in cwd."))


def add_parser_filegraph(subparsers):
    parser = subparsers.add_parser(
        "graph",
        description='Produce a graph dl file from an aanet file.')
    _add_args(parser)
    parser.set_defaults(func=get_run_orcasong(orcasong.core.FileGraph))


def add_parser_filebinner(subparsers):
    parser = subparsers.add_parser(
        "image",
        description='Produce an image dl file from an aanet file.')
    _add_args(parser)
    parser.set_defaults(func=get_run_orcasong(orcasong.core.FileBinner))


def get_run_orcasong(processor):
    def run_orcasong(infile, toml_file, detx_file=None, outfile=None):
        if outfile is None:
            outfile = f"{os.path.splitext(os.path.basename(infile))[0]}_dl.h5"

        cfg = toml.load(toml_file)
        if "detx_file" in cfg:
            if detx_file is not None:
                raise ValueError("detx_file passed to function AND defined in toml")
            detx_file = cfg.pop("detx_file")

        if "extractor" in cfg:
            extractor_name = cfg.pop("extractor")
            extractor_cfg = cfg.pop("extractor_config", {})
            extractor = EXTRACTORS[extractor_name](infile, **extractor_cfg)
        else:
            extractor = None

        processor(
            det_file=detx_file,
            extractor=extractor,
            **cfg,
        ).run(infile=infile, outfile=outfile)
    return run_orcasong
