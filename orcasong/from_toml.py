import toml
import orcasong.core
import orcasong.extractors as extractors

# built-in extractors. First argument has to be the input filename,
# other parameters can be set via 'extractor_config' dict in the toml
EXTRACTORS = {
    "neutrino_mc": extractors.get_neutrino_mc_info_extr,
    "neutrino_data": extractors.get_real_data_info_extr,
}

MODES = {
    "graph": orcasong.core.FileGraph,
    "image": orcasong.core.FileBinner,
}


def add_parser_run(subparsers):
    parser = subparsers.add_parser(
        "run",
        description='Produce a dl file from an aanet file.')
    parser.add_argument('infile', type=str, help="Aanet file in h5 format.")
    parser.add_argument('toml_file', type=str, help="Orcasong configuration in toml format.")
    parser.add_argument('--detx_file', type=str, default=None, help=(
        "Optional detx file to calibrate on the fly. Can not be used if a "
        "detx_file has also been given in the toml file."))
    parser.add_argument('--outfile', type=str, default=None, help=(
        "Path to output file. Default: Save with auto generated name in cwd."))
    parser.set_defaults(func=run_orcasong)


def run_orcasong(infile, toml_file, detx_file=None, outfile=None):
    setup_processor(infile, toml_file, detx_file).run(
        infile=infile, outfile=outfile)


def setup_processor(infile, toml_file, detx_file=None):
    cfg = toml.load(toml_file)
    processor = _get_verbose(cfg.pop("mode"), MODES)

    if "detx_file" in cfg:
        if detx_file is not None:
            raise ValueError("detx_file passed to run AND defined in toml")
        detx_file = cfg.pop("detx_file")

    if "extractor" in cfg:
        extractor_cfg = cfg.pop("extractor_config", {})
        extractor = _get_verbose(cfg.pop("extractor"), EXTRACTORS)(infile, **extractor_cfg)
    else:
        extractor = None

    return processor(
        det_file=detx_file,
        extractor=extractor,
        **cfg,
    )


def _get_verbose(key, d):
    if key not in d:
        raise KeyError(f"Unknown key {key} (available: {list(d.keys())}")
    return d[key]
