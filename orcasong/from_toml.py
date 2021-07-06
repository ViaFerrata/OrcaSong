import os
from pathlib import Path
import toml
import orcasong.core
import orcasong.extractors as extractors

# built-in extractors. First argument has to be the input filename,
# other parameters can be set via '[extractor_config]' in the toml
EXTRACTORS = {
    "nu_chain_neutrino": extractors.get_neutrino_mc_info_extr,
    "nu_chain_muon": extractors.get_muon_mc_info_extr,
    "nu_chain_noise": extractors.get_random_noise_mc_info_extr,
    "nu_chain_data": extractors.get_real_data_info_extr,
    "bundle_mc": extractors.BundleMCExtractor,
    "bundle_data": extractors.BundleDataExtractor,
}

MODES = {
    "graph": orcasong.core.FileGraph,
    "image": orcasong.core.FileBinner,
}


def add_parser_run(subparsers):
    parser = subparsers.add_parser(
        "run", description='Produce a dl file from an aanet file.')
    parser.add_argument('infile', type=str, help="Aanet file in h5 format.")
    parser.add_argument('config', type=str, help=(
        "Orcasong configuration in toml format. Use prefix 'orcasong:' to load "
        "a toml from OrcaSong/configs."))
    parser.add_argument('--detx', type=str, default=None, help=(
        "Optional detx file to calibrate on the fly."))
    parser.add_argument('--outfile', type=str, default=None, help=(
        "Path to output file. Default: Save with auto generated name in cwd."))
    parser.set_defaults(func=run_orcasong)


def run_orcasong(infile, config, detx=None, outfile=None):
    setup_processor(infile, config, detx).run(infile=infile, outfile=outfile)


def setup_processor(infile, toml_file, detx_file=None):
    if toml_file.startswith("orcasong:"):
        toml_file = _get_config(toml_file[9:])
    cfg = toml.load(toml_file)
    processor = _get_verbose(cfg.pop("mode"), MODES)

    if "extractor" in cfg:
        extractor_cfg = cfg.pop("extractor_config", {})
        extractor = _get_verbose(
            cfg.pop("extractor"), EXTRACTORS)(infile, **extractor_cfg)
    else:
        extractor = None

    return processor(
        det_file=detx_file,
        extractor=extractor,
        **cfg,
    )


def _get_verbose(key, d):
    if key not in d:
        raise KeyError(f"Unknown key '{key}' (available: {list(d.keys())})")
    return d[key]


def _get_config(filename):
    direc = os.path.join(Path(orcasong.core.__file__).parents[1], "configs")
    files = {file: os.path.join(direc, file) for file in os.listdir(direc)}
    return _get_verbose(filename, files)
