"""Argument parser functions."""

from configs.config import get_cfg_defaults

import argparse


def parse_args():
    """
    Parse the following arguments for a default parser for PySlowFast users.
    Args:
        cfg (str): path to the config file.
        opts (argument): provide additional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/generate_annotations_19-05-21.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See configs/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg_defaults()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    return cfg
