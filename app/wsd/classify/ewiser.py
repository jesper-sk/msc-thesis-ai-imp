import argparse

import torch


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-c",
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Path of trained EWISER checkpoint(s).",
    )

    return parser


def build_model(args):
    data = torch.load(args.checkpoints[0], map_location="cpu")
    model_args = data["args"]
    model_args.cpu = args.device == "cpu"
    model_args.context_embeddings_cache = args.device
    state = data["model"]
