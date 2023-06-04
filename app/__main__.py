import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np

import wsd.data.coarsewsd20 as cwsd
from wsd.vectorise import vectorise_coarsewsd20
from wsd.vectorise.bert import BertVectoriser


def print_args(args: Namespace):
    print(args)
    try:
        input("Press any key to continue, or CTRL+C to abort.")
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(0)


def vectorise(args: Namespace):
    out_path: Path = args.out
    assert (out_path.exists() and out_path.is_dir()) or out_path.suffix == ""
    out_path.resolve()
    out_path.mkdir(exist_ok=True, parents=True)

    dataset = cwsd.load_dataset(cwsd.Variant.REGULAR)
    bert_kwargs = {"device": "cuda" if args.cuda else None}
    vectoriser = BertVectoriser(**bert_kwargs)  # type: ignore

    for key, embedding in vectorise_coarsewsd20(vectoriser, dataset, args.batchsize):
        fn = out_path / f"{key}.npy"
        np.save(fn, embedding)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="print the config and parsed arguments"
    )

    subparsers = parser.add_subparsers(
        required=True, title="action", help="the action to perform"
    )

    parser_vectorise = subparsers.add_parser("vectorise")
    parser_vectorise.set_defaults(action=vectorise)
    parser_vectorise.add_argument(
        "data", choices=["coarsewsd"], help="the type of dataset to vectorise"
    )
    parser_vectorise.add_argument(
        "-b", "--batchsize", type=int, default=24, help="the batch size to use"
    )
    parser_vectorise.add_argument(
        "-p", "--path", type=Path, help="the path to the dataset"
    )
    parser_vectorise.add_argument(
        "-o",
        "--out",
        type=Path,
        default="./out/vectorised",
        help="the output directory",
    )
    parser_vectorise.add_argument(
        "--cuda", action="store_true", help="whether to enable CUDA"
    )

    args = parser.parse_args()
    if args.debug:
        print_args(args)
    args.action(args)


if __name__ == "__main__":
    main()
