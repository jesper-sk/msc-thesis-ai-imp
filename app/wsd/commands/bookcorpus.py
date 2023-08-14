from argparse import ArgumentParser, Namespace
from pathlib import Path

from ..util.path import validate_existing_dir
from .command import Command

WORDSFILE_DEFAULT = Path("data/selected-words.csv")


class MyCommand(Command):
    @staticmethod
    def name() -> str:
        return "bookcorpus"

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "-w",
            "--wordsfile",
            type=Path,
            default=None,
            help="the path to the selected words CSV",
        )
        parser.add_argument(
            "-o", "--out", type=str, default=None, help="the path to save the dataset"
        )
        parser.add_argument(
            "-n",
            "--nproc",
            type=int,
            default=None,
            help="the amount of parallel processes to use",
        )
        parser.add_argument(
            "-m",
            "--memory",
            action="store_true",
            help="whether to keep the dataset in-memory or to write to cache file",
        )

    @staticmethod
    def run(args: Namespace) -> None:
        import datasets

        with open(args.wordsfile or WORDSFILE_DEFAULT, "r") as file:
            next(file)
            words = [
                " " + word + " "
                for row in file.read().split("\n")
                for word in row.split(",")
                if word
            ]

        kwargs = {
            "keep_in_memory": args.memory,
            "num_proc": args.nproc,
        }

        out = validate_existing_dir(args.out)

        (
            datasets.load_dataset("bookcorpus", **kwargs)
            .filter(lambda x: any(word in x["text"] for word in words), **kwargs)
            .save_to_disk(out, num_proc=args.nproc)
        )
