from argparse import ArgumentParser, Namespace
from pathlib import Path

from .command import Command


class SplitEmbeddings(Command):
    @staticmethod
    def name() -> str:
        return "split-embeddings"

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "-p", "--path", type=Path, default=Path("out/vectorised/bert-base-uncased")
        )
        parser.add_argument(
            "-o", "--out", type=Path, default=Path("out/split/bert-base-uncased")
        )
        parser.add_argument(
            "-s", "--split", type=str, choices=["train", "test"], default=None
        )

    @staticmethod
    def run(args: Namespace) -> None:
        from ..data import coarsewsd20 as cwsd
        from ..data.embeddings import CwsdEmbeddingSplitter
        from ..util.path import validate_and_create_dir, validate_existing_dir

        embedding_path = validate_existing_dir(args.path)
        out_path = validate_and_create_dir(args.out)

        dataset = cwsd.load_dataset(cwsd.Variant.REGULAR)
        embeddings = CwsdEmbeddingSplitter(dataset)
        embeddings.load_all(embedding_path)
        embeddings.save_splitted(out_path)
