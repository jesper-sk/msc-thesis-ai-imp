import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

from .command import Command


class PrepEwiser(Command):
    @staticmethod
    def name() -> str:
        return "prep-ewiser"

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "-p",
            "--path",
            type=Path,
            help="The CoarseWSD data input path",
            default=None,
        )
        parser.add_argument(
            "-o",
            "--out",
            type=Path,
            help="The desired output directory",
            default="./out/coarsewsd",
        )

    @staticmethod
    def run(args: Namespace) -> None:
        from dataclasses import asdict

        from tqdm import tqdm

        from ..data import coarsewsd20 as cwsd
        from ..util.path import validate_and_create_dir, validate_existing_dir

        data_path = validate_existing_dir(args.path or cwsd.DATA_ROOT)
        out_path = validate_and_create_dir(args.out)

        data = cwsd.load_dataset(cwsd.Variant.REGULAR, data_path)

        sentences = []
        info = []
        for word in tqdm(cwsd.WORDS):
            entries = data[word].all()
            sentences += [
                " ".join(token) for token in cwsd.transpose_entries(entries).tokens
            ]
            info += [
                {
                    **{"word": word},
                    **{k: v for k, v in asdict(entry).items() if k != "tokens"},
                }
                for entry in entries
            ]

        with open(out_path / "sentences.txt", "w", encoding="utf-8") as file:
            file.write("\n".join(sentences))

        with open(out_path / "info.json", "w", encoding="utf-8") as file:
            json.dump(info, file)
