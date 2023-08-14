from argparse import ArgumentParser, Namespace
from pathlib import Path

from ..util.path import validate_and_create_dir
from .command import Command


class ExtractLemmata(Command):
    @staticmethod
    def name() -> str:
        return "lemmas"

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "-p", "--path", type=Path, help="the root path of the dataset", default=None
        )
        parser.add_argument(
            "-v",
            "--variant",
            type=str,
            help="The dataset variant to vectorise for wsdeval, ALL",
            default=None,
        )
        parser.add_argument(
            "-o",
            "--out",
            type=Path,
            help="The location for the output file",
            required=True,
        )

    @staticmethod
    def run(args: Namespace) -> None:
        from collections import defaultdict

        from tqdm import tqdm

        from ..data import wsdeval as wsde

        variant = (
            eval(f"wsde.Variant.{args.variant.upper()}")
            if args.variant
            else wsde.Variant.ALL
        )

        _, sentences = wsde.load(variant, args.path or wsde.DATA_ROOT)

        lemmata: dict[str, set[str]] = defaultdict(set)
        lemmata_count: dict[str, int] = defaultdict(int)
        for sentence in tqdm(sentences):
            for inst_idx, inst_pos in enumerate(sentence.instance_positions):
                lemma = sentence.lemmas[inst_pos]
                labels = sentence.instance_labels[inst_idx]

                lemmata_count[lemma] += 1
                lemmata[lemma] |= set(labels)

        out = validate_and_create_dir(args.out)

        with open(
            out / f"wsdeval_{args.variant or 'all'}_lemmata.csv", "w", encoding="utf-8"
        ) as file:
            for lemma, labels in lemmata.items():
                file.write(f"{lemma},{lemmata_count[lemma]},{','.join(labels)}\n")
