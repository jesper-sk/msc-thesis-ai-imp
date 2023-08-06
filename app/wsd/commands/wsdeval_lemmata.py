from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path

from ..data import wsdeval
from ..util.path import validate_existing_dir
from .command import Command


class ExtractLemmata(Command):
    @staticmethod
    def name() -> str:
        return "lemmas"

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "-f",
            "--framework",
            type=str,
            help="The framework to use: [wsdeval, xl-wsd, {path}]",
            default="wsdeval",
        )
        parser.add_argument(
            "-o",
            "--out",
            type=Path,
            help="The location for the output file",
            required=True,
        )
        parser.add_argument("-d", "--dataset", type=str, default=None)

    @staticmethod
    def run(args: Namespace) -> None:
        if args.framework == "wsdeval":
            path = Path("data/WSD_Evaluation_Framework/Evaluation_Datasets")
            dataset = args.dataset or "ALL"
        elif args.framework == "xl-wsd":
            path = Path("data/xl-wsd/evaluation_datasets")
            dataset = args.dataset or "test-en"
        else:
            path = Path(args.framework)
            dataset = args.dataset
            if not dataset:
                raise Exception("No dataset given")

        path /= args.dataset
        validate_existing_dir(path)

        gold = path / f"{args.dataset}.gold.key.txt"
        xml = path / f"{args.dataset}.data.xml"

        _, sentences = wsdeval.load(xml, gold)

        lemmata: dict[str, set[str]] = defaultdict(set)
        for sentence in sentences:
            for inst_idx, inst_pos in enumerate(sentence.instance_positions):
                lemma = sentence.lemmas[inst_pos]
                labels = sentence.instance_labels[inst_idx]

                lemmata[lemma] |= set(labels)

        with open(args.out, "w", encoding="utf-8") as file:
            for lemma, labels in lemmata.items():
                file.write(f"{lemma},{','.join(labels)}\n")
