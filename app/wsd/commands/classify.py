from argparse import ArgumentParser, Namespace

from .command import Command


class Classify(Command):
    @staticmethod
    def name() -> str:
        return "classify"

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        # parser.add_argument(
        #     "data", choices=["coarsewsd"], help="the type of dataset to vectorise"
        # )
        parser.add_argument(
            "-c", "--checkpoint", type=str, help="the ewiser checkpoint to use"
        )
        parser.add_argument(
            "-s",
            "--spacy",
            type=str,
            default="en_core_web_sm",
            help="The spacy model to use",
        )
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="cpu",
            help="the device to use (cpu, cuda, cuda:0, etc)",
        )
        parser.add_argument(
            "-o",
            "--out",
            type=str,
            default=None,
            help="the output directory",
        )

    @staticmethod
    def run(args: Namespace) -> None:
        from pathlib import Path

        from ..classify.ewiser import EwiserClassifier
        from ..util.path import is_valid_directoy

        out_path = Path(args.out if args.out else "./out/classified/ewiser")
        assert is_valid_directoy(out_path)

        classifier = EwiserClassifier(args.checkpoint, args.spacy, args.device, "en")
        classifier.import_load()
