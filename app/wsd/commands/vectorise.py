from argparse import ArgumentParser, Namespace

from .command import Command


class Vectorise(Command):
    @staticmethod
    def name() -> str:
        return "vectorise"

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        # parser.add_argument(
        #     "data", choices=["coarsewsd"], help="the type of dataset to vectorise"
        # )
        parser.add_argument(
            "-m",
            "--model",
            type=str,
            default="bert-base-uncased",
            help="The transformers model and tokenizer to use with vectorisation",
        )
        parser.add_argument(
            "-b",
            "--batchsize",
            type=int,
            default=24,
            help="the batch size to use (default 24)",
        )
        # parser.add_argument("-p", "--path", type=Path, help="the path to the dataset")
        parser.add_argument(
            "-o",
            "--out",
            type=str,
            default=None,
            help="the output directory",
        )
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="cpu",
            help="the device to use (cpu, cuda, cuda:0, etc)",
        )

    @staticmethod
    def run(args: Namespace) -> None:
        import numpy as np

        from ..data import coarsewsd20 as cwsd
        from ..util.path import validate_and_create_dir
        from ..vectorise import vectorise_coarsewsd20
        from ..vectorise.bert import BertVectoriser

        vectoriser = BertVectoriser(model_name_or_path=args.model, device=args.device)
        dataset = cwsd.load_dataset(cwsd.Variant.REGULAR)
        out_path = validate_and_create_dir(args.out or f"./out/vectorised/{args.model}")

        for key, embedding in vectorise_coarsewsd20(
            vectoriser, dataset, args.batchsize
        ):
            fn = out_path / f"{key}.npy"
            np.save(fn, embedding)
