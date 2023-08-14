from argparse import ArgumentParser, Namespace
from pathlib import Path

from .command import Command


class Vectorise(Command):
    @staticmethod
    def name() -> str:
        return "vectorise"

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "data",
            choices=["coarsewsd", "wsdeval"],
            help="the type of dataset to vectorise: coarsewsd=CoarseWSD-20; wsdeval=WSD Evaluation Framework",
        )
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
        parser.add_argument(
            "-p", "--path", type=Path, help="the root path of the dataset", default=None
        )
        parser.add_argument(
            "-o",
            "--out",
            type=str,
            default=None,
            help="the output directory",
        )
        parser.add_argument(
            "-v",
            "--variant",
            type=str,
            help="The dataset variant to vectorise, defaults: REGULAR (for coarsewsd), ALL (for wsdeval)",
            default=None,
        )
        parser.add_argument(
            "-d",
            "--device",
            type=str,
            default="cpu",
            help="the device to use (cpu, cuda, cuda:0, etc)",
        )

    @staticmethod
    def coarsewsd(args: Namespace, vectoriser, out_path: Path) -> None:
        import numpy as np

        from ..data import coarsewsd20 as cwsd
        from ..vectorise import vectorise_coarsewsd20

        variant = (
            eval(f"cwsd.Variant.{args.variant.upper()}")
            if args.variant
            else cwsd.Variant.REGULAR
        )

        dataset = cwsd.load_dataset(variant, args.path or cwsd.DATA_ROOT)
        vectoriser.load_prepare_models()

        for key, embedding in vectorise_coarsewsd20(
            vectoriser, dataset, args.batchsize
        ):
            fn = out_path / f"{key}.npy"
            np.save(fn, embedding)

    @staticmethod
    def wsdeval(args: Namespace, vectoriser, out_path: Path) -> None:
        import numpy as np

        from ..data import wsdeval as wsde
        from ..vectorise import vectorise_wsdeval

        variant = (
            eval(f"wsde.Variant.{args.variant.upper()}")
            if args.variant
            else wsde.Variant.ALL
        )

        goldmap, sentences = wsde.load(variant, args.path or wsde.DATA_ROOT)

        instances, embeddings = vectorise_wsdeval(vectoriser, sentences, args.batchsize)
        fn = out_path / "embeddings.npy"
        np.save(fn, embeddings)

        with open(out_path / "labels.txt", "w") as file:
            file.write("\n".join([instance.identifier for instance in instances]))

    @classmethod
    def run(cls, args: Namespace) -> None:
        from ..util.path import validate_and_create_dir
        from ..vectorise.bert import BertVectoriser

        vectoriser = BertVectoriser(
            model_name_or_path=args.model, device=args.device, preload=False
        )
        out_path = validate_and_create_dir(
            args.out
            or f"./out/vectorised/{args.data}_{args.model}_{args.variant or 'default'}"
        )

        {
            "coarsewsd": cls.coarsewsd,
            "wsdeval": cls.wsdeval,
        }[
            args.data
        ](args, vectoriser, out_path)
