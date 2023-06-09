from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace, _SubParsersAction


class Command(ABC):
    @classmethod
    def register(cls, parser: _SubParsersAction):
        subparser = parser.add_parser(cls.name())
        subparser.set_defaults(action=cls.run)
        cls.add_arguments(subparser)

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass

    @staticmethod
    @abstractmethod
    def add_arguments(parser: ArgumentParser) -> None:
        pass

    @staticmethod
    @abstractmethod
    def run(args: Namespace) -> None:
        pass
