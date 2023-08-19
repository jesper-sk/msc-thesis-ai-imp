# %%
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from lxml import etree

from .tokens import TokenInput, TokenInputConvertible

DATA_ROOT = Path("data/wsdeval/WSD_Evaluation_Framework")


class Variant(Enum):
    ALL = "Evaluation_Datasets/ALL/ALL.{dataset}"
    SEMEVAL2007 = "Evaluation_Datasets/semeval2007/semeval2007.{dataset}"
    SEMEVAL2013 = "Evaluation_Datasets/semeval2013/semeval2013.{dataset}"
    SEMEVAL2015 = "Evaluation_Datasets/semeval2015/semeval2015.{dataset}"
    SENSEVAL2 = "Evaluation_Datasets/senseval2/senseval2.{dataset}"
    SENESVAL3 = "Evaluation_Datasets/senseval3/senseval3.{dataset}"
    SEMCOR = "Training_Corpora/SemCor/semcor.{dataset}"
    SEMCOR_OMSTI = "Training_Corpora/SemCor+OMSTI/semcor+omsti.{dataset}"

    def paths(self, root: Path) -> tuple[Path, Path]:
        return (
            root / self.value.format(dataset="data.xml"),
            root / self.value.format(dataset="gold.key.txt"),
        )


@dataclass
class Instance(TokenInputConvertible):
    identifier: str
    words: list[str]
    target_position: int
    target_labels: list[str]

    def to_tokens(self):
        return TokenInput(self.words, self.target_position)


@dataclass
class Sentence:
    identifier: str
    lemmas: list[str]
    poss: list[str]
    words: list[str]
    instance_positions: list[int]
    instance_identifiers: list[str]
    instance_labels: list[list[str]]

    def unpack(
        self,
    ) -> tuple[
        str, list[str], list[str], list[str], list[int], list[str], list[list[str]]
    ]:
        return (
            self.identifier,
            self.lemmas,
            self.poss,
            self.words,
            self.instance_positions,
            self.instance_identifiers,
            self.instance_labels,
        )

    def instances(self):
        return [
            Instance(identifier, self.words, position, label)
            for identifier, position, label in zip(
                self.instance_identifiers, self.instance_positions, self.instance_labels
            )
        ]


Dataset = tuple[dict[str, list[str]], list[Sentence]]


@dataclass
class Sentences:
    identifiers: tuple[str]
    lemmas: tuple[list[str]]
    poss: tuple[list[str]]
    words: tuple[list[str]]
    instance_positions: tuple[list[int]]
    instances_identifiers: tuple[list[str]]
    instance_labels: tuple[list[list[str]]]

    @classmethod
    def make(cls, sentences: Iterable[Sentence]) -> Sentences:
        return cls(*zip(*map(lambda x: x.unpack(), sentences)))  # type: ignore


def load_gold(path: Path) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as file:
        return {
            items[0]: items[1:]
            for items in map(lambda x: x.strip().split(), file.readlines())
        }


def load_sentences(xml_path: Path, gold_map: dict[str, list[str]]) -> list[Sentence]:
    sentences = list()
    for _, sentence in etree.iterparse(xml_path, tag="sentence"):
        identifier = sentence.attrib["id"]
        words = list()
        lemmas = list()
        poss = list()
        instances = list()
        identifiers = list()
        labels = list()
        for idx, element in enumerate(el for el in sentence if el.text is not None):
            words.append(element.text)
            lemmas.append(element.attrib["lemma"])
            poss.append(element.attrib["pos"])

            if element.tag == "instance":
                if label := gold_map.get(element.attrib["id"]):
                    instances.append(idx)
                    identifiers.append(element.attrib["id"])
                    labels.append(label)

        sentences.append(
            Sentence(identifier, lemmas, poss, words, instances, identifiers, labels)
        )

    return sentences


def load(variant: Variant, root=DATA_ROOT) -> Dataset:
    return load_from_paths(*variant.paths(root))


def load_from_paths(xml_path: Path, gold_path: Path) -> Dataset:
    gold_map = load_gold(gold_path)
    sentences = load_sentences(xml_path, gold_map)

    return gold_map, sentences


# %%
