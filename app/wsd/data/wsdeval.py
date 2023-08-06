# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from lxml import etree


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


def load_sentences(xml_path: Path, gold_map: dict[str, list[str]]):
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


def load(xml_path: Path, gold_path: Path):
    gold_map = load_gold(gold_path)
    sentences = load_sentences(xml_path, gold_map)

    return gold_map, sentences


# %%
