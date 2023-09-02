import typing
from collections import defaultdict
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from tqdm import tqdm

from ..conceptors import Conceptor

DATA_ROOT = Path("data/bookcorpus")

Lemma: TypeAlias = Literal[
    "agribusiness",
    "alga",
    "ambrosia",
    "ancestor",
    "ancestry",
    "apple",
    "area",
    "arm",
    "aroma",
    "ash",
    "attic",
    "avocado",
    "banana",
    "bean",
    "beech",
    "beginning",
    "berry",
    "blackberry",
    "blight",
    "blush_wine",
    "boodle",
    "branch",
    "bud",
    "bundle",
    "bunk",
    "cabbage",
    "capsicum",
    "carbohydrate",
    "care",
    "chocolate",
    "chrysanthemum",
    "citron",
    "coconut",
    "coffee",
    "coffee_bean",
    "corn",
    "corn_whiskey",
    "country",
    "crush",
    "daffodil",
    "daisy",
    "decomposition",
    "decoration",
    "department_of_agriculture",
    "dessert",
    "dogwood",
    "eatage",
    "eggplant",
    "elm",
    "emergence",
    "entity",
    "etymon",
    "farm",
    "farming",
    "fig",
    "figure",
    "fix",
    "flower",
    "foliation",
    "forest",
    "freak",
    "fruit",
    "gamboge",
    "garden",
    "gelatin",
    "grape",
    "grapeshot",
    "grass",
    "green",
    "greenhouse",
    "greens",
    "growth",
    "hay",
    "hayfield",
    "herb",
    "home",
    "honeysuckle",
    "hyacinth",
    "increase",
    "jam",
    "jamming",
    "jelly",
    "landscape",
    "larkspur",
    "lawn",
    "leaf",
    "lemon",
    "lilac",
    "lineage",
    "luggage_compartment",
    "lumber",
    "maple",
    "matter",
    "melon",
    "monstrosity",
    "moss",
    "myrtle",
    "nation",
    "nectar",
    "oleander",
    "olfactory_property",
    "orange",
    "organism",
    "outgrowth",
    "palm",
    "park",
    "pasture",
    "pea",
    "pear",
    "peony",
    "pepper",
    "perfume",
    "petal",
    "pieplant",
    "pine",
    "place",
    "plant",
    "planting",
    "plaza",
    "plowing",
    "pod",
    "poppy",
    "position",
    "pot",
    "proboscis",
    "putrefaction",
    "radish",
    "radish_plant",
    "rocket",
    "root",
    "rose",
    "sage",
    "salad",
    "scent",
    "seat",
    "seed",
    "seeded_player",
    "semen",
    "shrub",
    "shrubbery",
    "skyrocket",
    "solution",
    "source",
    "space",
    "sphere",
    "spice",
    "spiciness",
    "state",
    "stead",
    "strawberry",
    "sugar",
    "supergrass",
    "sweet",
    "thing",
    "timber",
    "timbre",
    "tobacco",
    "tomato",
    "topographic_point",
    "torso",
    "tree",
    "trunk",
    "tulip",
    "turf",
    "vegetable",
    "verbena",
    "violet",
    "vitamin",
    "wallflower",
    "weed",
    "willow",
    "wood",
    "woodwind",
    "yield",
]

LEMMATA = typing.get_args(Lemma)


def get_synset_embeddings(
    directory: str | Path = DATA_ROOT / "prepared",
    do_print=True,
) -> dict[Lemma, dict[int, np.ndarray]]:
    directory = Path(directory)
    assert directory.exists() and directory.is_dir()

    embedding_map: dict[str, dict[int, np.ndarray]] = defaultdict(dict)

    wrap = tqdm if do_print else lambda x: x

    for file in wrap(directory.glob("*.embeddings.npy")):
        components = file.stem.split(".")
        lemma = components[0]
        if lemma not in LEMMATA:
            print(f"Warning: found unexpected synset {lemma}")

        idx = int(components[2])

        embedding_map[lemma][idx] = np.load(file)

    found_synsets = list(embedding_map.keys())
    for lemma in LEMMATA:
        if lemma not in found_synsets:
            print(f"Warning: could not find synset {lemma}")

    return embedding_map  # type: ignore


def get_synset_conceptors(
    aperture: float = 0.58,
    directory: str | Path = DATA_ROOT / "prepared",
    synset_embeddings: dict[Lemma, dict[int, np.ndarray]] | None = None,
    do_print=True,
) -> dict[Lemma, dict[int, Conceptor]]:
    wrap = tqdm if do_print else lambda x: x
    return {
        lemma: {
            idx: Conceptor.from_state_matrix(matrix, aperture)
            for idx, matrix in matrices.items()
        }
        for lemma, matrices in wrap(
            (
                synset_embeddings or get_synset_embeddings(directory, do_print=False)
            ).items()
        )
    }
