from pathlib import Path

import numpy as np

from . import coarsewsd20 as cwsd


class CwsdEmbeddings:
    def __init__(self, data: cwsd.Dataset):
        self.data = data

    def load_cwsd_npy(
        self,
        word: cwsd.Word,
        split=None,
        path: Path = Path("out/vectorised/sensebert-base-uncased"),
    ) -> np.ndarray:
        if split:
            return np.load(path / f"{word}.{split}.npy")
        return np.concatenate(
            (np.load(path / f"{word}.train.npy"), np.load(path / f"{word}.test.npy"))
        )

    def load_cwsd_word_embeddings_per_class(
        self,
        word: cwsd.Word,
        split=None,
        path: Path = Path("out/vectorised/sensebert-base-uncased"),
    ) -> dict[str, np.ndarray]:
        embeddings = self.load_cwsd_npy(word, split, path)
        class_ids = np.array(self.data[word].vertical(split).target_class_ids)
        classes = self.data[word].classes

        return {
            classes[str(current_id)]: embeddings[class_ids == current_id]
            for current_id in np.unique(class_ids)
        }
