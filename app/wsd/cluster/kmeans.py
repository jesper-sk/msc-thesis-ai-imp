from dataclasses import dataclass

import numpy as np
import sklearn.metrics as m
from sklearn.cluster import KMeans

from ..data import VerticalEntries


@dataclass
class ClusterResult:
    confusion_matrix: np.ndarray
    accuracy: float
    precision: float
    recall: float
    f1: float


def test_kmeans_cluster(embeddings: np.ndarray, entries: VerticalEntries) -> float:
    """Performs k-means clustering on the given embeddings.

    Parameters
    ----------
    `embeddings : np.ndarray`
        The embeddings to cluster.
    `entries : VerticalEntries`
        The entries corresponding to the embeddings.

    Returns
    -------
    `ClusterResult`
        The result of the clustering.
    """

    k = KMeans(n_clusters=len(np.unique(entries.target_class_ids)))
    k.fit(embeddings)

    predicted = k.labels_
    actual = entries.target_class_ids

    return m.v_measure_score(actual, predicted)
