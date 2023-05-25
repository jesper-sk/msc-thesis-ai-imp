from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from ..data import VerticalEntries


@dataclass
class ClusterResult:
    confusion_matrix: np.ndarray
    accuracy: float
    precision: float
    recall: float
    f1: float


def test_kmeans_cluster(
    embeddings: np.ndarray, entries: VerticalEntries
) -> ClusterResult:
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

    return ClusterResult(
        confusion_matrix=confusion_matrix(actual, predicted),
        accuracy=k.score(embeddings),
        precision=precision_score(actual, predicted, average="macro"),
        recall=recall_score(actual, predicted, average="macro"),
        f1=f1_score(actual, predicted, average="macro"),
    )
