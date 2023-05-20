# %%

# Third-party imports
import torch
from torch import Tensor

# First-party imports
import wsd.data.coarsewsd20 as cwsd
from wsd.data.entry import transpose_entries
from wsd.util.helpers import batched
from wsd.vectorise.bert import BertVectoriser

# %%

data = cwsd.load_dataset(cwsd.Variant.REGULAR, root="../data/CoarseWSD-20")[
    "seal"
].train
entries = transpose_entries(data)

# Third-party imports
# %%
import matplotlib.pyplot as plt
import numpy as np

umap_emb = np.load("../umap_seal_train.npy")

plt.scatter(
    umap_emb[:, 0],
    umap_emb[:, 1],
    c=[["red", "blue", "green", "purple"][x] for x in entries.target_class_ids],
)
plt.gca().set_aspect("equal", "datalim")
plt.title("UMAP projection of CWSD sentences 'chair'", fontsize=24)
# %%
