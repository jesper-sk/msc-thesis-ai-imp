# %%

# Third-party imports
import torch
from bertvectoriser import BertVectoriser
from torch import Tensor

# Local imports
import data.coarsewsd20 as cwsd
from data.entry import transpose_entries
from util.helpers import batched

# %%

data = cwsd.load_dataset(cwsd.Variant.REGULAR)
v = BertVectoriser()
batches = batched(data["chair"].train, 64)

# %%
