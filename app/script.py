# %%

# Third-party imports
import torch
from torch import Tensor

# Local imports
import data.coarsewsd20 as cwsd
from data.entry import transpose_entries
from util.helpers import batched
from vectorise.bert import BertVectoriser

# %%

data = cwsd.load_dataset(cwsd.Variant.REGULAR)
v = BertVectoriser()
gen = v(data["pitcher"].train, 64)

# %%
