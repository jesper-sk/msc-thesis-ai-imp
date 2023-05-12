# %%
import coarsewsd20 as cwsd
from transformers import BertModel, BertTokenizerFast
from foo import BertVectoriser

# %%

data = cwsd.load_dataset(cwsd.Variant.REGULAR)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# %%
