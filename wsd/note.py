# %%
import coarsewsd20 as cwsd
# %%
import torch
from transformers import BertConfig, BertModel, BertTokenizer

# %%
config = BertConfig.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", config=config)

# %%

model.eval()

# %%
texts = [
    "Hello, my dog is cute",
    "Hi, their cat is very cute",
]

pad_encoding = tokenizer.encode(
    texts, padding=True, truncation=True, max_length=512)

# TODO: CWSD20 is already tokenized, but needs to be encoded
# Already includes [CLS] and [SEP]
tokens = [tokenizer.encode(text) for text in texts]
# tokens = torch.tensor([tokens]) # Create tensor to for input to model
# %%
1
