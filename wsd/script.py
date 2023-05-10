# %%
import coarsewsd20 as cwsd
from transformers import BertModel, BertTokenizerFast
from vectorise import *

# %%

data = cwsd.load_dataset(cwsd.Variant.REGULAR)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
encoder = encode(tokenizer, data, "train")

model = BertModel.from_pretrained("bert-base-uncased")
vectoriser = vectorise(encoder, model)


# %%
foo = [x for x in encode(tokenizer, data, "train")]
# %%


def gen():
    i = 0
    while True:
        stop = yield "Hey"
        print(stop, type(stop))
        if stop:
            return "J'arrive pas à m'arrêter !"
        yield i
        i += 1


# %%
