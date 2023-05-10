# %%
import coarsewsd20 as cwsd
from encode import encode

data = cwsd.load_dataset(cwsd.Variant.REGULAR)

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
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
