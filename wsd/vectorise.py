import logging
from typing import Iterator

import coarsewsd20 as cwsd
import transformers as trans
from transformers.tokenization_utils import BatchEncoding
from transformers.utils.generic import PaddingStrategy, TensorType

CoarseWSD20DataSet = dict[str, cwsd.WordDataset]
Tokenizer = trans.BertTokenizerFast | trans.BertTokenizerFast
Model = trans.BertModel
Encoder = Iterator[tuple[str, BatchEncoding]]


def encode(
    tokenizer: Tokenizer, data: CoarseWSD20DataSet, split: cwsd.Split
) -> BatchEncoding:
    # for word in data.keys():
    logging.info("Tokenizing %s split...", split)
    word_tokens = [
        tokens
        for word in data.keys()
        for tokens in data[word].tokens(split)
        if len(tokens) <= tokenizer.model_max_length - 2  # [CLS] and [SEP]
    ]

    # if (diff := len(data[word].split(split)) - len(word_tokens)) > 0:
    #     logging.warning(
    #         "Removed %s entries for %s in %s, as they exceeded the maximum length of %s.",
    #         diff,
    #         word,
    #         split,
    #         tokenizer.model_max_length,
    #     )

    return tokenizer(
        word_tokens,
        is_split_into_words=True,
        padding=PaddingStrategy.MAX_LENGTH,  # Pad to max input length of model
        return_tensors=TensorType.PYTORCH,
        return_attention_mask=True,  # Return attention masks to distinguish padding from input
    )


def vectorise(encoder: Encoder, model: Model):
    for word, batch in encoder:
        logging.info("Vectorising word '%s'...", word)
        yield word, model(**batch)
