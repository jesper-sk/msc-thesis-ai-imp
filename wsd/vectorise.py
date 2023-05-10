import logging
from typing import Iterator

import coarsewsd20 as cwsd
import transformers as trans
from transformers.tokenization_utils import BatchEncoding
from transformers.utils.generic import PaddingStrategy, TensorType

CoarseWSD20DataSet = dict[str, cwsd.WordDataset]
Tokenizer = trans.PreTrainedTokenizerFast | trans.PreTrainedTokenizer


def encode(
    tokenizer: Tokenizer, data: CoarseWSD20DataSet, split: cwsd.Split
) -> Iterator[tuple[str, BatchEncoding]]:
    for word in data.keys():
        logging.info("Processing word '%s' in %s...", word, split)
        word_tokens = [
            tokens
            for tokens in data[word].tokens(split)
            if len(tokens) <= tokenizer.model_max_length - 2  # [CLS] and [SEP]
        ]

        if (diff := len(data[word].split(split)) - len(word_tokens)) > 0:
            logging.warning(
                "Removed %s entries for %s in %s, as they exceeded the maximum length of %s.",
                diff,
                word,
                split,
                tokenizer.model_max_length,
            )

        yield word, tokenizer(
            word_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.MAX_LENGTH,
            return_tensors=TensorType.PYTORCH,
        )


def foo():
    pass
