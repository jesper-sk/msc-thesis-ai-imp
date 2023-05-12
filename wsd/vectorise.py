from typing import Any, Callable

import coarsewsd20 as cwsd
import torch
import transformers as trans
from transformers.utils.generic import ModelOutput, PaddingStrategy, TensorType

Tokens = list[str]
Tokenizer = trans.BertTokenizerFast | trans.BertTokenizer
TokenizerOutput = trans.tokenization_utils.BatchEncoding
Model = trans.BertModel
CoarseWSD20 = dict[str, cwsd.WordDataset]
SubwordMergeOperation = Callable[[torch.Tensor, int, int], torch.Tensor]


def encode_tokens(
    tokenizer: Tokenizer, tokens: Tokens | list[Tokens]
) -> TokenizerOutput:
    return tokenizer(
        tokens,
        is_split_into_words=True,
        padding=PaddingStrategy.LONGEST,  # Pad to max input length of model
        return_tensors=TensorType.PYTORCH,
        return_attention_mask=True,  # Return attention masks to distinguish padding from input
    )


def model_pass(model: Model, encoded: TokenizerOutput) -> ModelOutput:
    return model(**encoded, output_hidden_states=True)


def merge_subword_first(layer: torch.Tensor, start: int, _: int) -> torch.Tensor:
    return layer[:, start, :]


def merge_subword_mean(layer: torch.Tensor, start: int, end: int) -> torch.Tensor:
    return layer[:, start:end, :].mean(dim=1)


def merge_subword_embeddings(
    layer: torch.Tensor,
    encoded: TokenizerOutput,
    target_word_id: int,
    merge_operation: SubwordMergeOperation,
) -> torch.Tensor:
    indices = [
        token_index
        for token_index, word_id in enumerate(encoded.word_ids())
        if word_id == target_word_id
    ]
    return merge_operation(layer, indices[0], indices[-1])


def merge_subword_embeddings_for_layers(
    model_output: ModelOutput, entry: cwsd.Entry, layers_of_interest: list[int]
) -> Any:
    for layer in (model_output["hidden_states"][i] for i in layers_of_interest):
        merge_subword_embeddings(layer, )


def vectorise_coarsewsd20(
    dataset: cwsd.Dataset, split: cwsd.Split, tokenizer: Tokenizer, model: Model
):
    for word in cwsd.WORDS:
        for entry in dataset[word].split(split):
            encoded = encode_tokens(tokenizer, entry.tokens)
            model_output = model_pass(model, encoded)
