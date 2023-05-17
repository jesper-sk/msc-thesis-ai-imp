# Standard library
from os import PathLike
from typing import Any, Callable, Iterator, Literal, Optional, TypeAlias

# Third-party imports
import torch
import transformers as trans
from torch import Tensor
from transformers.utils.generic import ModelOutput, PaddingStrategy, TensorType

# Local imports
from data.entry import Entry, transpose_entries
from util.helpers import batched

# Type aliases
Tokens: TypeAlias = list[str]
Tokenizer: TypeAlias = trans.BertTokenizerFast | trans.BertTokenizer
TokenizerOutput: TypeAlias = trans.tokenization_utils.BatchEncoding
Model: TypeAlias = trans.BertModel
SubwordMergeOperation: TypeAlias = Literal["first", "mean"]


class BertVectoriser:
    def __init__(
        self,
        model_name_or_path: str | PathLike[Any] = "bert-base-uncased",
        preload: bool = True,
        model: Optional[Model] = None,
        tokenizer: Optional[Tokenizer] = None,
        merge_operation: SubwordMergeOperation = "first",
        layers_of_interest: Optional[list[int]] = None,
    ):
        self.model: Model
        self.tokenizer: Tokenizer
        self.model_name_or_path: str | PathLike[Any]

        self.merge_subword = self.get_subword_merge_function(merge_operation)
        self.layers_of_interest: list[int] = layers_of_interest or [-1]

        if (model is None) ^ (tokenizer is None):
            raise ValueError(
                "Either both or neither of model and tokenizer must be provided"
            )

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        elif model_name_or_path is None:
            raise ValueError(
                "Either model_name_or_path must be provided or both model and tokenizer"
            )
        else:
            self.model_name_or_path = model_name_or_path
            if preload:
                self.ensure_models_loaded()

    @classmethod
    def get_subword_merge_function(
        cls, operation: SubwordMergeOperation
    ) -> Callable[[Tensor, int, int, int], Tensor]:
        match operation:
            case "first":
                return cls.merge_subword_first
            case "mean":
                return cls.merge_subword_mean
            case _:
                raise ValueError(f"Unknown merge operation: {operation}")

    def _is_loaded(self) -> bool:
        return hasattr(self, "model") and hasattr(self, "tokenizer")

    def ensure_models_loaded(self) -> None:
        if not self._is_loaded():
            self.tokenizer = trans.BertTokenizerFast.from_pretrained(
                self.model_name_or_path
            )
            self.model = trans.BertModel.from_pretrained(self.model_name_or_path)  # type: ignore

    @staticmethod
    def merge_subword_first(layer: Tensor, batch_id: int, start: int, _: int) -> Tensor:
        return layer[batch_id, start, :]  # TODO: Batch first or last dimension?

    @staticmethod
    def merge_subword_mean(
        layer: Tensor, batch_id: int, start: int, end: int
    ) -> Tensor:
        return layer[batch_id, start:end, :].mean(
            dim=1
        )  # TODO: Batch first or last dimension?

    def encode_tokens(self, tokens: Tokens | list[Tokens]) -> TokenizerOutput:
        return self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.LONGEST,
            return_tensors=TensorType.PYTORCH,
            return_attention_mask=True,  # Return attention masks to distinguish padding from input
        )

    def model_pass(self, encoded: TokenizerOutput) -> ModelOutput:
        return self.model(**encoded, output_hidden_states=True)

    def merge_subword_embeddings(
        self,
        layer: Tensor,
        target_word_id: int,
        word_ids: list[Any],
    ) -> Tensor:
        indices = [
            token_index
            for token_index, word_id in enumerate(word_ids)
            if word_id == target_word_id
        ]
        return self.merge_subword(layer, indices[0], indices[-1])

    def get_layer(self, model_output: ModelOutput, layer: int):
        return model_output["hidden_states"][layer]

    def get_target_subword_token_range(
        self, word_ids: list[int | None], target_word_id: int
    ):
        all_indices = [
            token_index
            for token_index, word_id in enumerate(word_ids)
            if word_id == target_word_id
        ]
        return all_indices[0], all_indices[-1] + 1

    def __call__(self, data: list[Entry], batch_size: int = 1) -> Iterator[Tensor]:
        self.ensure_models_loaded()

        for batch in batched(data, batch_size):
            entries = transpose_entries(batch)
            encoded = self.encode_tokens(entries.tokens)
            target_token_indices = torch.Tensor(
                [
                    self.get_target_subword_token_range(
                        word_ids=encoded.word_ids(entry_id),
                        target_word_id=entries.target_word_ids[entry_id],
                    )
                    for entry_id in range(batch_size)
                ],
                dtype=torch.int32,
            )

            model_output = self.model_pass(encoded)
            layers = [
                self.get_layer(model_output, layer_id)
                for layer_id in self.layers_of_interest
            ]
            merged = sum(layers, Tensor(0))

            output = t

            for sentence_encoding in merged:
                pass

            subword_embeddings = [
                self.merge_subword_embeddings(
                    layer=self.get_layer(layer, model_output),
                    target_word_id=entries.target_word_ids,
                )
                for layer in self.get_layers_of_interest(model_output)
            ]
            yield sum(subword_embeddings, Tensor(0))
