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
LayerMergeOperation: TypeAlias = Literal["sum"]


class BertVectoriser:
    def __init__(
        self,
        model_name_or_path: str | PathLike[Any] = "bert-base-uncased",
        preload: bool = True,
        model: Optional[Model] = None,
        tokenizer: Optional[Tokenizer] = None,
        layers_of_interest: Optional[list[int]] = None,
        subword_merge_operation: SubwordMergeOperation = "first",
        layer_merge_operation: LayerMergeOperation = "sum",
    ):
        self.model: Model
        self.tokenizer: Tokenizer
        self.model_name_or_path: str | PathLike[Any]
        self.layers_of_interest: list[int] = layers_of_interest or [-1]

        self.merge_subword_tokens = self.get_subword_merge_function(
            subword_merge_operation
        )
        self.merge_layers = self.get_layer_merge_function(layer_merge_operation)

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

    def _is_loaded(self) -> bool:
        return hasattr(self, "model") and hasattr(self, "tokenizer")

    def ensure_models_loaded(self) -> None:
        if not self._is_loaded():
            self.tokenizer = trans.BertTokenizerFast.from_pretrained(
                self.model_name_or_path
            )
            self.model = trans.BertModel.from_pretrained(self.model_name_or_path)  # type: ignore

        if not self.tokenizer.is_fast:
            raise ValueError(
                "Tokenizer should be a 'fast' variant (e.g. BertTokenizerFast)"
            )

    @classmethod
    def get_subword_merge_function(
        cls, operation: SubwordMergeOperation
    ) -> Callable[[Tensor], Tensor]:
        try:
            return getattr(cls, f"merge_subword_{operation}")
        except AttributeError:
            raise ValueError(
                f"Invalid subword merge operation: cannot find function `cls.merge_subword_{operation}`"
                " corresponding to operation '{operation}'"
            )

    @classmethod
    def get_layer_merge_function(
        cls, operation: LayerMergeOperation
    ) -> Callable[[list[Tensor]], Tensor]:
        try:
            return getattr(cls, f"merge_layers_{operation}")
        except AttributeError:
            raise ValueError(
                f"Invalid layer merge operation: cannot find function `cls.merge_layers_{operation}`"
                " corresponding to operation '{operation}'"
            )

    @staticmethod
    def merge_layers_sum(layers: list[Tensor]) -> Tensor:
        return sum(layers, Tensor([0]))

    @staticmethod
    def merge_subword_first(layer: Tensor) -> Tensor:
        return layer[0, :]

    @staticmethod
    def merge_subword_mean(layer: Tensor) -> Tensor:
        return layer.mean(dim=0)

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

    def get_layer(self, model_output: ModelOutput, layer: int) -> Tensor:
        return model_output["hidden_states"][layer]

    def get_target_subword_token_ranges(
        self,
        get_word_ids: Callable[[int], list[int | None]],
        target_word_ids: list[int],
        num_sentences: int,
    ) -> list[tuple[int, int]]:
        def get_target_subword_token_range(
            word_ids: list[int | None], target_word_id: int
        ):
            all_indices = [
                token_index
                for token_index, word_id in enumerate(word_ids)
                if word_id == target_word_id
            ]
            return all_indices[0], all_indices[-1] + 1

        return [  # list[(start_idx, end_idx)]
            get_target_subword_token_range(
                word_ids=get_word_ids(entry_id),
                target_word_id=target_word_ids[entry_id],
            )
            for entry_id in range(num_sentences)
        ]

    def __call__(self, data: list[Entry], batch_size: int = 1) -> Iterator[Tensor]:
        self.ensure_models_loaded()

        for batch in batched(data, batch_size):
            entries = transpose_entries(batch)
            encoded = self.encode_tokens(entries.tokens)
            target_token_indices = self.get_target_subword_token_ranges(
                encoded.word_ids, entries.target_word_ids, batch_size
            )
            model_output = self.model_pass(encoded)
            merged_embeddings = self.merge_layers(
                [
                    self.get_layer(model_output, layer_id)
                    for layer_id in self.layers_of_interest
                ]
            )

            output = torch.zeros(
                merged_embeddings.shape[0::2]
            )  # shape: (batch_size, embedding_dim)
            for sentence_index, (encoding, (start_index, end_index)) in enumerate(
                zip(merged_embeddings, target_token_indices)
            ):
                subword_tokens = encoding[start_index:end_index, :]
                output[sentence_index, :] = self.merge_subword_tokens(subword_tokens)

            yield output
