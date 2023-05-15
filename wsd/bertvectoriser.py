from os import PathLike
from typing import Any, Callable, Literal, Optional

import torch
import transformers as trans
from data.entry import Entry
from tqdm import tqdm
from transformers.utils.generic import ModelOutput, PaddingStrategy, TensorType

Tokens = list[str]
Tokenizer = trans.BertTokenizerFast | trans.BertTokenizer
TokenizerOutput = trans.tokenization_utils.BatchEncoding
Model = trans.BertModel
SubwordMergeFunction = Callable[[torch.Tensor, int, int], torch.Tensor]
SubwordMergeOperation = Literal["first", "mean"]


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

        self.merge_subword: SubwordMergeFunction = self.get_subword_merge_function(
            merge_operation
        )
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
                self.load_models()

    @classmethod
    def get_subword_merge_function(
        cls, operation: SubwordMergeOperation
    ) -> SubwordMergeFunction:
        match operation:
            case "first":
                return cls.merge_subword_first
            case "mean":
                return cls.merge_subword_mean
            case _:
                raise ValueError(f"Unknown merge operation: {operation}")

    def _is_loaded(self) -> bool:
        return hasattr(self, "model") and hasattr(self, "tokenizer")

    def load_models(self):
        if not self._is_loaded():
            self.tokenizer = trans.BertTokenizerFast.from_pretrained(
                self.model_name_or_path
            )
            self.model = trans.BertModel.from_pretrained(self.model_name_or_path)  # type: ignore

    @staticmethod
    def merge_subword_first(layer: torch.Tensor, start: int, _: int) -> torch.Tensor:
        return layer[:, start, :]

    @staticmethod
    def merge_subword_mean(layer: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return layer[:, start:end, :].mean(dim=1)

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
        layer: torch.Tensor,
        target_word_id: int,
        word_ids: list[Any],
    ) -> torch.Tensor:
        indices = [
            token_index
            for token_index, word_id in enumerate(word_ids)
            if word_id == target_word_id
        ]
        return self.merge_subword(layer, indices[0], indices[-1])

    def get_layers_of_interest(self, model_output: ModelOutput) -> list[torch.Tensor]:
        return [
            model_output["hidden_states"][layer] for layer in self.layers_of_interest
        ]

    def __call__(self, data: list[Entry]):
        self.load_models()

        embeddings = []
        for entry in tqdm(data):
            encoded = self.encode_tokens(entry.tokens)
            model_output = self.model_pass(encoded)
            subword_embeddings = [
                self.merge_subword_embeddings(
                    layer, entry.target_word_id, encoded.word_ids()
                )
                for layer in self.get_layers_of_interest(model_output)
            ]
            combined_embeddings = sum(subword_embeddings)
            embeddings.append(combined_embeddings)

        return embeddings
