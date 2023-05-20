# Standard library
from os import PathLike
from typing import Any, Callable, Iterator, Literal, Optional, Sequence, TypeAlias

# Third-party imports
import torch
import transformers as trans
from torch import Tensor
from transformers.utils.generic import ModelOutput, PaddingStrategy, TensorType

# Local imports
from ..data.entry import Entry, transpose_entries
from ..util.helpers import batched
from . import Vectoriser

# Type aliases
Tokens: TypeAlias = list[str]
Tokenizer: TypeAlias = trans.BertTokenizerFast | trans.BertTokenizer
TokenizerOutput: TypeAlias = trans.tokenization_utils.BatchEncoding
Model: TypeAlias = trans.BertModel
SubwordMergeOperation: TypeAlias = Literal["first", "mean"]
LayerMergeOperation: TypeAlias = Literal["sum"]


class BertVectoriser(Vectoriser):
    def __init__(
        self,
        model_name_or_path: str | PathLike[Any] = "bert-base-uncased",
        preload: bool = True,
        model: Optional[Model] = None,
        tokenizer: Optional[Tokenizer] = None,
        layers_of_interest: Optional[list[int]] = None,
        subword_merge_operation: SubwordMergeOperation = "first",
        layer_merge_operation: LayerMergeOperation = "sum",
        device: str | None = None,
    ):
        self.model: Model
        self.tokenizer: Tokenizer
        self.model_name_or_path: str | PathLike[Any]
        self.layers_of_interest: list[int] = layers_of_interest or [-1]
        self.device = device

        self.merge_subword_tokens = self._get_subword_merge_function(
            subword_merge_operation
        )
        self.merge_layers = self._get_layer_merge_function(layer_merge_operation)

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
                self.load_prepare_models()

    def _is_loaded(self) -> bool:
        return hasattr(self, "model") and hasattr(self, "tokenizer")

    def load_prepare_models(self) -> None:
        if not self._is_loaded():
            self.tokenizer = trans.BertTokenizerFast.from_pretrained(
                self.model_name_or_path
            )
            self.model = trans.BertModel.from_pretrained(self.model_name_or_path)  # type: ignore

        if not self.tokenizer.is_fast:
            raise ValueError(
                "Tokenizer should be a 'fast' variant (e.g. BertTokenizerFast)"
            )

        self.model.eval()
        if self.device:
            self.model.to("cuda")

    def to(self, device: str) -> None:
        self.device = device

    def _get_subword_merge_function(
        self, operation: SubwordMergeOperation
    ) -> Callable[[Tensor], Tensor]:
        try:
            return getattr(self, f"_merge_subword_{operation}")
        except AttributeError:
            raise ValueError(
                f"Invalid subword merge operation: cannot find function `self._merge_subword_{operation}`"
                " corresponding to operation '{operation}'"
            )

    def _merge_subword_first(self, layer: Tensor) -> Tensor:
        return layer[0, :]

    def _merge_subword_mean(self, layer: Tensor) -> Tensor:
        return layer.mean(dim=0)

    def _get_layer_merge_function(
        self, operation: LayerMergeOperation
    ) -> Callable[[list[Tensor]], Tensor]:
        try:
            return getattr(self, f"_merge_layers_{operation}")
        except AttributeError:
            raise ValueError(
                f"Invalid layer merge operation: cannot find function `self._merge_layers_{operation}`"
                " corresponding to operation '{operation}'"
            )

    def _merge_layers_sum(self, layers: list[Tensor]) -> Tensor:
        return sum(layers, self.tensor([0]))

    def tensor(self, data, *, dtype=None):
        return torch.tensor(data, dtype=dtype, device=self.device)

    def encode_tokens(self, tokens: Tokens | Sequence[Tokens]) -> TokenizerOutput:
        output = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.LONGEST,
            return_tensors=TensorType.PYTORCH,
            return_attention_mask=True,  # Return attention masks to distinguish padding from input
        )
        if self.device:
            return output.to(self.device)
        return output

    def model_pass(self, encoded: TokenizerOutput) -> ModelOutput:
        return self.model(**encoded, output_hidden_states=True)

    def get_layer(self, model_output: ModelOutput, layer: int) -> Tensor:
        return model_output["hidden_states"][layer]

    def get_target_subword_token_ranges(
        self,
        get_word_ids: Callable[[int], list[int | None]],
        target_word_ids: Sequence[int],
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

    def extract_target_embeddings(
        self, merged_embeddings: Tensor, target_token_indices: list[tuple[int, int]]
    ) -> Tensor:
        ret = torch.zeros(
            merged_embeddings.shape[0::2]
        )  # shape: (batch_size, embedding_dim)

        for sentence_index, (encoding, (start_index, end_index)) in enumerate(
            zip(merged_embeddings, target_token_indices)
        ):
            subword_tokens = encoding[start_index:end_index, :]
            ret[sentence_index, :] = self.merge_subword_tokens(subword_tokens)

        return ret

    def _iter_cleanup(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def __call__(self, data: list[Entry], batch_size: int = 1) -> Iterator[Tensor]:
        self.load_prepare_models()

        for batch in batched(data, batch_size):
            entries = transpose_entries(batch)
            encoded = self.encode_tokens(entries.tokens)

            target_token_indices = self.get_target_subword_token_ranges(
                encoded.word_ids, entries.target_word_ids, len(encoded.encodings)
            )
            model_output = self.model_pass(encoded)
            merged_embeddings = self.merge_layers(
                [
                    self.get_layer(model_output, layer_id)
                    for layer_id in self.layers_of_interest
                ]
            )
            target_embeddings = self.extract_target_embeddings(
                merged_embeddings, target_token_indices
            )

            self._iter_cleanup()
            yield target_embeddings.detach()  # detaching from autograd graph, pevents OOM
