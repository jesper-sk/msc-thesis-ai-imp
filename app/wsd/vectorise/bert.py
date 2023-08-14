from os import PathLike
from typing import Any, Callable, Iterable, Literal, Optional, Sequence, TypeAlias

import torch
import transformers as trans
from torch import Tensor
from transformers import BertModel, BertTokenizerFast
from transformers.utils.generic import ModelOutput, PaddingStrategy, TensorType

from ..data.tokens import TokenBatch, TokenInputConvertible
from . import Vectoriser

# Type aliases
Tokens: TypeAlias = list[str]
Tokenizer: TypeAlias = BertTokenizerFast
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
        subword_merge_operation: SubwordMergeOperation = "mean",
        layer_merge_operation: LayerMergeOperation = "sum",
        device: str | None = None,
    ):
        """Create and initialise a BertVectoriser.

        Parameters
        ----------
        `model_name_or_path : str | PathLike[Any]`, optional
            The name or path to load a model from, by default "bert-base-uncased".
        `preload : bool`, optional
            Whether to preload the model and tokenizer (only relevant if `model` and
            `tokenizer` are omitted), by default `True`.
        `model : Optional[Model]`, optional
            The model to use, by default `None`. If this is provided, `tokenizer` must
            also be provided. If given, overrides `model_name_or_path`.
        `tokenizer : Optional[Tokenizer]`, optional
            The tokenizer to use, by default None. If this is provided, `model` must
            also be provided. The tokenizer must be a fast tokenizer (e.g.
            BertTokenizerFast). If given, overrides `model_name_or_path`.
        `layers_of_interest : Optional[list[int]]`, optional
            The model output hidden layer indices that will be combined for the contextual
            embeddings, by default `None`. If omitted, only the last hidden layer will be
            used.
        `subword_merge_operation : SubwordMergeOperation`, optional
            The subword merge operation to be used, by default `"first"`. The options are:
            - `"first"`: Use the first target subword token.
            - `"mean"`: Use the mean of all target subword tokens.

        `layer_merge_operation : LayerMergeOperation`, optional
            The layer merge operation to be used, by default `"sum"`. The options are:
            - `"sum"`: Sum the hidden states of the layers of interest.

        `device : str | None`, optional
            The device to run the vectoriser on, by default `None`. If `None`, the
            vectoriser will run on the CPU.

        Raises
        ------
        `ValueError`
            When `model` and `tokenizer` are not both provided or both omitted.
        `ValueError`
            When `model_name_or_path`, `model` and `tokenizer` are all omitted.
        """
        self.model: Model
        self.tokenizer: Tokenizer
        self.model_name_or_path: str | PathLike[Any]
        self.layers_of_interest: list[int] = layers_of_interest or [-1, -2, -3, -4]
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
        """Load the model and tokenizer, if they aren't already loaded. Also set the model
        to evaluation mode.

        Raises
        ------
        ValueError
            If the tokenizer is not a 'fast' variant (e.g. BertTokenizerFast).
        """
        if not self._is_loaded():
            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name_or_path)
            self.model: Model = BertModel.from_pretrained(self.model_name_or_path)  # type: ignore

        if not self.tokenizer.is_fast:
            raise ValueError(
                "Tokenizer should be a 'fast' variant (e.g. BertTokenizerFast)"
            )

        self.model.eval()
        if self.device:
            self.model.to(self.device)

    def to(self, device: str) -> None:
        """Move the vectoriser to a device.

        Parameters
        ----------
        device : str
            The device to move the vectoriser to.
        """
        self.device = device
        if self._is_loaded():
            self.model.to(device)

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

    def tensor(self, data, *, dtype=None) -> Tensor:
        """Create a tensor on the vectoriser's device.

        Parameters
        ----------
        `data : _type_`
            The data to create a tensor from.
        `dtype : _type_, optional`
            The dtype of the tensor, by default None

        Returns
        -------
        `Tensor`
            The tensor created from the given data.
        """
        return torch.tensor(data, dtype=dtype, device=self.device)

    def encode_tokens(self, tokens: Tokens | list[Tokens]) -> TokenizerOutput:
        """Encode the given tokens into subword tokens using the tokenizer.

        Parameters
        ----------
        `tokens : Tokens | list[Tokens]`
            The tokens or batch of tokens to encode.

        Returns
        -------
        `TokenizerOutput`
            The encoded tokens.
        """
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
        """Perform a forward pass of the model on the given encoded tokens.

        Parameters
        ----------
        `encoded : TokenizerOutput`
            The encoded tokens to pass through the model.

        Returns
        -------
        `ModelOutput`
            The output of the model.
        """
        return self.model(**encoded, output_hidden_states=True)

    def get_layer(self, model_output: ModelOutput, layer: int) -> Tensor:
        """Get the hidden states for a given layer from the model output.

        Parameters
        ----------
        `model_output : ModelOutput`
            The output of the model.
        `layer : int`
            The layer index to get the hidden states for.

        Returns
        -------
        `Tensor`
            The hidden states for the given layer.
        """
        return model_output["hidden_states"][layer]

    def get_target_subword_token_ranges(
        self,
        get_word_ids: Callable[[int], list[int | None]],
        target_word_ids: Sequence[int],
    ) -> list[tuple[int, int]]:
        """Get the start and end indices of the subword tokens corresponding to the target
        words for each sentence in the batch.

        Parameters
        ----------
        `get_word_ids : Callable[[int], list[int  |  None]]`
            A function that takes an entry ID and returns a list of word IDs for that
            entry. The word IDs indicate what subword token corresponds to what word in
            the entry. If a word is split into multiple subword tokens, the word ID will
            be repeated for each subword token. If a token is a special token (e.g.
            `[CLS]`, `[SEP]`, `[PAD]`), the word ID will be None.
        `target_word_ids : Sequence[int]`
            The word IDs of the target word for each sentence in the batch.

        Returns
        -------
        `list[tuple[int, int]]`
            The start and end indices of the subword tokens corresponding to the target
            words for each sentence in the batch.
        """

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
            for entry_id in range(len(target_word_ids))
        ]

    def extract_target_embeddings(
        self, merged_embeddings: Tensor, target_token_ranges: list[tuple[int, int]]
    ) -> Tensor:
        """Extract embeddings for target tokens from merged layer embeddings.

        Parameters
        ----------
        `merged_embeddings : Tensor (batch_size, num_subword_tokens, embedding_dim)`
            The merged layer embeddings for all subword tokens in the batch.
        `target_token_ranges : list[tuple[int, int]]`
            The start and end indices of the target subword tokens in the merged
            embeddings.

        Returns
        -------
        `Tensor (batch_size, embedding_dim)`
            The embeddings for the target tokens, for each sentence in the batch.
        """
        ret = torch.zeros(
            merged_embeddings.shape[0::2]
        )  # shape: (batch_size, embedding_dim)

        for sentence_index, (encoding, (start_index, end_index)) in enumerate(
            zip(merged_embeddings, target_token_ranges)
        ):
            subword_tokens = encoding[start_index:end_index, :]
            ret[sentence_index, :] = self.merge_subword_tokens(subword_tokens)

        return ret

    def _iter_cleanup(self):
        """Cleanup GPU memory after each iteration, if applicable"""
        if self.device == "cuda":
            torch.cuda.empty_cache()

    # def __call__(self, batch: Iterable[Entry]) -> Tensor:
    #     """Call the vectoriser on the given batch of sentences. This will vectorise the
    #     sentences and return the output of the model for each sentence. Note that you must
    #     call `load_prepare_models()` before calling this method, if `preload` was set to
    #     False in the constructor.

    #     Parameters
    #     ----------
    #     `batch : Iterable[Entry]`
    #         The batch of sentences to vectorise.

    #     Returns
    #     -------
    #     `Iterable[Tensor]`
    #         The output of the model for each entry in the batch.
    #     """
    #     entries = VerticalEntries.make(batch)
    #     encoded = self.encode_tokens(list(entries.tokens))

    #     target_token_indices = self.get_target_subword_token_ranges(
    #         encoded.word_ids, entries.target_word_ids
    #     )
    #     model_output = self.model_pass(encoded)
    #     merged_embeddings = self.merge_layers(
    #         [
    #             self.get_layer(model_output, layer_id)
    #             for layer_id in self.layers_of_interest
    #         ]
    #     )
    #     target_embeddings = self.extract_target_embeddings(
    #         merged_embeddings, target_token_indices
    #     )

    #     self._iter_cleanup()
    #     return target_embeddings.detach()  # detaching from autograd graph, pevents OOM

    def __call__(self, batch: TokenBatch):
        """Call the vectoriser on the given batch of sentences. This will vectorise the
        sentences and return the output of the model for each sentence. Note that you must
        call `load_prepare_models()` before calling this method, if `preload` was set to
        False in the constructor.

        Parameters
        ----------
        `batch : Iterable[Entry]`
            The batch of sentences to vectorise.

        Returns
        -------
        `Iterable[Tensor]`
            The output of the model for each entry in the batch.
        """
        encoded = self.encode_tokens(list(batch.tokens))

        target_token_indices = self.get_target_subword_token_ranges(
            encoded.word_ids, batch.target_token_id
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
        return target_embeddings.detach()  # detaching from autograd graph, pevents OOM
