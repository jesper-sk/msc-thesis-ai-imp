from typing import Any, Callable, Iterable, Literal, Optional, Sequence, TypeAlias

import torch
import transformers as trans
from torch import Tensor
from transformers import BertTokenizerFast, Pipeline
from transformers.utils.generic import ModelOutput, PaddingStrategy, TensorType

# Type aliases
Tokens: TypeAlias = list[str]
Tokenizer: TypeAlias = BertTokenizerFast
TokenizerOutput: TypeAlias = trans.tokenization_utils.BatchEncoding
Model: TypeAlias = trans.BertModel
SubwordMergeOperation: TypeAlias = Literal["first", "mean"]
LayerMergeOperation: TypeAlias = Literal["sum"]


class WsdPipeline(Pipeline):
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

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {key: value for key, value in kwargs.items() if key in []}
        forward_kwargs = {key: value for key, value in kwargs.items() if key in []}
        postprocess_kwargs = {
            key: value for key, value in kwargs.items() if key in ["layers_of_interest"]
        }

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

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

    def preprocess(self, input):
        if not self.tokenizer:
            raise RuntimeError("Expected a tokenizer to be given, found None")

        tokenized = self.tokenizer(
            input["tokens"],
            is_split_into_words=True,
            padding=PaddingStrategy.LONGEST,
            return_tensors=TensorType.PYTORCH,
            return_attention_mask=True,  # Return attention masks to distinguish padding from input
        )

        target_token_indices = self.get_target_subword_token_ranges(
            tokenized.word_ids, input["target_token_id"]
        )

        return {"tokenized": tokenized, "target_token_indices": target_token_indices}

    def _forward(self, model_inputs):
        output = self.model(**model_inputs, output_hidden_states=True)
        return {
            "model_output": output,
            "target_token_indices": model_inputs["target_token_indices"],
        }

    def _tensor(self, data, /, *, dtype=None) -> Tensor:
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

    def _merge_layers_sum(self, layers: Iterable[Tensor]) -> Tensor:
        return sum(layers, self._tensor([0]))

    def _get_layer(self, model_output: ModelOutput, layer: int) -> Tensor:
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

    def extract_target_embeddings(
        self,
        merged_embeddings: Tensor,
        target_token_ranges: list[tuple[int, int]],
        merge_subword_tokens: Callable[[Tensor], Tensor],
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
            ret[sentence_index, :] = merge_subword_tokens(subword_tokens)

        return ret

    def _iter_cleanup(self):
        """Cleanup GPU memory after each iteration, if applicable"""
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def postprocess(
        self,
        model_outputs,
        layers_of_interest: list[int] = [-1, -2, -3, -4],
        merge_subword_tokens: Optional[Callable[[Tensor], Tensor]] = None,
    ):
        output = model_outputs["model_output"]
        merged_embeddings = self._merge_layers_sum(
            [self._get_layer(output, layer_id) for layer_id in layers_of_interest]
        )
        target_embeddings = self.extract_target_embeddings(
            merged_embeddings,
            model_outputs["target_token_indices"],
            merge_subword_tokens or self._merge_subword_first,
        )
        self._iter_cleanup()

        return {"target_embeddings": target_embeddings.detach()}
