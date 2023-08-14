import torch as t
import torch.nn as nn
from transformers import AutoModel


class WsdClassification(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        model_name_or_path="bert-base-uncased",
        layers_of_interest: list[int] = [-1, -2, -3, -4],
    ):
        self.num_labels = vocabulary_size
        self.layers_of_interest

        self.model = AutoModel.from_pretrained(
            model_name_or_path, output_attentions=True, output_hidden_states=True
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(768),
            nn.Linear(768, 768),
            nn.SiLU(),  # swish
            nn.Linear(768, vocabulary_size),
            nn.Softmax(),
        )

    def forward(
        self, tokenized, target_token, layers_of_interest: list[int] = [-1, -2, -3, -4]
    ):
        model_out = self.model(**tokenized)

        embeddings = t.sum(
            *[model_out["hidden_states"][layer] for layer in layers_of_interest]
        )
        output = self.classifier(embeddings)
        return output
