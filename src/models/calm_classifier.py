# -*- coding: utf-8 -*-
import dataclasses
from typing import Literal, Optional

import torch
import transformers as tf

from ..models.gpt import TransformerDecoder


@dataclasses.dataclass
class ModelArguments:
    text_model_name_or_path: Optional[str] = dataclasses.field(default=None)
    calm_model_name_or_path: Optional[str] = dataclasses.field(default=None)
    num_labels: int = dataclasses.field(default=None)
    freeze_text_model: bool = dataclasses.field(default=False)
    freeze_calm_model: bool = dataclasses.field(default=False)


def freeze_params(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def classification_head(
    input_size: int, proj_size: int, output_size: int
) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, proj_size),  # Dense projection layer.
        #  torch.nn.LayerNorm(proj_size),           # XXX: Make optional?
        torch.nn.ReLU(),                         # Activation. TODO: Dropout?
        torch.nn.Linear(proj_size, output_size)  # Classifier.
    )


# XXX: WIP
class CalmClassifier(torch.nn.Module):
    def __init__(self, config: ModelArguments):
        super().__init__()
        self.config = config
        # Load the text model.
        self.text_model = (
            tf.AutoModel.from_pretrained(config.text_model_name_or_path)
            if config.text_model_name_or_path
            else None
        )
        if self.text_model and self.config.freeze_text_model:
            freeze_params(self.text_model)
        # Load the calm model.
        self.calm_model = (
            tf.AutoModel.from_pretrained(config.calm_model_name_or_path)
            if config.calm_model_name_or_path
            else None
        )
        if self.calm_model and self.config.freeze_calm_model:
            freeze_params(self.calm_model)
        # Throw if neither is given.
        if (not self.text_model and not self.calm_model):
            raise ValueError("No text or calm model(s) specified.")
        # XXX: Currently only both provided is supported.
        if (not self.text_model or not self.calm_model):
            raise ValueError("Need both text model and calm model specified.")
        # Find max sequence length (block size)
        block_size = (
            self.text_model.config.n_positions + self.calm_model.config.n_positions
        )
        # Initialize latent layers.
        text_hidden_size = self.text_model.config.hidden_size if self.text_model else 0
        calm_hidden_size = self.calm_model.config.hidden_size if self.calm_model else 0
        latent_size = round((text_hidden_size + calm_hidden_size) / 2)
        self.text_latent_proj = torch.nn.Linear(text_hidden_size, latent_size)
        self.calm_latent_proj = torch.nn.Linear(calm_hidden_size, latent_size)
        # TODO: Make a cross-atteneding Block?
        #  self.latent_layers = TransformerDecoder(block_size=2 * latent_size)
        self.latent_layers = TransformerDecoder(block_size=block_size)
        # Initialize classification head.
        self.classification_head = classification_head(
            latent_size, latent_size, config.num_labels
            #  block_size * latent_size, latent_size, config.num_labels
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        **kwargs
    ):
        # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/models/t5/modeling_t5.py#L2053
        device = self.classification_head[0].weight.device
        text_features = torch.tensor([]).to(device)
        if self.text_model and self.text_model.__class__.__name__ == "T5Model":
            decoder_input_ids = self.text_model._shift_right(input_ids)
            text_features = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids
            )[0]  # NOTE: They don't use last_hidden_state for some reason.
        elif self.text_model:
            raise ValueError(f"Unsupported model: {self.text_model.__class__.__name__}")
        calm_features = torch.tensor([]).to(device)
        if self.calm_model and self.calm_model.__class__.__name__ == "T5Model":
            decoder_input_ids = self.calm_model._shift_right(input_ids)
            calm_features = self.calm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids
            )[0]  # NOTE: They don't use last_hidden_state for some reason.
        elif self.calm_model:
            raise ValueError(f"Unsupported model: {self.calm_model.__class__.__name__}")
        # Project latents.
        print("text_features.shape:", text_features.shape)
        print("calm_features.shape:", calm_features.shape)
        text_latents = self.text_latent_proj(text_features)
        calm_latents = self.calm_latent_proj(calm_features)
        print("text_latents.shape:", text_latents.shape)
        print("calm_latents.shape:", calm_latents.shape)
        # Latent decoder layers.
        latents = self.latent_layers(torch.cat([text_latents, calm_latents], dim=1))
        #  latents = torch.flatten(latents, 1)
        print("latents.shape:", latents.shape)
        #  print("flatten(latents, 1).shape:", torch.flatten(latents, 1).shape)
        # Classification logits.
        batch_size, _, hidden_size = latents.shape  # XXX
        latents_repr = latents.view(batch_size, -1, hidden_size)[:, -1, :]  # XXX
        print("latents_repr.shape:", latents_repr.shape)
        logits = self.classification_head(latents_repr)
        print("logits.shape:", logits.shape)
        # Compute loss.
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # TODO: Consider weighting? label_smoothing?
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # Return.
        return tf.modeling_outputs.SequenceClassifierOutput(loss=loss, logits=logits)
