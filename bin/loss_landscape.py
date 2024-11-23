#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WIP: Compute loss landscape for a given model and task.
"""
import itertools
import json
import os
from typing import Any

import loss_landscapes
import loss_landscapes.metrics
import torch
import numpy as np
import transformers as tf

from src.core.app import harness
from src.core.context import Context
from src.core.path import dirparent
from src.data import tasks


# ---------------------------------------------------------------------------
# Patching support for GPUs
# ---------------------------------------------------------------------------
from loss_landscapes.model_interface.model_parameters import ModelParameters

# NOTE: I did this.
def patched_filter_normalize_(self, ref_point: 'ModelParameters', order=2):
    """
    In-place filter-wise normalization of the tensor.
    :param ref_point: use this model's filter norms, if given
    :param order: norm order, e.g. 2 for L2 norm
    :return: none
    """
    self.parameters = [p.to("cuda:0") for p in self.parameters]
    for l in range(len(self.parameters)):
        # normalize one-dimensional bias vectors
        if len(self.parameters[l].size()) == 1:
            self.parameters[l] *= (ref_point.parameters[l].norm(order) / self.parameters[l].norm(order))
        # normalize two-dimensional weight vectors
        for f in range(len(self.parameters[l])):
            try:
                self.parameters[l][f] *= ref_point.filter_norm((l, f), order) / (self.filter_norm((l, f), order))
            except ZeroDivisionError:
                continue
loss_landscapes.model_interface.model_parameters.ModelParameters.filter_normalize_ = patched_filter_normalize_

# SRC: https://github.com/marcellodebernardi/loss-landscapes/issues/1
def rand_u_like(example_vector: 'ModelParameters') -> 'ModelParameters':
    """
    Create a new ModelParameters object of size and shape compatible with the given
    example vector, such that the values in the ModelParameter are uniformly distributed
    in the range [0,1].
    :param example_vector: defines by example the size and shape the new vector will have
    :return: new vector with uniformly distributed values
    """
    new_vector = []

    for param in example_vector:
        new_vector.append(torch.rand(size=param.size(), dtype=example_vector[0].dtype).to(param.device))

    return ModelParameters(new_vector)
loss_landscapes.model_interface.model_parameters.rand_u_like = rand_u_like

# SRC: https://github.com/marcellodebernardi/loss-landscapes/issues/1
def rand_n_like(example_vector: 'ModelParameters') -> 'ModelParameters':
    """
    Create a new ModelParameters object of size and shape compatible with the given
    example vector, such that the values in the ModelParameter are normally distributed
    as N(0,1).
    :param example_vector: defines by example the size and shape the new vector will have
    :return: new vector with normally distributed values
    """
    new_vector = []

    for param in example_vector:
        new_vector.append(torch.randn(size=param.size(), dtype=example_vector[0].dtype).to(param.device))

    return ModelParameters(new_vector)
loss_landscapes.model_interface.model_parameters.rand_n_like = rand_n_like


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wrap_transformers_model(model: tf.AutoModel) -> torch.nn.Module:
    class WrappedTransformer(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model
        
        def forward(self, x) -> torch.FloatTensor:
            return self.model(x).logits
    return WrappedTransformer(model)


def compute_loss_landscape(
    model_path: str,
    task: str,
    task_config: dict[str, Any],
    steps: int = 150,  # XXX
    distance: int = 50
):
    # Load the model and tokenizer.
    model = tf.AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = tf.AutoTokenizer.from_pretrained(model_path)
    # Load the dataset.
    data = tasks.load_kfold(
        task_config["dataset"],
        **task_config["dataset_kwargs"],
        fold=0,
        k=5,
        seed=19
    ).rename_columns({
        task_config["label_column"]: "label"
    })
    label_list = sorted(set(itertools.chain(*[data[split]["label"] for split in data])))
    def tokenize(examples):
        # Label processing.
        if not task_config.get("do_regression", False):
            # Expects labels to be strings and uses the label_list to map them to ints.
            examples["label"] = list(
                map(lambda l: label_list.index(l), examples["label"])
            )
        # Text processing.
        return tokenizer(
            examples[task_config["text_column"]],
            padding="max_length",
            max_length=512,
            truncation=True
        )
    data = data["train"].select(range(16)).map(tokenize, batched=True)
    x, y = torch.tensor(data["input_ids"]), torch.tensor(data["label"])
    # Compute landscape.
    device = "cuda:0"
    metric = loss_landscapes.metrics.Loss(
        torch.nn.CrossEntropyLoss(), x.to(device), y.to(device)
    )
    return loss_landscapes.random_plane(
        model=wrap_transformers_model(model.to(device)).to(device),
        metric=metric,
        distance=distance,
        steps=steps,
        normalization="filter",
        deepcopy_model=True
    )


def main(ctx: Context) -> None:
    default_outputs_dir = os.path.join(
        dirparent(os.path.realpath(__file__), 2),
        "outputs", "loss_landscape"
    )
    default_task_config = os.path.join(
        dirparent(os.path.realpath(__file__), 2),
        "configs", "classification", "tasks.json"
    )
    ctx.parser.add_argument("-c", "--task-config", default=default_task_config)
    ctx.parser.add_argument("-o", "--outputs_dir", default=default_outputs_dir)
    ctx.parser.add_argument("-t", "--tasks", nargs="*")
    ctx.parser.add_argument("-m", "--model_path")
    args = ctx.parser.parse_args()

    with open(args.task_config, "r") as fd:
        task_config = json.load(fd)
    for task in args.tasks or list(task_config):
        if task not in task_config:
            parser.error(f"unknown task: {task} {list(task_config)}")
        loss_data = compute_loss_landscape(args.model_path, task, task_config[task])
        # Write data.
        model_name = args.model_path
        if "outputs/seq2seq" in model_name:
            model_name = model_name.split("seq2seq")[-1].split(os.sep)[1]
        elif "outputs/classification" in model_name:
            assert task in model_name
            model_name = model_name.split("classification")[-1].split(os.sep)[1]
        elif len(model_name.split(os.sep)) <= 2:
            model_name = model_name.split(os.sep)[-1]
        else:
            raise ValueError("unexpected model: {args.model_path}")
        path = os.path.join(args.outputs_dir, task, f"{model_name}.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ctx.log.info("writing: %s", path)
        np.save(path, loss_data)


if __name__ == "__main__":
    harness(main)
