#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""An example script"""
import dataclasses
import hashlib
import itertools
import os
import sys
from copy import copy
from typing import Optional

import datasets
import evaluate
import numpy as np
import pandas as pd
import transformers as tf

from src.core.context import Context, get_context
from src.core.app import harness
from src.core.path import dirparent
from src.data import wikiface


@dataclasses.dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = dataclasses.field(default=None)


@dataclasses.dataclass
class DataArguments:
    data_num_folds: int
    data_fold: int = dataclasses.field(default=None)
    do_regression: bool = dataclasses.field(default=False)
    metric_for_classification: str = dataclasses.field(default="f1")
    metric_for_regression: str = dataclasses.field(default="mae")
    text_max_length: int = dataclasses.field(
        default=256,
        metadata={
            "help": (
                "The maximum total text input sequence length after tokenization. "
		"Sequences longer than this will be truncated, sequences shorter "
		"will be padded."
            )
        },
    )


def update_metrics(
    preds: list,
    refs: list,
    metric: str,
    trainer: tf.Trainer,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: tf.TrainingArguments
):
    logger = get_context().log
    # Get run id and output dir.
    args = vars(model_args) | vars(data_args) | training_args.to_dict()
    for key in ("data_fold", "output_dir", "logging_dir"):
        del args[key]  # Do not use in run_id.
    run_id = hashlib.md5(str(sorted(args.items())).encode("utf-8")).hexdigest()
    run_id += f"-{data_args.data_fold}"
    args["data_fold"] = data_args.data_fold
    logger.info("\nRUN_ID: %s", run_id)
    output_dir = os.path.join(dirparent(training_args.output_dir, 2), "runs")
    os.makedirs(output_dir, exist_ok=True)
    # Compute the new results.
    results = evaluate.combine([metric]).compute(
        predictions=preds, references=refs, average="micro"
    )  # XXX: handle f1 better. include pearsonr.
    df = pd.DataFrame([args | results])
    df["last_modified"] = pd.Timestamp.now()
    df["current_epoch"] = trainer.state.epoch
    df.to_csv(os.path.join(output_dir, f"{run_id}.csv"), index=False)
    # Write out predictions.
    pd.DataFrame({"refs": refs, "preds": preds, "run_id": run_id}).to_csv(
        os.path.join(output_dir, f"{run_id}.preds.csv"), index=False
    )


def run(
    ctx: Context,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: tf.TrainingArguments
) -> None:
    # Make a directory per fold.
    training_args.output_dir = os.path.join(
        training_args.output_dir, f"fold_{data_args.data_fold}"
    )
    ctx.log.info(f"Training parameters {training_args}")
    ctx.log.info(f"Data parameters {data_args}")
    ctx.log.info(f"Model parameters {model_args}")
    # Set seed before initializing model.
    tf.set_seed(training_args.seed)
    # Configure for regression if needed.
    metric = (
        data_args.metric_for_regression
        if data_args.do_regression
        else data_args.metric_for_classification
    )
    # XXX: Currently not needed.
    training_args.greater_is_better = metric not in ("loss", "eval_loss", "mse", "mae")
    # Load training data.
    data = wikiface.load_kfold(
        fold=data_args.data_fold,
        k=data_args.data_num_folds,
        seed=training_args.data_seed
    )
    if data_args.do_regression:
        data = data.remove_columns("label").rename_column("label_float", "label")
        model_args.num_labels = 1  # NOTE: Just used to stratify.
    # Preprocess training data.
    label_list = sorted(set(itertools.chain(*[data[split]["label"] for split in data])))
    tokenizer = tf.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    assert tokenizer.model_max_length >= data_args.text_max_length
    def preprocess_fn(examples):
        # Label processing.
        examples["label"] = list(map(lambda l: label_list.index(l), examples["label"]))
        # Text processing.
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=data_args.text_max_length,
            truncation=True
        )
    data = data.map(preprocess_fn, batched=True, batch_size=16)
    train_dataset, eval_dataset = data["train"], data["test"]
    # Model training.
    config = tf.AutoConfig.from_pretrained(
       	model_args.model_name_or_path,
        finetuning_task="text-classification",
        label2id={v: i for i, v in enumerate(label_list)},  # XXX: Is this needed?
        id2label={i: v for i, v in enumerate(label_list)},  # XXX: Is this needed?
    )
    model = tf.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )
    trainer = tf.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    def compute_metrics(eval_pred: tf.EvalPrediction):
        # Get logits.
        if isinstance(eval_pred.predictions, tuple):
            logits = eval_pred.predictions[0]
        else:
            logits = eval_pred.predictions
        # Get predictions.
        if data_args.do_regression:
            predictions = np.squeeze(logits)
        else:
            predictions = np.argmax(logits, axis=1)
        # Save predictions to file.
        pdf = eval_dataset.to_pandas().assign(pred=predictions)
        assert np.allclose(pdf.label, eval_pred.label_ids)
        pdf.to_csv(os.path.join(training_args.output_dir, "preds.csv"))
        # Update aggregated evaluation results.
        update_metrics(
            predictions, eval_pred.label_ids, metric, trainer,
            model_args, data_args, training_args
        )
        # Return metrics.
        return evaluate.combine([metric]).compute(
            predictions=predictions,
            references=eval_pred.label_ids,
            average="micro",
        )  # XXX: handle f1 better. include pearsonr.
    trainer.compute_metrics = compute_metrics
    trainer.train()
    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def main(ctx: Context) -> None:
    # Parse arguments.
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, tf.TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        parser.error("No configuration passed")
    # Run the training loop.
    if data_args.data_fold is not None:
        return run(ctx, model_args, data_args, training_args)
    for fold in range(data_args.data_num_folds):
        data_args.data_fold = fold
        run(ctx, copy(model_args), copy(data_args), copy(training_args))


if __name__ == "__main__":
    harness(main)
