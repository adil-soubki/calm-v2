#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run agents zero-shot on a given task.

Usage Examples:
    $ error_analysis.py                   # No args needed.
    $ error_analysis.py -o path/to/outdir # Custom outdir.
    $ error_analysis.py --model flan-t5-base
"""
import functools
import glob
import json
import os
from typing import Any

import datasets as ds
import pandas as pd

from src.agents.huggingface import HFAgent
from src.core.app import harness
from src.core.context import Context, get_context
from src.core.path import dirparent
from src.data import tasks


def add_zeroshot_pred(df: pd.DataFrame) -> pd.DataFrame:
    assert len(df.task.unique()) == 1
    task = df.task.unique()[0]
    cdx = list(df.columns).index("flan-t5-base_pred")
    generations = (
        df["flan-t5-base"].str.strip()
            .str.lower()
            .str.replace("(", "")
            .str.replace(")", "")
    )
    if task == "fantom_bin":
        zs_preds = ["yes" if g.startswith("yes") else "no" for g in generations]
    elif task == "fantom_mc":
        zs_preds = ["a" if g.startswith("a") else "b" for g in generations]
    df.insert(loc=cdx, column="flan-t5-base_zs-pred", value=zs_preds)
    return df


def do_accuracies(df: pd.DataFrame) -> dict[str, float]:
    assert len(df.task.unique()) == 1
    task = df.task.unique()[0]
    ret = {}
    for col in df.columns:
        if not col.endswith("pred"): continue
        ret[col] = (df[col] == df["correct_answer"]).mean()
    return ret


def do_consolidate(outdir: str, task_config: dict[str, dict[str, Any]]) -> None:
    log = get_context().log
    summary = {}
    for task in task_config:
        ret = []
        for path in glob.glob(os.path.join(outdir, "*", f"{task}.csv")):
            model_name = os.path.basename(os.path.dirname(path))
            cols = ["input_text", "generation", "pred", "correct_answer", "task"]
            df = pd.read_csv(path)[cols].rename(columns={
                "generation": model_name,
                "pred": f"{model_name}_pred"
            })
            ret.append(df)
        if not ret:
            continue  # Nothing to consolidate.
        outpath = os.path.join(outdir, f"{task}.csv")
        log.info("writing: %s", outpath)
        mcols = ["input_text", "correct_answer", "task"]
        rslt = functools.reduce(
            lambda l, r: l.merge(r, on=mcols, validate="1:1"), ret
        )
        cols = (
            ["input_text", "correct_answer"] +
            [c for c in rslt.columns if c not in mcols] +
            ["task"]
        )
        rslt = rslt[cols]
        for col in rslt.columns:
            if not col.endswith("_pred") and col != "correct_answer": continue
            if task == "fantom_bin":
                rslt[col] = rslt[col].replace(0, "no").replace(1, "yes").replace("no:long", "no")
            elif task == "fantom_mc":
                rslt[col] = rslt[col].replace(0, "a").replace(1, "b")
        rslt = add_zeroshot_pred(rslt)
        rslt.to_csv(outpath, index=False)
        summary[task] = do_accuracies(rslt)
    summary = pd.DataFrame(summary)
    log.info("accuracies:")
    list(map(log.info, summary.to_string().split("\n")))


def main(ctx: Context) -> None:
    model_map = {
        "flan-t5-base": "google/flan-t5-base",
        "calm-flan-t5-base": "../calm/outputs/seq2seq/default/fold_0/checkpoint-240581",
        "calm-both-flan-t5-base": "outputs/seq2seq/wsj-llama3-calm-both/fold_0/checkpoint-240586",
        "calm-author-flan-t5-base": "outputs/seq2seq/wsj-llama3-calm-author/fold_0/checkpoint-240581",
    }
    default_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "outputs", "error_analysis"
    )
    default_task_config = os.path.join(
        dirparent(os.path.realpath(__file__), 2),
        "configs", "classification", "tasks.json"
    )
    ctx.parser.add_argument("-o", "--outdir", default=default_outdir)
    ctx.parser.add_argument("-c", "--task-config", default=default_task_config)
    ctx.parser.add_argument("-t", "--tasks", nargs="*")
    ctx.parser.add_argument("-m", "--models", nargs="*", choices=list(model_map))
    ctx.parser.add_argument("--consolidate", action="store_true")
    args = ctx.parser.parse_args()

    models = args.models or list(model_map)
    cls_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "outputs", "classification"
    )
    with open(args.task_config, "r") as fd:
        task_config = json.load(fd)
    if args.consolidate:
        return do_consolidate(args.outdir, task_config)
    for model in models:
        agent = HFAgent(
            model_map[model],
            min_new_tokens=256,
            max_new_tokens=256,
            num_beams=4,
            #  length_penalty=10.0,
            #  renormalize_logits=True
        )
        for task in args.tasks or list(task_config):
            if task not in task_config:
                parser.error(f"unknown task: {task} {list(task_config)}")
            text_column = task_config[task]["text_column"]
            label_column = task_config[task]["label_column"]
            gstr = os.path.join(cls_outdir, model, task, "*", "eval_preds.csv")
            path = glob.glob(gstr)[0]
            df = pd.read_csv(path)[:50]  # NOTE: Only sampling the first 50 rows.
            generations = [g["generation"] for g in agent.generate(df[text_column])]
            df = df.assign(generation=generations, task=task, model=model)
            outpath = os.path.join(args.outdir, model, f"{task}.csv")
            ctx.log.info("writing: %s", outpath)
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            df.to_csv(outpath, index=False)


if __name__ == "__main__":
    harness(main)
