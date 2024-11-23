#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarizes eval_results.json files in the outputs/ dir.

Usage Examples:
    $ oinfo.py  # No arguments needed.
"""
import glob
import json
import os
from typing import Any

import pandas as pd

from src.core.app import harness
from src.core.context import Context
from src.core.path import dirparent


def print_summary(summary: pd.DataFrame) -> None:
    col_order = [
        "task", "config",
        "fold", "seed", "metric", "value"
        #  "fold", "seed",
        #  "all_strict",
        #  "all_naive",
        #  "tom:belief:accessible:multiple-choice",
        #  "tom:belief:inaccessible:multiple-choice",
        #  "tom:answerability:binary",
        #  "tom:info_accessibility:binary",
    ]
    prev_task = None
    lines = summary[col_order].to_string(index=False).split("\n")
    max_length = max(map(len, lines))
    for line in lines:
        curr_task = line.split()[0]
        if prev_task != curr_task:
            char = "=" if prev_task == "task" else "-"
            print(char * max_length)
        prev_task = curr_task
        print(line)
    print("=" * max_length)


def main(ctx: Context) -> None:
    default_outputs_dir = os.path.join(
        dirparent(os.path.realpath(__file__), 2),
        "outputs", "classification"
    )
    ctx.parser.add_argument("-d", "--outputs_dir", default=default_outputs_dir)
    args = ctx.parser.parse_args()
    
    data = []
    gstr = os.path.join(args.outputs_dir, "**", "eval_preds.csv")
    for path in glob.glob(gstr, recursive=True):
        if "fantom_bin" not in path: continue
        config, task, fold_seed = path.split(os.sep)[-4:-1]
        df_bin = pd.read_csv(path).drop(columns="Unnamed: 0")
        df_mc = pd.read_csv(path.replace("_bin", "_mc")).drop(columns="Unnamed: 0")
        df = pd.concat([df_bin, df_mc])
        df = df.assign(pred_correct=df.label == df.pred)
        # Compute accuracy scores.
        acc = df.groupby("question_type").mean(numeric_only=True).pred_correct
        acc_naive = df.mean(numeric_only=True).pred_correct
        acc_strict = df.groupby("set_id").all().mean(numeric_only=True).pred_correct
        acc["all_naive"] = acc_naive
        acc["all_strict"] = acc_strict
        #  data.append({
        #      "config": config,
        #      "task": "fantom",
        #      "fold": int(fold_seed.split("_")[1]),
        #      "seed": int(fold_seed.split("_")[3]),
        #  } | acc.to_dict())
        for task, value in acc.to_dict().items():
            data.append({
                "config": config,
                "task": task,
                "fold": int(fold_seed.split("_")[1]),
                "seed": int(fold_seed.split("_")[3]),
                "metric": "accuracy",
                "value": value
            })

    gcols = ["task", "config"]
    print_summary(pd.DataFrame(data).set_index(gcols).sort_index().reset_index())
    df = pd.DataFrame(data).groupby(gcols).mean(
        numeric_only=True
    ).reset_index().assign(metric="accuracy")
    summary = []
    for task in sorted(df.task.unique()):
        ascending = False
        summary.append(
            df[df.task == task].sort_values(
                "value", ascending=ascending
            ).reset_index(drop=True)
        )
    print_summary(pd.concat(summary))


if __name__ == "__main__":
    harness(main)
