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
        "fold", "seed", "epoch", "metric", "value"
    ]
    if "p_value" in summary.columns:
        col_order.append("p_value")
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


def significance_testing(df: pd.DataFrame) -> pd.DataFrame:
    import scipy.stats

    ret = []
    tasks = df.task.unique()
    baseline = df[df.config == "flan-t5-base"]
    treatments = df[df.config != "flan-t5-base"]
    for task in tasks:
        configs = treatments[treatments.task == task].config.unique()
        for config in configs:
            treatment = treatments[treatments.config == config]
            #  u_statistic, p_value = scipy.stats.ttest_ind(
            u_statistic, p_value = scipy.stats.mannwhitneyu(
                baseline[baseline.task == task].value.to_numpy(),
                treatment[treatment.task == task].value.to_numpy()
            )
            ret.append({
                "config": config,
                "task": task,
                "u_statistic": u_statistic,
                "p_value": float(p_value)
            })
    return pd.DataFrame(ret)


def main(ctx: Context) -> None:
    default_outputs_dir = os.path.join(
        dirparent(os.path.realpath(__file__), 2),
        "outputs", "classification"
    )
    ctx.parser.add_argument("-d", "--outputs_dir", default=default_outputs_dir)
    args = ctx.parser.parse_args()
    
    data = []
    gstr = os.path.join(args.outputs_dir, "**", "eval_results.json")
    metrics = [
        "eval_f1_micro", # "eval_f1_macro",
        "eval_accuracy",
        "eval_mae"
    ]
    task_to_metric = {}
    for path in glob.glob(gstr, recursive=True):
        if "seed_" not in path: continue
        if "fold_" not in path: continue
        with open(path, "r") as fd:
            dct = json.load(fd)
        config, task, fold_seed = path.split(os.sep)[-4:-1]
        assert sum([m in dct for m in metrics]) == 1
        for metric in metrics:
            if metric not in dct: continue
            task_to_metric[task] = metric
            data.append({
                "config": config,
                "task": task,
                "fold": int(fold_seed.split("_")[1]),
                "seed": int(fold_seed.split("_")[3]),
                "epoch": dct["epoch"],
                "metric": metric,
                "value": dct[metric]
            })

    def set_metric(row: pd.Series) -> pd.Series:
        row["metric"] = task_to_metric[row["task"]]
        return row

    gcols = ["task", "config"]
    print_summary(pd.DataFrame(data).set_index(gcols).sort_index().reset_index())
    df = pd.DataFrame(data).groupby(gcols).mean(
        numeric_only=True
    ).reset_index().apply(set_metric, axis=1)
    df = df.merge(
        significance_testing(pd.DataFrame(data)), on=["task", "config"], how="left"
    ).fillna(1.0)  # Add in significant testing results.
    summary = []
    for task in sorted(df.task.unique()):
        ascending = False if task_to_metric[task] != "eval_mae" else True
        summary.append(
            df[df.task == task].sort_values(
                "value", ascending=ascending
            ).reset_index(drop=True)
        )
    print_summary(pd.concat(summary))


if __name__ == "__main__":
    harness(main)
