#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run agents zero-shot on a given task.

Usage Examples:
    $ prompting.py                   # No args needed.
    $ prompting.py -o path/to/outdir # Custom outdir.
    $ prompting.py --model meta-llama/Meta-Llama-3-8B-Instruct
"""
import json
import os
from typing import Any

import datasets as ds
import pandas as pd

from src.agents.openai import GPTAgent
from src.agents.pricing import estimate_cost
from src.core.app import harness
from src.core.context import Context
from src.core.path import dirparent
from src.data import prompts, tasks


def load_task(task: str, task_config: dict[str, Any]) -> ds.DatasetDict:
    ddict = tasks.load_kfold(
        task_config[task]["dataset"],
        **task_config[task]["dataset_kwargs"],
        fold=0,
        k=5,
        seed=19
    ).rename_columns({
        task_config[task]["text_column"]: "text_column",
        task_config[task]["label_column"]: "label_column"
    })
    return pd.concat(
        [dt.to_pandas().assign(split=sp) for sp, dt in ddict.items()]
    )


def main(ctx: Context) -> None:
    default_outdir = os.path.join(
        dirparent(os.path.realpath(__file__), 2), "outputs", "zero_shot"
    )
    default_task_config = os.path.join(
        dirparent(os.path.realpath(__file__), 2),
        "configs", "classification", "tasks.json"
    )
    ctx.parser.add_argument("-o", "--outdir", default=default_outdir)
    ctx.parser.add_argument("-c", "--task-config", default=default_task_config)
    ctx.parser.add_argument("-t", "--tasks", nargs="*")
    ctx.parser.add_argument("-m", "--model", default="gpt-4o")
    args = ctx.parser.parse_args()

    with open(args.task_config, "r") as fd:
        task_config = json.load(fd)
    for task in args.tasks or list(task_config):
        if task not in task_config:
            parser.error(f"unknown task: {task} {list(task_config)}")
        df = load_task(task, task_config)
        prompt_template = prompts.load("zero_shot", task)
        df = df.assign(
            prompt=list(map(lambda t: prompt_template.format(t), df.text_column))
        )
        ctx.log.info(
            f"Example prompt:\n"
            f"[Input] {df.iloc[0].prompt}\n"
            f"{' ':->7}\n"
            f"[Label] {df.iloc[0].label_column}"
        )
        # Estimate cost.
        cost = estimate_cost(args.model, df.prompt.tolist())
        if cost > 20.0:  # Ask for confirmation on anything more than $20.
            answer = input(f"{' ':->37}Estimated cost is ${cost:.2f}. Continue? [y/N] ")
            if answer.lower() not in ["y", "yes"]:
                ctx.log.info(f"Skipping {task} with estimated cost of ${cost:.2f}")
                continue
        # Make requests.
        gpt = GPTAgent(args.model)
        generations = pd.DataFrame(gpt.generate(df.prompt.tolist()))
        if len(generations) != len(df):
            ctx.log.error(f"Some prompts did not receive an output from {args.model}")
            result = generations
        else:
            try:
                result = df.merge(generations, on="prompt", validate="1:1")
            except Exception as ex:
                ctx.log.error(f"Error encountered merging output: %s", ex)
                result = generations
        outpath = os.path.join(args.outdir, task, f"{args.model}.csv")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        ctx.log.info("writing: %s", outpath)
        result.to_csv(outpath, index=False)


if __name__ == "__main__":
    harness(main)
