# -*- coding: utf-8 -*-
import glob
import os
from typing import Any

import datasets as ds
import pandas as pd
from more_itertools import one

from ..core.path import dirparent


CTOM_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "common_tom")


def load_raw() -> ds.Dataset:
    ret = []
    for path in glob.glob(os.path.join(CTOM_DIR, "*.csv.gz")):
        ret.append(pd.read_csv(path))
    ret = pd.concat(ret).reset_index(drop=True)
    return ds.Dataset.from_pandas(ret, preserve_index=False)


def load(context_size: int = 5) -> ds.DatasetDict:
    raw = load_raw()
    def add_input_text(row: dict[str, Any]) -> dict[str, Any]:
        turns = row["context"].split("\n")
        stop_idx = one([idx for idx, turn in enumerate(turns) if "🛑" in turn])
        row["context_short"] = "\n".join(
            turns[stop_idx - context_size:stop_idx + context_size + 1]
        )
        row["input_text"] = "\n".join([
            f"Dialog: {row['context_short']}",
            "",
            f"Question: {row['question']}"
        ])
        return row
    raw = raw.map(add_input_text)
    # Split the data cannonically.
    test_cid = 4431
    assert test_cid in raw["cid"]
    return ds.DatasetDict({
        "train": raw.filter(lambda r: r["cid"] != test_cid),
        "test": raw.filter(lambda r: r["cid"] == test_cid)
    })
