# -*- coding: utf-8 -*-
import dataclasses
from typing import Any

import datasets

from ..data import (
    commitment_bank, commitment_bank_text_only, common_tom, fact_bank,
    fantom, goemotions, iemocap, simple_tom, super_glue, wikiface, wsj
)


DMAP = {
    "commitment_bank": commitment_bank,
    "commitment_bank_text_only": commitment_bank_text_only,
    "fact_bank": fact_bank,
    "fantom": fantom,
    "simple_tom": simple_tom,
    "wikiface": wikiface,
    "wsj": wsj,
}
TASKS = list(DMAP) + list(super_glue.TASK_TO_FN) + ["common_tom", "imdb", "iemocap"]


def load_kfold(name: str, fold: int, k: int = 5, seed: int = 42, **data_kwargs: Any):
    if name in DMAP:
        return DMAP[name].load_kfold(**data_kwargs, fold=fold, k=k, seed=seed)
    # If we are asking for the first fold just return the default splits.
    elif name in super_glue.TASKS and fold == 0:
        return load(name, seed, **data_kwargs)
    elif name in ("common_tom", "imdb", "iemocap", "goemotions") and fold == 0:
        return load(name, seed, **data_kwargs)
    raise NotImplementedError


def load(name: str, seed: int = 42, **data_kwargs: Any):
    if name in DMAP:
        return DMAP[name].load_kfold(**data_kwargs, fold=0, k=5, seed=seed)
    elif name in super_glue.TASKS:
        return super_glue.load(name, **data_kwargs)
    elif name == "common_tom":
        return common_tom.load(**data_kwargs)
    elif name == "imdb":
        imdb = datasets.load_dataset("stanfordnlp/imdb")
        del imdb["unsupervised"]  # Unused and unlabeled.
        return imdb
    elif name == "iemocap":
        return iemocap.load()
    elif name == "goemotions":
        return goemotions.load(**data_kwargs)
    raise NotImplementedError
