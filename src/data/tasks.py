# -*- coding: utf-8 -*-
import dataclasses
from typing import Any

import datasets

from ..data import commitment_bank, fact_bank, super_glue, wikiface, wsj


DMAP = {
    "commitment_bank": commitment_bank,
    "fact_bank": fact_bank,
    "wikiface": wikiface,
    "wsj": wsj,
}


def load_kfold(name: str, fold: int, k: int = 5, seed: int = 42, **data_kwargs: Any):
    if name in DMAP:
        return DMAP[name].load_kfold(**data_kwargs, fold=fold, k=k, seed=seed)
    # If we are asking for the first fold just return the default splits.
    elif name in super_glue.TASKS and fold == 0:
        return load(name, seed, **data_kwargs)
    raise NotImplementedError


def load(name: str, seed: int = 42, **data_kwargs: Any):
    if name in DMAP:
        return DMAP[name].load_kfold(**data_kwargs, fold=0, k=5, seed=seed)
    elif name in super_glue.TASKS:
        return super_glue.load(name, **data_kwargs)
    raise NotImplementedError
