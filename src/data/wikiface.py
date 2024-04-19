# -*- coding: utf-8 -*-
import os
from glob import glob

import datasets
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..core.path import dirparent


WF_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "wikiface")


def set_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"face_act": "face_act_adil"})

    def fn(row: pd.Series) -> pd.Series:
        # No double entries.
        assert (pd.isna(row.face_act_adil) and not pd.isna(row.face_act_shyne)) or (
            not pd.isna(row.face_act_adil) and pd.isna(row.face_act_shyne)
        ), row
        # get the annotation.
        face_act = (
            row.face_act_adil if not pd.isna(row.face_act_adil) else row.face_act_shyne
        )
        assert not pd.isna(face_act)
        row["face_act"] = face_act
        row["label"] = face_act
        return row

    return df.apply(fn, axis=1)


def load() -> datasets.Dataset:
    gstr = os.path.join(WF_DIR, "seed-annotations", "*")
    df = set_label(pd.concat([pd.read_csv(p) for p in glob(gstr)]))
    df = df.assign(sentence=df.input_text)  # Set the default input text column.
    # Create dataset.
    return datasets.Dataset.from_pandas(df, preserve_index=False)


def load_kfold(
    fold: int, k: int = 5, seed: int = 42
) -> datasets.DatasetDict:
    assert fold >= 0 and fold <= k - 1
    kf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)
    wf = load()
    train_idxs, test_idxs = list(kf.split(wf, wf["label"]))[fold]
    return datasets.DatasetDict({
        "train": wf.select(train_idxs),
        "test": wf.select(test_idxs),
    })
