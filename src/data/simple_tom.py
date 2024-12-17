# -*- coding: utf-8 -*-
from typing import Any

import datasets as ds
from sklearn.model_selection import KFold


QuestionType = ["mental-state-qa", "behavior-qa", "judgment-qa"]


def load(question_type: QuestionType) -> ds.Dataset:
    data = ds.load_dataset("allenai/SimpleToM", question_type)
    assert len(data) == 1
    data = data["test"]
    # Add column column for the input text.
    def add_input_text(row: dict[str, Any]) -> dict[str, Any]:
        assert len(row["choices"]["label"]) == len(row["choices"]["text"])
        choices = zip(row["choices"]["label"], row["choices"]["text"])
        row["input_text"] = "\n".join([
            f"Story: {row['story']}",
            "",
            f"Question: {row['question']}",
            "\n".join([f"({lbl}) {txt}" for lbl, txt in choices])
        ])
        return row
    return data.map(add_input_text)


# XXX: Should split by scenario_name so scenarios don't appear in both train and test.
def load_kfold(
    question_type: QuestionType,
    fold: int, k: int = 5, seed: int = 42
) -> ds.DatasetDict:
    assert fold >= 0 and fold <= k - 1
    kf = KFold(n_splits=k, random_state=seed, shuffle=True)
    data = load(question_type)
    scenarios = sorted(set(data["scenario_name"]))
    train_idxs, test_idxs = list(kf.split(scenarios))[fold]
    train_scenarios = [scenarios[idx] for idx in train_idxs]
    test_scenarios = [scenarios[idx] for idx in test_idxs]
    return ds.DatasetDict({
        "train": data.filter(lambda r: r["scenario_name"] in train_scenarios),
        "test": data.filter(lambda r: r["scenario_name"] in test_scenarios),
    })
