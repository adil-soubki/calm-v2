# -*- coding: utf-8 -*-
import abc

import datasets
from sklearn.model_selection import KFold


TASKS = {}
def register(task: Task) -> None:
    TASKS[task.name] = task


class Task(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: pass

    @property
    @abc.abstractmethod
    def text_column(self) -> str: pass

    @property
    @abc.abstractmethod
    def label_column(self) -> str: pass

    @property
    @abc.abstractmethod
    def has_canonical_splits(self) -> bool: pass

    @abc.abstractmethod
    def load_dataset(self) -> datasets.Dataset:
        pass

    def load(
        self, fold: int = 0, num_folds: int = 5, seed: int = 42
    ) -> datasets.DatasetDict:
        # Fail early if arguments are bad.
        assert fold >= 0 and fold <= num_folds - 1
        if self.has_canonical_splits and fold != 0:
            raise ValueError("Only fold 0 exists for tasks with canonical splits")
        data = self.load_dataset()
        # If there are no canonical splits do kfold splitting.
        if not self.has_canonical_splits:
            return split_kfold(data, fold, num_folds, seed)
        # Otherwise use the existing splits (specified in the 'split' column).
        if self.has_canonical_splits and "split" not in data:
            raise ValueError("Expected a 'split' column for task with canonical splits")
        splits = sorted(set(test["split"]))
        return datasets.DatasetDict({
            split: data.filter(lambda r: r["split"] == split) for split in splits
        })


def split_kfold(
    data: datasets.Dataset, fold: int = 0, num_folds: int = 5, seed: int = 42
) -> datasets.DatasetDict:
    assert fold >= 0 and fold <= num_folds - 1
    kf = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
    train_idxs, test_idxs = list(kf.split(data))[fold]
    return datasets.DatasetDict({
        "train": data.select(train_idxs),
        "test": data.select(test_idxs),
    })
