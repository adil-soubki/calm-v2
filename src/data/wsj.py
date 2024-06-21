# -*- coding: utf-8 -*-
import os
from glob import glob
from typing import Any

import datasets
import pandas as pd
import transformers as tf
from ..core.path import dirparent


WSJ_DIR = os.path.join(dirparent(os.path.realpath(__file__), 3), "data", "wsj")


def load() -> pd.DataFrame:
    ret = []
    for path in glob(os.path.join(WSJ_DIR, "**", "*"), recursive=False):
        with open(path, "r", encoding="latin-1") as fd:
            text = fd.read().replace(".START", "").strip()
        ret.append({"fid": int(os.path.basename(path).split("_")[1]), "text": text})
    df = pd.DataFrame(ret)
    assert len(df) == 2499
    return df


def tokenize_text(text: str, tokenizer: Any) -> list[str]:
    # Spacy tokenizer.
    if not hasattr(tokenizer, "convert_ids_to_tokens"):
        tkns = []
        for tkn in tokenizer(rec["text"]):
            tkns.append(tkn.text)
            if tkn.whitespace_:
                tkns.append(tkn.whitespace_)
    # Transformers tokenizer.
    return list(
        map(
            lambda t: tokenizer.convert_tokens_to_string([t]),
            tokenizer.convert_ids_to_tokens(tokenizer(text)["input_ids"])
        )
    )


def load_tokenized(tokenizer_name: str = "openai-community/gpt2") -> pd.DataFrame:
    ret = []
    try:
        tokenizer = tf.AutoTokenizer.from_pretrained(tokenizer_name)
    except OSError:
        tokenizer = spacy.load(tokenizer)
    for rec in load().to_dict("records"):
        tkns = tokenize_text(rec["text"], tokenizer)
        for idx in range(len(tkns) + 1):
            ret.append({
                "fid": rec["fid"],
                "stop_token_index": idx,
                "tokenizer_name": tokenizer_name,
                "text": "".join(tkns[:idx])
            })
    return pd.DataFrame.from_records(ret)
