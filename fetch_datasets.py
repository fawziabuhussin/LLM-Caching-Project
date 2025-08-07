#!/usr/bin/env python3
"""
fetch_datasets.py

Demonstrates how to download/use and save to CSV:
  1) Hugging Face Datasets (WikiText-103)
  2) UCI ML Repo (Wine Quality)
  3) OpenML (MNIST)
  4) Hugging Face openwebtext & bookcorpus for short/long prompts
  5) Hugging Face The Pile (public) to load and sample examples
  6) Stack Exchange (via StackAPI)

Requires:
  pip install pandas openml datasets stackapi

Note: For The Pile, ensure you have `datasets` >=1.18 and `trust_remote_code=True`.
"""

import pandas as pd
import openml
from datasets import load_dataset
from stackapi import StackAPI

OUTPUT_DIR = "./output_csv"


def ensure_dir():
    import os
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def hf_wikitext(n=1000):
    ds = load_dataset("wikitext", "wikitext-103-v1", split=f"train[:{n}]")
    df = pd.DataFrame([{"text": ex["text"]} for ex in ds])
    df.to_csv(f"{OUTPUT_DIR}/wikitext103.csv", index=False)


def uci_wine_quality():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    df.to_csv(f"{OUTPUT_DIR}/winequality_red.csv", index=False)


def openml_mnist():
    ds = openml.datasets.get_dataset("mnist_784")
    X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
    df = X.copy()
    df[ds.default_target_attribute] = y
    df.to_csv(f"{OUTPUT_DIR}/mnist.csv", index=False)


def hf_short_long(n=100):
    short = load_dataset("openwebtext", split=f"train[:{n}]")
    df_short = pd.DataFrame([{"text": ex["text"]} for ex in short])
    df_short.to_csv(f"{OUTPUT_DIR}/openwebtext.csv", index=False)

    long = load_dataset("bookcorpus", split=f"train[:{n}]")
    df_long = pd.DataFrame([{"text": ex["text"]} for ex in long])
    df_long.to_csv(f"{OUTPUT_DIR}/bookcorpus.csv", index=False)


def hf_pile_examples(n=200):
    ds = load_dataset(
        "EleutherAI/pile", name="all", split=f"train[:{n}]", trust_remote_code=True
    )
    df = pd.DataFrame([
        {"text": ex["text"], "subset": ex["meta"]["pile_set_name"]}
        for ex in ds
    ])
    df.to_csv(f"{OUTPUT_DIR}/pile_sample.csv", index=False)


def stackexchange_questions(site="stackoverflow", n=100):
    api = StackAPI(site)
    qs = api.fetch("questions", pagesize=n, filter="withbody")
    df = pd.DataFrame([{"title": q["title"], "body": q["body"]} for q in qs["items"]])
    df.to_csv(f"{OUTPUT_DIR}/stackexchange_{site}.csv", index=False)


if __name__ == "__main__":
    ensure_dir()
    hf_wikitext()
    uci_wine_quality()
    openml_mnist()
    hf_short_long()
    hf_pile_examples()
    stackexchange_questions()
