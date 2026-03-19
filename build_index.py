"""
 Build the FAISS semantic-search index from the CSV dataset.
"""
import argparse
import json
import re

import faiss
import pandas as pd

from utils import (
    INDEX_DIR,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    embed_texts,
    DATASET_DIR
)

from transformers.utils import logging
logging.set_verbosity_error()


def clean_description(raw: str) -> str:
    """Basic cleanup for CSV text fields."""
    return str(raw).strip().strip("'\"").strip()


def build_index(csv_path: str = DATASET_DIR, batch_size: int = 64) -> None:
    print(f"Loading dataset from '{csv_path}'…")
    df = pd.read_csv(csv_path)

    # validate schema
    if "description" not in df.columns or "svg" not in df.columns:
        raise ValueError("CSV must have 'description' and 'svg' columns.")

    # clean descriptions
    df["description"] = df["description"].apply(clean_description)
    n = len(df)
    print(f"  {n} entries loaded.")

    # generate embeddings
    print("Generating embeddings")
    embeddings = embed_texts(df["description"].tolist(), batch_size=batch_size)
    dim = embeddings.shape[1]

    # build FAISS index (cosine similarity via normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_FILE))

    # build metadata store
    metadata = {
        "model": "all-MiniLM-L6-v2",
        "dim": dim,
        "entries": [
            {"id": int(i), "description": row["description"], "svg": row["svg"]}
            for i, row in df.iterrows()
        ],
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    print("\nIndex built successfully.")
    print(f"  Entries  : {n}")
    print(f"  Dimension: {dim}")
    print(f"  Files    : {FAISS_INDEX_FILE}, {METADATA_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SVG semantic-search index.")
    parser.add_argument("--csv", default=DATASET_DIR, help="Path to the dataset CSV.")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    args = parser.parse_args()
    build_index(args.csv, args.batch_size)
