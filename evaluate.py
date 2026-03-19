"""
evaluate.py  –  Measure retrieval quality of the SVG semantic-search index.

Ground-truth assumption:
    Each (description, svg) pair in the dataset is treated as a query whose
    single correct answer is its own SVG.  For each query, we embed the
    description and check at what rank the correct entry appears.

Metrics reported:
    Recall@K   (K = 1, 5, 10)  – fraction of queries where the correct item
                                   appears in the top-K results
    MRR        (Mean Reciprocal Rank) – average of 1/rank over all queries

Usage:
    # Evaluate on a random 200-query sample
    python evaluate.py --sample 200

    # Evaluate on ALL queries (slower)
    python evaluate.py --sample 0
"""

import argparse
import random

import faiss
import numpy as np

from utils import FAISS_INDEX_FILE, METADATA_FILE, embed_texts, load_metadata
from transformers.utils import logging
logging.set_verbosity_error()


def evaluate(sample: int = 200, seed: int = 42) -> None:
    """Evaluate retrieval quality using Recall@K and MRR."""
    # make sure index exists
    if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError("Index not found. Run python build_index.py first.")

    # load index + metadata
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    meta = load_metadata()
    entries = meta["entries"]
    n = len(entries)

    # sample queries
    indices = list(range(n))
    if sample and sample < n:
        random.seed(seed)
        indices = random.sample(indices, sample)

    print(f"Evaluating {len(indices)} queries out of {n} entries…")

    queries = [entries[i]["description"] for i in indices]

    # embed all queries
    q_vecs = embed_texts(queries, batch_size=64)

    # search top-K
    K = 10
    scores_all, ids_all = index.search(q_vecs, K)

    # compute metrics
    reciprocal_ranks: list[float] = []
    found_at: list[int] = []  # rank where match is found (1-indexed)

    for q_pos, true_idx in enumerate(indices):
        retrieved = ids_all[q_pos].tolist()
        rank = K + 1

        for r, ret_idx in enumerate(retrieved, start=1):
            if ret_idx == true_idx:
                rank = r
                break

        found_at.append(rank)
        reciprocal_ranks.append(1.0 / rank if rank <= K else 0.0)

    found_at_arr = np.array(found_at)

    print("\n" + "=" * 50)
    print("  Retrieval Evaluation Results")
    print("=" * 50)

    for k in (1, 5, 10):
        recall = (found_at_arr <= k).mean()
        print(f"  Recall@{k:<2}  : {recall:.4f}  ({(found_at_arr <= k).sum()}/{len(indices)})")

    mrr = np.mean(reciprocal_ranks)
    print(f"  MRR        : {mrr:.4f}")
    print("=" * 50)

    # quick examples
    print("\nSample results (first 5 queries):")
    for i in range(min(5, len(indices))):
        true_idx = indices[i]
        rank = found_at[i]

        q_desc = entries[true_idx]["description"]
        top1_idx = ids_all[i][0]
        top1_desc = entries[top1_idx]["description"]

        print(f"\n  Query    : {q_desc[:80]}")
        print(f"  Top-1    : {top1_desc[:80]}")
        print(f"  Correct? : {'YES (rank 1)' if rank == 1 else f'NO (found at rank {rank})'}")
        print(f"  Score    : {scores_all[i][0]:.4f}")


if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description="Evaluate SVG retrieval quality.")
    parser.add_argument(
        "--sample",
        type=int,
        default=200,
        help="Number of queries to evaluate (0 = all).",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    evaluate(args.sample, args.seed)