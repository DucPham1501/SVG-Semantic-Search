"""
search.py  –  CLI semantic search over the SVG index.

Usage:
    python search.py "sunset over mountains" [--top-k 5] [--show-svg]
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np

from utils import FAISS_INDEX_FILE, METADATA_FILE, embed_texts, load_metadata

from transformers.utils import logging
logging.set_verbosity_error()


def search(query: str, top_k: int = 5, show_svg: bool = False) -> list[dict]:
    """
    Search top-k most similar SVGs for a query.

    Parameters:
        query : str
            Input text query.
        top_k : int, default=5
            Number of results to return.
        show_svg : bool, default=False
            Whether to include SVG content in output.

    Returns: 
        list[dict]
            List of results with fields: rank, score, description, svg, id.
        """
    # make sure index exists
    if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
        sys.exit("Index not found. Run python build_index.py first.")

    # load index + metadata
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    meta = load_metadata()
    entries = meta["entries"]

    # embed query
    q_vec = embed_texts([query])  # (1, dim)

    # search
    scores, ids = index.search(q_vec, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        entry = entries[idx]
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "description": entry["description"],
                "svg": entry["svg"],
                "id": entry["id"],
            }
        )

    return results


def print_results(results: list[dict], show_svg: bool = False) -> None:
    """Pretty print search results."""
    print(f"\n{'-'*60}")
    for r in results:
        print(f"[{r['rank']}] score={r['score']:.4f}  id={r['id']}")
        print(f"    {r['description']}")
        if show_svg:
            print(f"    SVG: {r['svg'][:120]}...")
    print(f"{'-'*60}\n")


if __name__ == "__main__":
    # CLI
    parser = argparse.ArgumentParser(description="Semantic SVG search.")
    parser.add_argument("query", help="Text query, e.g. 'sunset over mountains'")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results.")
    parser.add_argument(
        "--show-svg", action="store_true", help="Print a snippet of the SVG markup."
    )
    parser.add_argument(
        "--json", dest="as_json", action="store_true", help="Output results as JSON."
    )

    args = parser.parse_args()

    results = search(args.query, top_k=args.top_k, show_svg=args.show_svg)

    if args.as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f'\nQuery: "{args.query}"')
        print_results(results, show_svg=args.show_svg)