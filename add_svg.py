"""
add_svg.py  –  Add a new SVG to the search index.

The description is optional. When omitted, it is auto-generated using a local
LLM (SmolLM2-1.7B-Instruct via generate_description.py) and the result is
also appended to dataset.csv.

Usage:
    # With an explicit description
    python add_svg.py path/to/image.svg --description "A red circle on white"

    # Auto-generate the description with the local LLM
    python add_svg.py path/to/image.svg

"""

import argparse
import sys

import faiss
import numpy as np

from utils import (
    FAISS_INDEX_FILE,
    METADATA_FILE,
    embed_texts,
    load_metadata,
    save_metadata,
)
from generate_description import generate_svg_description

from transformers.utils import logging
logging.set_verbosity_error()


def add_svg(svg_content: str, description: str | None = None) -> dict:
    """
    Add a new SVG entry into the index.

    Auto-generates description if missing.
    Returns the inserted entry.
    """
    # index must exist first
    if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
        sys.exit("Index not found. Run python build_index.py first.")

    # generate description if not provided
    if not description:
        print("No description provided – generating with local LLM…")
        description = generate_svg_description(svg_content)
        print(f"  Generated description: {description!r}")

    # load index + metadata
    index = faiss.read_index(str(FAISS_INDEX_FILE))
    meta = load_metadata()
    entries = meta["entries"]

    # assign next id
    new_id = max((e["id"] for e in entries), default=-1) + 1

    # embed description
    embedding = embed_texts([description])  # (1, dim)

    # add to FAISS
    index.add(embedding)
    faiss.write_index(index, str(FAISS_INDEX_FILE))

    # update metadata
    new_entry = {"id": new_id, "description": description, "svg": svg_content}
    entries.append(new_entry)
    save_metadata(meta)

    print(f"Added entry id={new_id}. Index now contains {index.ntotal} entries.")
    return new_entry


if __name__ == "__main__":
    # CLI setup
    parser = argparse.ArgumentParser(description="Add a new SVG to the search index.")
    parser.add_argument(
        "svg_file",
        help="Path to an .svg file, or '-' to read from stdin.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional human-written description.",
    )
    args = parser.parse_args()

    # read svg input (file or stdin)
    if args.svg_file == "-":
        svg_content = sys.stdin.read()
    else:
        with open(args.svg_file, encoding="utf-8") as f:
            svg_content = f.read()

    add_svg(svg_content, args.description)