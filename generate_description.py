"""
generate_description.py  –  Auto-generate an SVG description via a local LLM
                             and persist the result to dataset.csv.

Usage (module):
    from generate_description import generate_svg_description

    with open("ball.svg", encoding="utf-8") as f:
        svg_code = f.read()

    description = generate_svg_description(svg_code)
    print(description)

Usage (CLI):
    python generate_description.py ball.svg
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
logging.set_verbosity_error()

from utils import normalize_svg_colors, DATASET_DIR


# paths
PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompt" / "make_description.txt"

# model config
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MAX_INPUT_TOKENS = 8192
MAX_NEW_TOKENS = 15  # short descriptions

_model = None
_tokenizer = None


def _get_model_and_tokenizer():
    """Lazy load + cache model."""
    global _model, _tokenizer
    if _model is None:
        print(f"Loading local LLM '{MODEL_NAME}' (first call only)…")

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
        )
        _model.eval()
        print("Model ready.")

    return _tokenizer, _model


def _build_prompt(svg_code: str) -> str:
    """Inject SVG into prompt template."""
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace("{svg_code}", svg_code)


def _clean_output(raw: str) -> str:
    """Clean model output to a single short sentence."""
    text = re.sub(r"(?i)^(description\s*[:\-]?\s*)", "", raw.strip())
    text = re.sub(r"(?i)^this svg represents\s*", "", text)
    text = re.sub(r"(?i)^the svg represents\s*", "", text)

    for line in text.splitlines():
        line = line.strip()
        if line:
            return re.sub(r"\s{2,}", " ", line)

    return text.strip()


def _append_to_csv(description: str, svg_code: str) -> None:
    """Append (description, svg) to dataset."""
    file_exists = DATASET_DIR.exists() and DATASET_DIR.stat().st_size > 0

    with open(DATASET_DIR, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["description", "svg"])

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "description": description,
            "svg": svg_code
        })


def generate_svg_description(svg_code: str) -> str:
    """
    Generate a short description for an SVG and store it.
    """
    tokenizer, model = _get_model_and_tokenizer()

    # normalize colors for better LLM understanding
    svg_for_llm = normalize_svg_colors(svg_code)

    # reduce token size a bit
    svg_for_llm = re.sub(r"\s+", " ", svg_for_llm)

    # build prompt
    prompt = _build_prompt(svg_for_llm)
    messages = [{"role": "user", "content": prompt}]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )

    input_length = inputs["input_ids"].shape[1]

    # generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # decode only new tokens
    new_tokens = output_ids[0, input_length:]
    raw_description = tokenizer.decode(new_tokens, skip_special_tokens=True)

    description = _clean_output(raw_description)

    # save original SVG
    _append_to_csv(description, svg_code)
    print(f"Description saved to {DATASET_DIR}: {description!r}")

    return description


if __name__ == "__main__":
    # CLI
    if len(sys.argv) < 2:
        print("Usage: python generate_description.py <path/to/file.svg>")
        sys.exit(1)

    svg_path = Path(sys.argv[1])
    svg_content = svg_path.read_text(encoding="utf-8")

    desc = generate_svg_description(svg_content)
    print(f"\nGenerated description:\n  {desc}")