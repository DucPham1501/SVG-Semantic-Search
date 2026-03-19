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

# Import utility that converts hex colors to common English color names
# This is used only for the LLM prompt so the model understands colors better.
from utils import normalize_svg_colors, DATASET_DIR


# ── Paths ─────────────────────────────────────────────────────────────────────
PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompt" / "make_description.txt"

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MAX_INPUT_TOKENS = 8192  # SmolLM2 context window
MAX_NEW_TOKENS = 15     # descriptions are short (5-12 words)

_model = None       # lazy singletons
_tokenizer = None


# ── model loader ─────────────────────────────────────────────────────────

def _get_model_and_tokenizer():
    """Return cached (tokenizer, model) for SmolLM2."""
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


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(svg_code: str) -> str:
    """Load the few-shot template and inject *svg_code*."""
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace("{svg_code}", svg_code)


# ── Output cleaner ────────────────────────────────────────────────────────────

def _clean_output(raw: str) -> str:
    """
    Strip any leftover artefacts from TinyLlama output and return a single sentence.

    SmolLM2 sometimes prefixes the answer with 'Description' or a newline.
    """
    # Remove leading label words
    text = re.sub(r"(?i)^(description\s*[:\-]?\s*)", "", raw.strip())
    text = re.sub(r"(?i)^this svg represents\s*", "", text)
    text = re.sub(r"(?i)^The SVG represents\s*", "", text)
    # Keep only the first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return re.sub(r"\s{2,}", " ", line)
    return text.strip()


# ── CSV helper ────────────────────────────────────────────────────────────────

def _append_to_csv(description: str, svg_code: str) -> None:
    """Append one row (description, svg) to dataset.csv, creating headers if needed."""
    file_exists = DATASET_DIR.exists() and DATASET_DIR.stat().st_size > 0
    with open(DATASET_DIR, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["description", "svg"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"description": description, "svg": svg_code})


# ── Public API ────────────────────────────────────────────────────────────────

def generate_svg_description(svg_code: str) -> str:
    """
    Generate a one-sentence description of *svg_code* using a local LLM,
    append the result to dataset.csv, and return the description string.

    Parameters
    ----------
    svg_code : str
        Raw SVG markup.

    Returns
    -------
    str
        A short natural-language description (5–12 words).
    """
    tokenizer, model = _get_model_and_tokenizer()

    # ── SVG preprocessing for the LLM ─────────────────────────────────────────
    # Convert hex colors (#RRGGBB) to common English color names
    # Example: "#1E3F66" → "darkblue"
    # This helps the language model better understand the visual semantics.
    # IMPORTANT: the original SVG is preserved and saved to the dataset.
    svg_for_llm = normalize_svg_colors(svg_code)

    # Optional whitespace normalization to slightly reduce token count
    svg_for_llm = re.sub(r"\s+", " ", svg_for_llm)

    # Wrap the few-shot prompt inside Phi-3's chat template so the instruct
    # model knows it should produce the assistant reply (the description).
    prompt = _build_prompt(svg_for_llm)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # appends <|assistant|>\n
    )

    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,       # greedy — deterministic, faster on CPU
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Decode only the newly generated tokens (skip the echoed prompt).
    new_tokens = output_ids[0, input_length:]
    raw_description = tokenizer.decode(new_tokens, skip_special_tokens=True)
    description = _clean_output(raw_description)

    # Save the ORIGINAL SVG (not the normalized one) to the dataset
    _append_to_csv(description, svg_code)
    print(f"Description saved to {DATASET_DIR}: {description!r}")

    return description


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_description.py <path/to/file.svg>")
        sys.exit(1)

    svg_path = Path(sys.argv[1])
    svg_content = svg_path.read_text(encoding="utf-8")

    desc = generate_svg_description(svg_content)
    print(f"\nGenerated description:\n  {desc}")