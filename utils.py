

import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


_HERE = Path(__file__).parent
INDEX_DIR = _HERE / "index"
METADATA_FILE = INDEX_DIR / "metadata.json"
FAISS_INDEX_FILE = INDEX_DIR / "vectors.index"
DATASET_DIR = _HERE / "dataset" / "dataset.csv"

COMMON_COLORS = {
"black":"#000000","white":"#FFFFFF","gray":"#808080","silver":"#C0C0C0",
"red":"#FF0000","darkred":"#8B0000","crimson":"#DC143C","salmon":"#FA8072",
"orange":"#FFA500","darkorange":"#FF8C00","coral":"#FF7F50","tomato":"#FF6347",
"yellow":"#FFFF00","gold":"#FFD700","khaki":"#F0E68C",
"green":"#008000","darkgreen":"#006400","lime":"#00FF00","limegreen":"#32CD32","forestgreen":"#228B22",
"blue":"#0000FF","darkblue":"#00008B","navy":"#000080","royalblue":"#4169E1","skyblue":"#87CEEB","deepskyblue":"#00BFFF",
"cyan":"#00FFFF","teal":"#008080","turquoise":"#40E0D0",
"purple":"#800080","indigo":"#4B0082","violet":"#EE82EE","magenta":"#FF00FF",
"pink":"#FFC0CB","hotpink":"#FF69B4",
"brown":"#8B4513","sienna":"#A0522D","chocolate":"#D2691E","tan":"#D2B48C",
"beige":"#F5F5DC","wheat":"#F5DEB3","ivory":"#FFFFF0"
}

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None  # lazy singleton


def get_model():
    """Return the (cached) SentenceTransformer model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_metadata() -> dict:
    if METADATA_FILE.exists():
        with open(METADATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"entries": []}


def save_metadata(meta: dict) -> None:
    INDEX_DIR.mkdir(exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def extract_svg_description(svg_content: str) -> str:
    """
    Auto-generate a natural-language description from SVG markup.

    Strategy (in priority order):
      1. HTML comments  – SVG authors typically embed human-readable notes
      2. <title> / <desc> elements  – standard SVG metadata elements
      3. Visible <text> elements
      4. Structural fallback: dominant shapes and colour palette
    """
    parts: list[str] = []

    # 1. Comments
    comments = re.findall(r"<!--(.*?)-->", svg_content, re.DOTALL)
    comment_text = "; ".join(c.strip() for c in comments if len(c.strip()) > 3)
    if comment_text:
        parts.append(comment_text)

    # 2 & 3. Parse XML
    try:
        # Strip namespaces so ElementTree tag lookups stay simple
        svg_clean = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', "", svg_content)
        root = ET.fromstring(svg_clean)

        # <title> and <desc>
        for tag in ("title", "desc"):
            for el in root.iter(tag):
                if el.text and el.text.strip():
                    parts.append(el.text.strip())

        # Visible text
        text_items = [
            el.text.strip()
            for el in root.iter("text")
            if el.text and el.text.strip()
        ]
        if text_items:
            parts.append("text: " + " ".join(text_items[:5]))

        # 4. Structural fallback
        if not parts:
            shape_counts: dict[str, int] = {}
            colors: set[str] = set()

            SHAPE_TAGS = {
                "rect", "circle", "ellipse", "line",
                "polyline", "polygon", "path", "image",
            }
            COLOR_ATTRS = ("fill", "stroke", "stop-color", "color")

            for elem in root.iter():
                tag = elem.tag.split("}")[-1]
                if tag in SHAPE_TAGS:
                    shape_counts[tag] = shape_counts.get(tag, 0) + 1
                for attr in COLOR_ATTRS:
                    val = elem.get(attr, "").strip()
                    if val and val not in ("none", "transparent", "inherit", ""):
                        colors.add(val)

            if shape_counts:
                dominant = sorted(shape_counts.items(), key=lambda x: -x[1])[:4]
                parts.append("shapes: " + ", ".join(f"{v} {k}" for k, v in dominant))
            if colors:
                parts.append("colors: " + ", ".join(sorted(colors)[:6]))

    except ET.ParseError:
        pass

    return " | ".join(parts) if parts else "SVG graphic"


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Return L2-normalised float32 embeddings for a list of strings."""
    model = get_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 10,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0,2,4))


def closest_color(hex_color):
    r,g,b = hex_to_rgb(hex_color)
    return min(
        COMMON_COLORS,
        key=lambda name: sum(
            (a-b)**2 for a,b in zip((r,g,b),hex_to_rgb(COMMON_COLORS[name]))
        )
    )


def normalize_svg_colors(svg):
    return re.sub(
        r"#([0-9a-fA-F]{6})",
        lambda m: closest_color(m.group(0)),
        svg
    )