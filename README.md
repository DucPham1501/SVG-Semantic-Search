# SVG Semantic Search

SVG Semantic Search is an end-to-end semantic retrieval system that enables searching SVG images using natural language queries. The system leverages lightweight transformer-based embeddings to encode textual descriptions into dense vector representations, which are indexed using FAISS for efficient similarity search. By applying vector normalization and cosine similarity, the system delivers accurate and low-latency retrieval of semantically relevant SVG assets.

The architecture is organized into two core components: the Indexing Pipeline and the Query Pipeline. The Indexing Pipeline processes and transforms raw dataset entries into a structured vector index, while the Query Pipeline performs real-time semantic search over this index. This separation ensures high performance, scalability, and a clean modular design, making the system easy to extend into multimodal retrieval or production-grade search services.

---

## 🔄 System Flow

                ┌──────────────────────┐
                │     CSV Dataset      │
                │ (description, SVG)   │
                └─────────┬────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │   Embedding Model    │
                │ (text → vector)      │
                └─────────┬────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │      FAISS Index     │
                │ (vector storage)     │
                └─────────┬────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
┌───────────────┐                 ┌────────────────────┐
│   User Query  │                 │   Add New SVG      │
│ (description) │                 │                    │
└───────┬───────┘                 └─────────┬──────────┘
        │                                   │
        ▼                                   ▼
┌───────────────┐               ┌────────────────────────────┐
│   Embedding   │               │   Has Description?         │
│ (query → vec) │               └─────────┬──────────────────┘
└───────┬───────┘                         │
        │                                 │ No
        ▼                                 ▼
┌───────────────┐               ┌────────────────────────────┐
│  Similarity   │               │   LLM Generates Description│
│    Search     │               └─────────┬──────────────────┘
└───────┬───────┘                         │
        │                                 ▼
        ▼                      ┌────────────────────────────┐
┌───────────────┐             │  Embedding (new desc)       │
│   Top-K SVG   │             └─────────┬──────────────────┘
│   Results     │                       │
└───────────────┘                       ▼
                                ┌────────────────────────────┐
                                │     Update FAISS Index     │
                                └────────────────────────────┘

## Key Features

- Semantic search using dense embeddings instead of keyword matching  
- Fast and scalable retrieval powered by FAISS vector indexing  
- Automatic description generation using a local LLM for unlabeled SVGs  
- Incremental indexing without rebuilding the entire system  
- Modular pipeline design (Indexing vs Query separation)  

## Environment setup

```bash
pip install -r requirements.txt
```

---

## Quick-start

### 1. Build the index

```bash
python build_index.py --csv dataset/dataset.csv
```

Reads `dataset.csv`, generates 384-dimensional embeddings for each description
using `all-MiniLM-L6-v2`, and writes two files into `index/`:

```
index/
  vectors.index   # FAISS flat inner-product index
  metadata.json   # descriptions + SVG markup for every entry
```

---

### 2. Run search queries

```bash
python search.py "sunset over rolling hills" --top-k 5
```

Example output:

```
Query: "sunset over rolling hills"

------------------------------------------------------------
[1] score=0.7823  id=14
    A vibrant sunset over rolling hills
[2] score=0.7631  id=201
    Sunset over a tranquil lake with pine trees
[3] score=0.7518  id=88
    Golden wheat fields under a setting sun
...
------------------------------------------------------------
```

Options:

```
--top-k INT      Number of results to return  (default: 5)
--show-svg       Print a snippet of the SVG markup for each result
--json           Output results as JSON
```

More example queries:

```bash
python search.py "golden wheat fields"
python search.py "a cat sitting on a windowsill" --show-svg
python search.py "abstract blue gradient background" --top-k 10
python search.py "mountain lake reflection" --json
```

---

### 3. Add a new SVG

**With an explicit description:**

```bash
python add_svg.py sample/my_image.svg --description "A red circle on white background"
```

**Without a description (auto-generated by local LLM):**

```bash
python add_svg.py sample/my_image.svg
```

When no description is provided, the system automatically generates one using
**`HuggingFaceTB/SmolLM2-1.7B-Instruct`** — a small local LLM that runs
entirely on your machine with no API key required. The flow is:

1. The SVG is fed through a few-shot prompt (`prompt/make_description.txt`)
2. SmolLM2 generates a single concise description sentence
3. The description and SVG are appended to `dataset.csv`
4. The description is embedded and added to the FAISS index — no full rebuild needed

> **First run:** model weights (~3.4 GB) are downloaded from HuggingFace and
> cached locally. Subsequent calls load from cache.

---

### 4. Evaluate retrieval quality

```bash
python evaluate.py --sample 200   # random 200-query sample
python evaluate.py --sample 0     # all 1 000 queries
```
---

## Results (200-query random sample, seed = 42)

| Metric | Value |
|---|---|
| Recall@1 | **0.640** (128 / 200) |
| Recall@5 | **0.955** (191 / 200) |
| Recall@10 | **0.995** (199 / 200) |
| MRR | **0.781** |

- **Recall@1 = 0.64**: nearly two-thirds of queries surface the exact correct
  SVG as the top result. The jump to Recall@5 (0.955) shows that most failures
  are off-by-a-few rather than completely wrong — the correct item is very
  close.
- **MRR = 0.78**: on average the correct result appears at approximately rank
  1.3, which is strong for a zero-shot dense retrieval system with no
  fine-tuning.
- Several Recall@1 failures are caused by near-duplicate descriptions in the
  dataset (e.g., `"A vibrant sunset over rolling hills"` vs
  `"A vibrant sunset over rolling hills',"` — a trailing punctuation artefact
  that gives two items a cosine score of 1.000). These are not true retrieval
  errors.
---

## Project layout

```
.
├── dataset.csv                 description–SVG pairs (source of truth)
├── requirements.txt
├── build_index.py              build FAISS index from CSV
├── search.py                   CLI semantic search
├── add_svg.py                  add new SVG (with or without description)
├── generate_description.py     LLM-based description generation (SmolLM2)
├── evaluate.py                 retrieval quality evaluation
├── utils.py                    shared helpers (embedding, I/O)
├── README.md
├── writeup.md                  design decisions and evaluation write-up
├── sample/                    sample SVG files for testing
│   └── *.svg
├── prompt/
│   └── make_description.txt    few-shot prompt template for SmolLM2
└── index/                      (created by build_index.py)
    ├── vectors.index
    └── metadata.json
```
