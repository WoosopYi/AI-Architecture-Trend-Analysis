<!-- Language Switcher -->
<p align="right">
  <img alt="English" src="https://img.shields.io/badge/EN-English-black?style=flat-square">
  <a href="README.kor.md">
    <img alt="한국어" src="https://img.shields.io/badge/KR-Korean-blue?style=flat-square">
  </a>
</p>

# AI‑Architecture‑Trend‑Analysis

[![PyPI - Python](https://img.shields.io/badge/python-v3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/WoosopYi/AI-Architecture-Trend-Analysis?style=social)](https://github.com/WoosopYi/AI-Architecture-Trend-Analysis/stargazers)

<img src="docs/assets/pipeline_diagram.png" width="38%" align="right" alt="Pipeline overview" />

**AI‑Architecture‑Trend‑Analysis** is a turn‑key pipeline for extracting, clustering, and interpreting architectural design trends from caption‑like text.

* **Pipeline:** Sentence‑Transformers → UMAP/HDBSCAN (BERTopic) → LLM‑assisted re‑categorisation
* **Input:** Image captions or curated descriptions *(captions can be generated separately with a VLM; this notebook starts from text)*

---

### 2) Key Features (replace the whole section with this)

##  Key Features

* **Text‑only start:** Use captions/descriptions—no manual labels needed.
* **BERTopic discovery:** Embeddings → UMAP → HDBSCAN → c‑TF‑IDF for clear, interpretable topics.
* **Domain‑aware mapping:** Re‑categorise raw topics into architect‑friendly classes via LLM or rules/prompts.
* **Prompt generation:** Turn category keywords into ready‑to‑use prompts for generative image‑AI.
* **Single‑notebook workflow:** Everything runs in `AI_Trend_Analysis.ipynb` for easy reproduction.

---



<h2>What's in this repo</h2>
<p>A quick inventory of the repo:</p>

<table>
  <thead>
    <tr>
      <th align="left">path</th>
      <th align="left">purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="AI_Trend_Analysis.ipynb"><code>AI_Trend_Analysis.ipynb</code></a></td>
      <td>Main Code</td>
    </tr>
    <tr>
      <td><a href="Data/"><code>Data/</code></a></td>
      <td>Put your input CSVs(e.g., <code>data.csv</code>)</td>
    </tr>
    <tr>
      <td><a href="output/"><code>output/</code></a></td>
      <td>Outputs(CSV/JSON/플롯)</td>
    </tr>
    <tr>
      <td><a href="prompt/"><code>prompt/</code></a></td>
      <td>LLM Prompts</td>
    </tr>
    <tr>
      <td><a href="requirements1.txt"><code>requirements1.txt</code></a></td>
      <td>Python Dependencies</td>
    </tr>
    <tr>
      <td><a href="LICENSE"><code>LICENSE</code></a></td>
      <td>MIT License</td>
    </tr>
    <tr>
      <td><a href="README.md"><code>README.md</code></a> · <a href="README.kor.md"><code>README.kor.md</code></a></td>
      <td>Documentation</td>
    </tr>
  </tbody>
</table>



<details>
<summary><strong>Folder Tree</strong> (click to expand)</summary>

```text
AI-Architecture-Trend-Analysis/
├── AI_Trend_Analysis.ipynb
├── Data/
│   └── data.csv                
├── docs/
│   └── assets/
│       └── pipeline_diagram.png
├── output/
├── prompt/
├── requirements1.txt
├── LICENSE
├── README_kor.md
└── README.md
````

</details>

---

##  Quick Start

Clone, install, prepare data, (optionally) add a local GGUF model, then run:

```bash
git clone https://github.com/WoosopYi/AI-Architecture-Trend-Analysis.git
cd AI-Architecture-Trend-Analysis

# 1) Install dependencies (tested on Python 3.10+)
pip install -r requirements1.txt

# If you do not have a requirements file (or to mirror the notebook setup):
pip install --upgrade pip plotly kaleido
pip install "llama-cpp-python" bertopic datasets jinja2 spacy spacy-transformers
python -m spacy download en_core_web_trf

# 2) Prepare your input CSV at ./Data/data.csv (see schema in README)

# 3) (Optional, for local LLM topic labeling) Put a GGUF model at ./model/
# Example: openhermes-2.5-mistral-7b.Q4_K_M.gguf (quantized)
# mkdir -p model && cd model
# wget https://huggingface.co/.../openhermes-2.5-mistral-7b.Q4_K_M.gguf

# 4) Run the notebook top → bottom
jupyter lab AI_Trend_Analysis.ipynb
```

**Hardware tips**

* GPU strongly recommended for `BAAI/bge-m3` embeddings and for `llama-cpp-python` acceleration (`n_gpu_layers=-1` in the notebook).
* CPU‑only works but will be slower.

---

##  Input data schema (CSV)

Place a CSV at `./Data/data.csv` with at least a **`description`** column containing caption‑like text per image/project:

```csv
description
"Slender concrete structure with vertical fins and recessed glazing..."
"Timber lattice facade with perforated metal screens..."
"Terraced green roof massing with porous podium..."
```

> The notebook currently loads: `load_dataset("csv", data_files="Data/data.csv", encoding="cp949")["train"]` and uses `dataset["description"]`. If your file is UTF‑8, change `encoding` to `"utf-8-sig"`.

---
Got it. Below is a **drop‑in replacement** for the **Step‑by‑Step** section only.
It now **matches `AI_Trend_Analysis.ipynb` exactly** (paths, function names, arguments). I’ve also added a one‑line **Core idea** under each step.

Key fixes vs. your previous draft:

* Input CSV path → `"/app/paper/data/data.csv"`
* Local GGUF model path → `"/app/paper/model/openhermes-2.5-mistral-7b.Q4_K_M.gguf"`
* Outputs → `"/app/Github/output/..."`
* Kept the exact function names/signatures used in the notebook (e.g., `spacy_tokenizer`, `LlamaCPP`, `visualize_*`, etc.)

---

##  Step‑by‑Step

### 0) Environment setup (required library installation)

Install all dependencies used in the notebook; run once when you set up the environment.

```bash
!pip install --upgrade plotly kaleido
!pip install --upgrade pip

# Install llama-cpp-python (CMake options can be adjusted)
...
!pip install jinja2 # For token-level distribution visualization
!pip install spacy spacy-transformers en-core-web-trf
!python -m spacy download en_core_web_trf

# (Optional) GPU acceleration stacks
!pip install cudf-cu12 dask-cudf-cu12 --extra-index-url=https://pypi.nvidia.com
!pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
!pip install cugraph-cu12 --extra-index-url=https://pypi.nvidia.com
!pip install cupy-cuda12x -f https://pip.cupy.dev/aarch64

# (Optional) Extra visualization library
!git clone https://github.com/TutteInstitute/datamapplot.git
!pip install datamapplot/.
```

---

### 0‑B) (Optional) Download a quantized LLM

Download a local quantized GGUF model once and place it under `/model` for faster/offline topic labeling.

```bash
#!wget https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf
```

Save it to: `/app/paper/model/openhermes-2.5-mistral-7b.Q4_K_M.gguf`

---

### 1) Imports & spaCy pipeline loading

Load core libraries and the `en_core_web_trf` spaCy model used by the custom tokenizer.

```python
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import spacy
nlp = spacy.load("en_core_web_trf")

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

import plotly
import plotly.express as px

from datasets import load_dataset
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# BERTopic related
from bertopic.representation import KeyBERTInspired, LlamaCPP
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
```

---

### 2) Custom tokenizer (spaCy‑based)

Clean tokens using spaCy + domain stopwords to improve topic quality and interpretability.

```python
domain_stopwords = {
    "set", "none", "nothing", "unidentified", "not", "unvisible",
    "there", "was", "is", "in", "of", "feature", "building", "fa?ade",
    "structure", "context", "element", "architectural", "emphasize",
    "visual", "create", "clearly", "include", "overhead", "provide",
    "compose", "highlight", "dramatic", "area", "earth", "form", "mass",
    "perspective", "style", "scale", "absence", "depth", "proportion",
    # ... (list continues in the notebook)
}

def spacy_tokenizer(doc_text: str):
    doc = nlp(doc_text)
    tokens = []
    for token in doc:
        lemma = token.lemma_.lower().strip()

        # 1) remove basic stopwords/punct/whitespace/very short tokens
        if token.is_stop or token.is_punct or lemma == "" or len(lemma) < 2:
            continue

        # 2) remove domain stopwords
        if lemma in domain_stopwords:
            continue

        tokens.append(lemma)
    return tokens
```

---

### 3) Load your CSV

Read caption‑like descriptions from CSV; the notebook expects a `description` column.

```python
from datasets import load_dataset

dataset = load_dataset(
    "csv",
    data_files="/app/paper/data/data.csv",
    encoding='cp949'  # or 'utf-8-sig'
)["train"]

docs = dataset["description"]

print("Sample document count::", len(docs))
print("First document example:", docs[0])
```

---

### 4) Load local Llama model (`llama_cpp_python`)

Configure your local GGUF LLM for topic labeling (used via BERTopic’s `LlamaCPP` representation).

```python
llm = Llama(
    model_path="/app/paper/model/openhermes-2.5-mistral-7b.Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=4000,
    stop=["Q:", "\n"]
)

label_prompt = """Q:

You are an expert in architecture.

I have a topic that contains the following documents:

[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the above information, can you give a short label of the topic of at most 5 words? 
Focus on architectural keywords

A:
"""

representation_model = {
    "KeyBERT": KeyBERTInspired(),
    "LLM": LlamaCPP(llm, prompt=label_prompt),
}
```

---

### 5) Embedding model & vectorizer

Encode documents with `BAAI/bge-m3` and tokenize with the custom spaCy‑backed vectorizer.

```python
# (1) SentenceTransformer embedding
embedding_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
embeddings = embedding_model.encode(docs, show_progress_bar=True)

# (2) Custom CountVectorizer (spaCy)
custom_vectorizer = CountVectorizer(
    tokenizer=spacy_tokenizer,
    token_pattern=None,
    # (optional) min_df, max_df, ngram_range, ...
)
```

---

### 6) UMAP/HDBSCAN (clustering)

Reduce embedding dimensionality and discover dense clusters for topic modeling.

```python
# UMAP/HDBSCAN used for actual clustering
umap_model = UMAP(
    n_neighbors=6,
    n_components=4,
    min_dist=1,
    spread=1.5,
    metric='cosine',
    random_state=32
)

hdbscan_model = HDBSCAN(
    min_cluster_size=5,
    metric='euclidean',
    cluster_selection_method='eom',
    prediction_data=True
)

# 2-D UMAP for visualization
reduced_embeddings_2d = UMAP(
    n_neighbors=5,
    n_components=2,
    min_dist=1,
    spread=1.5,
    metric='cosine',
    random_state=32
).fit_transform(embeddings)
```

---

### 7) Create BERTopic & fit

Train BERTopic with your embedding/reduction/clustering pipeline and rich representations.

```python
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    representation_model=representation_model,
    top_n_words=30,
    calculate_probabilities=True,
    verbose=True,
    vectorizer_model=custom_vectorizer
)

topics, probs = topic_model.fit_transform(docs, embeddings)
topic_info = topic_model.get_topic_info()
print("Generated top 30 topic info:")
print(topic_info.head(30))
```

**Export the topic table:**

```python
info = topic_model.get_topic_info()
info.to_csv("/app/Github/output/topic_model.csv", index=True)
```

---

### 8) Visualization (overview → hierarchy → distributions)

Explore topics globally and locally with BERTopic’s built‑in interactive figures.

```python
# (1) Topics
fig_topics = topic_model.visualize_topics(); fig_topics.show()

# (2) Documents in 2D
fig_docs_2d = topic_model.visualize_documents(
    docs, reduced_embeddings=reduced_embeddings_2d, hide_document_hover=False
); fig_docs_2d.show()

# (3) Hierarchical topics (tree)
hierarchical_topics = topic_model.hierarchical_topics(docs)
fig_hierarchy = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
fig_hierarchy.show()

# (4) Hierarchical documents + topics
fig_h_docs_2d = topic_model.visualize_hierarchical_documents(
    docs, hierarchical_topics, reduced_embeddings=reduced_embeddings_2d, hide_document_hover=False
); fig_h_docs_2d.show()

# (5) Barchart of topic terms
fig_barchart = topic_model.visualize_barchart(top_n_topics=24, n_words=10); fig_barchart.show()

# (6) Topic similarity heatmap
fig_heatmap = topic_model.visualize_heatmap(n_clusters=10); fig_heatmap.show()

# (7) Term rank / score decline
fig_term_rank = topic_model.visualize_term_rank(); fig_term_rank.show()

# (8) Probability distribution for one document
fig_dist = topic_model.visualize_distribution(probs[0], min_probability=0.0); fig_dist.show()
```

---

### 9) Build keyword table per topic

Extract `(topic_id, keyword, score)` and sort; this becomes the basis for LLM re‑categorization.

```python
topic_keywords = []
all_topics = topic_model.get_topic_info().Topic.unique()
for t_id in all_topics:
    if t_id == -1:
        continue
    top_words = topic_model.get_topic(t_id)  # [(word, score), ...]
    for (w, score) in top_words:
        topic_keywords.append({
            "topic_id": int(t_id),
            "keyword": w,
            "score": float(score),
        })

df_keywords = pd.DataFrame(topic_keywords)
df_keywords.sort_values(["topic_id","score"], ascending=[True,False], inplace=True)
```

**Export JSON for LLM:**

```python
import json

items_for_llm = []
for _, row in df_keywords.iterrows():
    items_for_llm.append({
        "topic_id": int(row["topic_id"]),
        "keyword": row["keyword"],
    })

with open("/app/Github/output/topic_keywords_for_llm.json", "w", encoding="utf-8") as f:
    json.dump(items_for_llm, f, ensure_ascii=False, indent=2)
```

---

### 10) Pre‑process keywords with an LLM (outside the notebook)

Map each `(topic_id, keyword)` to an architectural **category** using the provided prompt, then save.

* Use the prompt file: `'/Github/Prompt/json_pre_process_LLM_prompt.txt'`
* Save the result as: `"/app/Github/output/architecture_category_llm_output.json"`

Example JSON shape:

```json
[
  {"topic_id":0,"keyword":"window","category":"Facade composition"},
  {"topic_id":0,"keyword":"concrete","category":"Materials and Textures"}
]
```

---

### 11) Load LLM categories & merge

Join predicted categories back to the keyword table and drop `"Unknown"`.

```python
import json
import pandas as pd

with open("/app/Github/output/architecture_category_llm_output.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df_classified = pd.DataFrame(data)

df_merged = pd.merge(
    df_keywords,
    df_classified,
    on=["topic_id","keyword"],
    how="left"
)

# Remove items with "Unknown"
df_merged = df_merged[df_merged["category"] != "Unknown"]
print(df_merged.head(210))
```

---

### 12) Aggregate: category scores & counts

Summarize **how often** each category appears and **how strong** it is via score sums.

```python
# Sum of scores per category
df_cat_scores = df_merged.groupby("category")["score"].sum().reset_index()
df_cat_scores.sort_values("score", ascending=False, inplace=True)
print(df_cat_scores)

# Count of keywords per category
cat_counts = df_merged.groupby("category")["keyword"].count().reset_index()
cat_counts = cat_counts.sort_values("keyword", ascending=False)
print("\n=== Number of keywords per category ===")
print(cat_counts)

# (Optional) another alias in the notebook
df_scoresum = df_merged.groupby("category")["score"].sum().reset_index()
df_scoresum = df_scoresum.sort_values("score", ascending=False)
print("\n=== Sum of cTFIDF per category ===")
print(df_scoresum)
```

---

### 13) Visualize category distributions

Compare categories by frequency and by total score.

```python
import plotly.express as px

cat_scores = df_merged.groupby("category")["score"].sum().reset_index(name="total_score")

fig1 = px.bar(cat_counts.sort_values("keyword", ascending=False), x="category", y="keyword",
              title="Category distribution (by keyword frequency)")
fig1.show()

fig2 = px.bar(cat_scores.sort_values("total_score", ascending=False), x="category", y="total_score",
              title="Category importance (sum of c‑TF‑IDF scores)")
fig2.show()

TOP_N = 5
top_categories_count = cat_counts.head(TOP_N)
print(f"\n=== Top {TOP_N} categories based on total keyword appearances ===")
print(top_categories_count)

TOP_N_SCORE = 5
cat_scores_sorted = cat_scores.sort_values("total_score", ascending=False)
top_categories_score = cat_scores_sorted.head(TOP_N_SCORE)
print(f"\n=== Top {TOP_N_SCORE} categories based on total keyword scores ===")
print(top_categories_score)
```

---

### 14) Top‑N keywords per category (score‑based)

For each category, list the strongest keywords (by aggregated score) and optionally plot bars.

```python
import pandas as pd
import plotly.express as px
from IPython.display import display

# ===== User settings =====
KEYWORDS_PER_CATEGORY = 10
SHOW_BAR_CHARTS = True
# =========================

# 1) Keyword-level aggregation: freq (count), total_score (sum)
agg = (df_merged.groupby(["category", "keyword"])
       .agg(freq=("keyword", "count"), total_score=("score", "sum"))
       .reset_index())

# 2) For each category, get top-N by total_score
for cat in sorted(agg["category"].unique()):
    sub = agg[agg["category"] == cat].copy()
    top_n = sub.sort_values("total_score", ascending=False).head(KEYWORDS_PER_CATEGORY)

    print(f"\n=== Category: {cat} ===")
    display(top_n)

    if SHOW_BAR_CHARTS and not top_n.empty:
        fig = px.bar(top_n, x='keyword', y='total_score', text='freq',
                     title=f"Top {KEYWORDS_PER_CATEGORY} keywords in '{cat}' (by score)")
        fig.show()
```

---

### 15) Generate example prompts for generative image‑AI

Compose short, category‑specific prompts using top keywords (score‑based or frequency‑based).

```python
def make_image_prompt(category_name, df_merged, top_k=5, by_score=True):
    """
    Select top_k keywords by score (or frequency) in a category and compose an example prompt.
    """
    subset = df_merged[df_merged["category"] == category_name]
    if subset.empty:
        return None

    if by_score:
        key_sorted = (subset.groupby("keyword")["score"]
                      .sum().reset_index(name="total_score")
                      .sort_values("total_score", ascending=False))
        top_k_words = key_sorted.head(top_k)["keyword"].tolist()
    else:
        freq_sorted = (subset.groupby("keyword")["score"]
                       .count().reset_index(name="freq")
                       .sort_values("freq", ascending=False))
        top_k_words = freq_sorted.head(top_k)["keyword"].tolist()

    if len(top_k_words) == 0:
        return None

    return (f"A futuristic '{category_name}' architecture design focusing on "
            f"{', '.join(top_k_words)}. Ultra-detailed, photorealistic rendering.")

# Example usage (use your top categories list)
for cat_name in cat_counts.head(5)["category"]:
    prompt = make_image_prompt(cat_name, df_merged, top_k=5, by_score=True)
    print(f" - [{cat_name}] Prompt: {prompt}" if prompt else f" - [{cat_name}] -> No keyword in the category")
```

---

### 16) Final analysis via LLM

**Core idea:** Use your LLM prompts (e.g., `'final_analysis.txt'`) with the re‑categorized results to generate narrative insights.

> See the notebook’s final markdown cell: *“Use LLM and re‑categorized results (keyword, frequency, scores) for Final Analysis by LLM (use `final_analysis.txt` prompt)”*.

---

##  Configuration you may want to change

* **CSV path/encoding:** `Data/data.csv`, `encoding="cp949"` → change to your file & encoding.
* **spaCy model:** `en_core_web_trf` (transformer‑based, large). If memory is tight, try `en_core_web_sm`.
* **Embedding model:** `"BAAI/bge-m3"` → switch to a smaller model for CPU‑only runs.
* **LLM labeling:** remove the Llama aspect in `representation_model` if you do not use a local GGUF model.
* **Outputs:** all exports live under `./output/`.

---

##  Citation

```plain
이우섭. (2025‑04‑23). AI 기반 이미지 텍스트화를 활용한 건축 이미지 데이터 추이 분석
- 건축물 이미지 기반 트렌드 분석 사례를 중심으로.
대한건축학회 학술발표대회 논문집, 서울.
```

---

##  Acknowledgements

This project builds on **BERTopic** for topic discovery (UMAP/HDBSCAN + c‑TF‑IDF, multi‑aspect representations, rich visualisations, etc.). See the BERTopic README & docs for algorithmic details and advanced features.&#x20;

---

##  Contributing

Bug reports, feature ideas and real‑world use cases are welcome – open an **Issue** or **Pull Request**.

---

##  License

Released under the **MIT License**.

```
