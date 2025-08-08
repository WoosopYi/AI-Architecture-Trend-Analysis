<!-- 언어 스위처 -->
<p align="right">
  <a href="README.md">
    <img alt="English" src="https://img.shields.io/badge/EN-English-blue?style=flat-square">
  </a>
  <img alt="한국어" src="https://img.shields.io/badge/KR-Korean-black?style=flat-square">
</p>

# AI‑Architecture‑Trend‑Analysis

[![PyPI - Python](https://img.shields.io/badge/python-v3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/WoosopYi/AI-Architecture-Trend-Analysis?style=social)](https://github.com/WoosopYi/AI-Architecture-Trend-Analysis/stargazers)

<img src="docs/assets/pipeline_diagram.png" width="38%" align="right" alt="Pipeline overview" />

**AI‑Architecture‑Trend‑Analysis**는 캡션 형태의 텍스트로부터 건축 디자인 트렌드를 **추출·클러스터링·해석**하는 즉시 사용 가능한 파이프라인입니다.

- **파이프라인:** Sentence‑Transformers → UMAP/HDBSCAN (BERTopic) → LLM 기반 재분류  
- **입력:** 이미지 캡션 또는 정리된 설명 *(캡션은 VLM으로 별도 생성 가능하며, 본 노트북은 텍스트부터 시작합니다)*

---

##  주요 특징

- **텍스트만으로 시작:** 캡션/설명 텍스트만 있으면 라벨링 없이 바로 분석.
- **BERTopic 기반 발견:** 임베딩 → UMAP → HDBSCAN → c‑TF‑IDF로 명확하고 해석 가능한 토픽 도출.
- **도메인 친화 재분류:** LLM(또는 규칙/프롬프트)로 원시 토픽을 건축가 친화 카테고리로 매핑.
- **프롬프트 생성:** 카테고리 핵심 키워드로 생성형 이미지‑AI용 프롬프트 자동 구성.
- **단일 노트북 워크플로:** `AI_Trend_Analysis.ipynb` 하나로 재현성 있게 전 과정을 실행.

---

<h2>리포지토리 구성</h2>
<p>리포지토리 한눈에 보기:</p>

<table>
  <thead>
    <tr>
      <th align="left">경로</th>
      <th align="left">용도</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="AI_Trend_Analysis.ipynb"><code>AI_Trend_Analysis.ipynb</code></a></td>
      <td>메인 코드</td>
    </tr>
    <tr>
      <td><a href="Data/"><code>Data/</code></a></td>
      <td>입력 CSV 저장 위치(예: <code>data.csv</code>)</td>
    </tr>
    <tr>
      <td><a href="output/"><code>output/</code></a></td>
      <td>결과물(CSV/JSON/플롯)</td>
    </tr>
    <tr>
      <td><a href="prompt/"><code>prompt/</code></a></td>
      <td>LLM 프롬프트</td>
    </tr>
    <tr>
      <td><a href="requirements1.txt"><code>requirements1.txt</code></a></td>
      <td>환경 구축용 패키지 목록</td>
    </tr>
    <tr>
      <td><a href="LICENSE"><code>LICENSE</code></a></td>
      <td>MIT 라이선스</td>
    </tr>
    <tr>
      <td><a href="README.md"><code>README.md</code></a> · <a href="README.kor.md"><code>README.kor.md</code></a></td>
      <td>문서</td>
    </tr>
  </tbody>
</table>



<details>
<summary><strong>폴더 트리</strong> (펼치기)</summary>

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
└── README.md
````

</details>

---

##  빠른 시작

클론 → 설치 → 데이터 준비 → (선택) 로컬 GGUF 모델 추가 → 실행

```bash
git clone https://github.com/WoosopYi/AI-Architecture-Trend-Analysis.git
cd AI-Architecture-Trend-Analysis

# 1) 의존성 설치 (Python 3.10+ 권장)
pip install -r requirements1.txt

# requirements 파일이 없거나 노트북 내 설치와 동일하게 하려면:
pip install --upgrade pip plotly kaleido
pip install "llama-cpp-python" bertopic datasets jinja2 spacy spacy-transformers
python -m spacy download en_core_web_trf

# 2) 입력 CSV를 ./Data/data.csv 로 준비 (아래 스키마 참고)

# 3) (선택) 로컬 LLM 토픽 라벨링용 GGUF 모델을 ./model/ 에 배치
# 예: openhermes-2.5-mistral-7b.Q4_K_M.gguf (quantized)
# mkdir -p model && cd model
# wget https://huggingface.co/.../openhermes-2.5-mistral-7b.Q4_K_M.gguf

# 4) 노트북을 위에서 아래로 실행
jupyter lab AI_Trend_Analysis.ipynb
```

**하드웨어 팁**

* `BAAI/bge-m3` 임베딩 및 `llama-cpp-python` 가속을 위해 **GPU 권장** (`n_gpu_layers=-1` 설정).
* **CPU‑only**도 동작하지만 속도가 느릴 수 있습니다.

---

##  입력 데이터 스키마 (CSV)

`./Data/data.csv`에  **`description`** 컬럼(이미지/프로젝트별 캡션형 텍스트)을 포함해 주세요.

```csv
description
"Slender concrete structure with vertical fins and recessed glazing..."
"Timber lattice facade with perforated metal screens..."
"Terraced green roof massing with porous podium..."
```

> 노트북은 `load_dataset("csv", data_files="Data/data.csv", encoding="cp949")["train"]`로 로드하고 `dataset["description"]`을 사용합니다. UTF‑8 파일이라면 `encoding="utf-8-sig"`로 변경하세요.

---

##  Step‑by‑Step

### 0) 환경 설정(필수 라이브러리 설치)

노트북에서 사용하는 의존성을 한 번 설치합니다.

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

### 0‑B) (선택) 양자화 LLM 다운로드

로컬 GGUF 모델을 받아 `/model`에 두고 오프라인/고속 라벨링에 사용합니다.

```bash
#!wget https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf
```

저장 경로: `/app/paper/model/openhermes-2.5-mistral-7b.Q4_K_M.gguf`

---

### 1) 임포트 & spaCy 파이프라인 로딩

커스텀 토크나이저에 사용할 `en_core_web_trf`를 로드합니다.

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

### 2) 커스텀 토크나이저(spaCy 기반)

불용어/도메인 불용어를 제거해 토픽의 해석력을 높입니다.

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

### 3) CSV 로드

`description` 컬럼을 갖는 캡션형 텍스트를 로딩합니다.

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

### 4) 로컬 Llama 모델 로드(`llama_cpp_python`)

BERTopic의 `LlamaCPP` 표현 모델로 사용할 GGUF LLM을 초기화합니다.

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

### 5) 임베딩 모델 & 벡터라이저

`BAAI/bge-m3`로 문서를 임베딩하고, spaCy 기반 토크나이저를 벡터라이저에 적용합니다.

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

### 6) UMAP/HDBSCAN(클러스터링)

차원 축소 후 밀집 클러스터를 찾아 주제군을 형성합니다.

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

### 7) BERTopic 생성 & 학습

임베딩/차원축소/클러스터링/표현 모델을 하나의 BERTopic 파이프라인으로 훈련합니다.

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

**토픽 테이블 내보내기:**

```python
info = topic_model.get_topic_info()
info.to_csv("/app/Github/output/topic_model.csv", index=True)
```

---

### 8) 시각화(개요 → 계층 → 분포)

내장 인터랙티브 플롯으로 토픽/문서/계층 구조를 탐색합니다.

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

### 9) 토픽별 키워드 테이블 구축

`(topic_id, keyword, score)`를 추출·정렬해 LLM 재분류의 입력으로 사용합니다.

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

**LLM 입력 JSON 내보내기:**

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

### 10) LLM으로 키워드 전처리(노트북 외부)

각 `(topic_id, keyword)`를 **건축 카테고리**로 매핑해 저장합니다.

* 프롬프트 파일: `'/Github/Prompt/json_pre_process_LLM_prompt.txt'`
* 결과 저장: `"/app/Github/output/architecture_category_llm_output.json"`

예시 JSON:

```json
[
  {"topic_id":0,"keyword":"window","category":"Facade composition"},
  {"topic_id":0,"keyword":"concrete","category":"Materials and Textures"}
]
```

---

### 11) LLM 카테고리 병합

예측 카테고리를 키워드 테이블에 조인하고 `"Unknown"`을 제거합니다.

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

### 12) 카테고리 집계: 점수 & 빈도

카테고리별 \*\*빈도(키워드 수)\*\*와 \*\*중요도(점수 합)\*\*를 요약합니다.

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

### 13) 카테고리 분포 시각화

**빈도**와 **점수 합** 기준으로 카테고리를 비교합니다.

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

### 14) 카테고리별 Top‑N 키워드(점수 기준)

카테고리별로 점수 합 기준 상위 키워드를 나열하고, 필요 시 막대그래프로 표시합니다.

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

### 15) 생성형 이미지‑AI용 프롬프트 생성

카테고리별 상위 키워드로 짧은 프롬프트를 구성합니다(점수/빈도 기준 선택).

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

### 16) LLM 기반 최종 해석

재분류 결과(키워드, 빈도, 점수)를 바탕으로 `final_analysis.txt` 프롬프트 등으로 **서술형 인사이트**를 생성합니다.

> 노트북 마지막 마크다운 참고: *“Use LLM and re‑categorized results (keyword, frequency, scores) for Final Analysis by LLM (use `final_analysis.txt` prompt)”*.

---

##  변경 가능한 설정

* **CSV 경로/인코딩:** `Data/data.csv`, `encoding="cp949"` → 파일 환경에 맞춰 변경
* **spaCy 모델:** `en_core_web_trf`(대형 트랜스포머). 메모리 제약 시 `en_core_web_sm`
* **임베딩 모델:** `"BAAI/bge-m3"` → CPU‑only 환경에선 더 작은 모델 고려
* **LLM 라벨링:** 로컬 GGUF 모델을 쓰지 않으면 `representation_model`에서 LLM 부분 제거
* **출력 경로:** 모든 산출물은 기본적으로 `./output/` 하위에 저장

---

##  인용

```plain
이우섭. (2025‑04‑23). AI 기반 이미지 텍스트화를 활용한 건축 이미지 데이터 추이 분석
- 건축물 이미지 기반 트렌드 분석 사례를 중심으로.
대한건축학회 학술발표대회 논문집, 서울.
```

---

##  감사의 말

본 프로젝트는 **BERTopic**(UMAP/HDBSCAN + c‑TF‑IDF, 다중 표현, 풍부한 시각화 등)에 기반합니다. 알고리즘/고급 기능은 BERTopic 문서를 참고하세요.

---

##  기여

버그 제보, 기능 제안, 실제 활용 사례 환영합니다. **Issue** 또는 **Pull Request**를 열어주세요.

---

##  라이선스

**MIT License**로 배포됩니다.

```
