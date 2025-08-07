````markdown
<!-- ─────────────── Language switcher ─────────────── -->
[🇰🇷 한국어 README](README.kor.md)

# AI‑Architecture‑Trend‑Analysis
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **turn‑key pipeline** for extracting, clustering & interpreting architectural design trends  
from raw images with **Vision‑Language Models (VLMs) → BERTopic → LLM‑powered re‑categorisation**.

---

## ✨ Key Features
| What it does | Why it matters |
|--------------|----------------|
| **Zero‑label captioning** with a pre‑trained VLM | Analyse thousands of images at negligible human cost |
| **BERTopic** topic discovery on captions | Yields semantically rich clusters beyond word‑counts |
| **Domain‑aware re‑categorisation** via LLM | Converts raw topics into architect‑friendly classes (facade, form, material …) |
| **Prompt auto‑generation** for generative image‑AI | Instantly visualise or share emerging trends |
| **Single Jupyter notebook** (`AI_Trend_Analysis.ipynb`) | End‑to‑end demo for research, teaching or practice |

---

## 🔬 Methodology Overview
![Pipeline](images/pipeline_diagram.png)

1. **Input** Curate award‑winning projects, star‑architect works & magazine imagery.  
2. **VLM** Generate a full natural‑language description of each image.  
3. **BERTopic** Embed → reduce (UMAP) → cluster (HDBSCAN) → topic keywords (c‑TF‑IDF).  
4. **Re‑Categorise** LLM maps keywords to architectural categories.  
5. **Analysis** Keyword frequencies feed dashboards & *prompt templates* for generative AI.

*Full study: [`docs/AI_image_trend_paper.pdf`](docs/AI_image_trend_paper.pdf)*

---

## 🚀 Quick Start

```bash
git clone https://github.com/WoosopYi/AI-Architecture-Trend-Analysis.git
cd AI-Architecture-Trend-Analysis
pip install -r requirements.txt        # or Conda / Poetry
jupyter lab AI_Trend_Analysis.ipynb    # run top → bottom
```

---

## 📂 Repository Structure

```
AI-Architecture-Trend-Analysis/
│
├─ AI_Trend_Analysis.ipynb      # main notebook
├─ Data/                        # (optional) CSV / JSON inputs
├─ images/
│   └─ pipeline_diagram.png     # diagram shown above
├─ docs/
│   └─ AI_image_trend_paper.pdf # conference paper
├─ requirements.txt
└─ README.md  |  README.kor.md
```

---

## 📝 Citation

```plain
이우섭. (2025-04-23). AI 기반 이미지 텍스트화를 활용한 건축 이미지 데이터 추이 분석 - 건축물 이미지 기반 트렌드 분석 사례를 중심으로. 대한건축학회 학술발표대회 논문집, 서울.
```

---

## 🤝 Contributing
Bug reports, feature ideas and real‑world use cases are welcome – open an **Issue** or **Pull Request**.

---

## 📜 License
Released under the **MIT License**.
````
