<!-- ─────────────────────────── Language Switcher ─────────────────────────── -->
[🇰🇷 한국어 README](README.kor.md)

# AI‑Architecture‑Trend‑Analysis 
[![PyPI - Python](https://img.shields.io/badge/python-v3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/WoosopYi/AI-Architecture-Trend-Analysis?style=social)](https://github.com/WoosopYi/AI-Architecture-Trend-Analysis/stargazers)

<img src="docs/assets/pipeline_diagram.png" width="38%" align="right" alt="Pipeline overview" />

A **turn‑key pipeline** that extracts, clusters & interprets architectural design trends  
from raw images using **Vision‑Language Models (VLMs) → BERTopic → LLM‑powered re‑categorisation**.

---

## ✨ Key Features

| What it does | Why it matters |
|--------------|----------------|
| **Zero‑label captioning** with a pre‑trained VLM | Analyse thousands of images at negligible human cost |
| **BERTopic** topic discovery on captions | Creates semantically rich clusters beyond word‑counts |
| **Domain‑aware re‑categorisation** via LLM | Maps raw topics to architect‑friendly classes (facade, form, material …) |
| **Prompt auto‑generation** for generative image‑AI | Instantly visualise or share emerging trends |
| **Single Jupyter notebook** (`AI_Trend_Analysis.ipynb`) | End‑to‑end demo for research, teaching or practice |

---

## 🔬 Methodology Overview

1. **Input** Curate award‑winning projects, star‑architect works & magazine imagery  
2. **VLM** Generate a full natural‑language description of each image  
3. **BERTopic** Embed → reduce (UMAP) → cluster (HDBSCAN) → topic keywords (c‑TF‑IDF)  
4. **Re‑Categorise** LLM maps keywords to architectural categories  
5. **Analysis** Keyword frequencies feed dashboards & *prompt templates* for generative AI  
   <br>*(see [`docs/assets/AI_image_trend_paper.pdf`](docs/AI_image_trend_paper.pdf) for full study)*

---

## ⚡ Quick Start

```bash
git clone https://github.com/WoosopYi/AI-Architecture-Trend-Analysis.git
cd AI-Architecture-Trend-Analysis
pip install -r requirements.txt        # or Conda / Poetry
jupyter lab AI_Trend_Analysis.ipynb    # run top → bottom
```

> **Tip 💡** Use a GPU‑backed environment for faster VLM inference.

---

## 📂 Repository Structure

```
AI-Architecture-Trend-Analysis/
├─ AI_Trend_Analysis.ipynb      ← main notebook
├─ Data/                        ← (optional) CSV / JSON inputs
├─ images/pipeline_diagram.png  ← diagram shown above
├─ docs/AI_image_trend_paper.pdf← conference paper
├─ requirements.txt
└─ README.md  |  README.kor.md
```

---

## 📝 Citation

```plain
이우섭. (2025‑04‑23). AI 기반 이미지 텍스트화를 활용한 건축 이미지 데이터 추이 분석
- 건축물 이미지 기반 트렌드 분석 사례를 중심으로.
대한건축학회 학술발표대회 논문집, 서울.
```

---

## 🤝 Contributing

Bug reports, feature ideas and real‑world use cases are welcome – open an **Issue** or **Pull Request**.

---

## 📜 License

Released under the **MIT License**.
