<!-- ────────────────────────────────  헤더  ──────────────────────────────── -->

# AI‑Architecture‑Trend‑Analysis
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Automatically extract, cluster, and interpret architectural design trends from raw images  
using Vision‑Language Models (VLMs), BERTopic, and LLM‑powered recategorisation.**

---

## ✨ Key Features
| 무엇을 하나요? | 어떤 이점이 있나요? |
|----------------|--------------------|
| **라벨링 없이** 건축 사진에서 자연어 설명(*caption*) 추출 | 수작업 비용 0 — 대규모 이미지도 빠르게 처리 |
| **BERTopic**로 토픽(cluster)‑기반 트렌드 도출 | 단순 빈도 통계보다 의미 있는 구조화 |
| **LLM 재분류**로 ‘파사드 구성 / 형태 / 재료 …’ 등 도메인 카테고리 생성 | 건축 전문가가 바로 해석 가능한 결과 |
| **프롬프트 자동 생성** → 생성형‑AI 예시 이미지 | 트렌드를 시각적으로 검증·의사소통 |
| **노트북 한 개**로 end‑to‑end 진행 | 학습·연구/실무 DEMO용으로 간편 |

---

## 🔬 Methodology Overview
![Pipeline](images/pipeline_diagram.png)

1. **Input** 건축 수상작·유명 건축가·매거진 등에서 수집한 이미지를 입력  
2. **VLM** 사전학습 VLM이 이미지 → 자연어 설명(*full caption*) 생성  
3. **BERTopic** 캡션 임베딩 → UMAP 차원축소 → HDBSCAN 클러스터링  
4. **Re‑Categorisation** 토픽별 키워드를 LLM이 건축 카테고리로 재분류  
5. **Analysis** 카테고리·키워드 빈도 기반 트렌드 해석 & 생성형 AI용 프롬프트 추출

> 자세한 연구 배경·성과는 📄 [논문 전문 PDF](docs/AI_image_trend_paper.pdf) 에서 확인할 수 있습니다.  
> (2025 춘계학술발표대회, “AI 기반 이미지 텍스트화를 활용한 건축 이미지 데이터 추이 분석”) :contentReference[oaicite:0]{index=0}

---

## 🚀 Quick Start

```bash
git clone https://github.com/WoosopYi/AI-Architecture-Trend-Analysis.git
cd AI-Architecture-Trend-Analysis
pip install -r requirements.txt         # 또는 Poetry / Conda
jupyter lab AI_Trend_Analysis.ipynb     # 한 셀씩 실행
```
📂 Repository Structure

AI-Architecture-Trend-Analysis/
│
├─ AI_Trend_Analysis.ipynb   # 메인 노트북 (end‑to‑end)
├─ Data/                     # (선택) CSV · JSON 등 입력 데이터
├─ images/
│  └─ pipeline_diagram.png   # README용 다이어그램
├─ docs/
│  └─ AI_image_trend_paper.pdf  # 발표 논문
├─ requirements.txt          # 필수 파이썬 패키지
└─ README.md

📝 How to Cite

이우섭. (2025-04-23). AI 기반 이미지 텍스트화를 활용한 건축 이미지 데이터 추이 분석 - 건축물 이미지 기반 트렌드 분석 사례를 중심으로. 대한건축학회 학술발표대회 논문집, 서울.

🤝 Contributing
Pull request · Issue 환영합니다! 버그·개선 아이디어·실무 적용 사례를 공유해 주세요.

📜 License
MIT License — 자유롭게 사용·수정·배포할 수 있지만, 출처와 라이선스를 표시해 주세요.
