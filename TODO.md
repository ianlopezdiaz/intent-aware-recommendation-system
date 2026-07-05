# Intent-Aware Recommendation System

## Repository Setup

- [x] Create GitHub repository
- [x] Add MIT License
- [x] Add project scaffolding
- [ ] Create virtual environment
- [ ] Add pyproject.toml
- [ ] Configure pre-commit (optional)

---

## Data

- [ ] Download dataset
- [ ] Create data dictionary
- [ ] Validate data integrity
- [ ] Document dataset

---

## Notebook 01 – Data Understanding

- [ ] Import libraries
- [ ] Load dataset
- [ ] Dataset overview
- [ ] Missing values
- [ ] Target distribution
- [ ] Numerical features
- [ ] Text features
- [ ] Initial findings

---

## Notebook 02 – Product Classification

- [ ] Baseline model
- [ ] Feature engineering
- [ ] Model comparison
- [ ] Evaluation

---

## Notebook 03 – Search Intent Modeling

- [ ] Define hypotheses
- [ ] Engineer query features
- [ ] Clustering
- [ ] Interpret clusters
- [ ] Intent labels

---

## Notebook 04 – Recommendation Engine

- [ ] Popularity baseline
- [ ] Content-based recommender
- [ ] Hybrid recommender
- [ ] Recommendation examples

---

## Notebook 05 – System Integration

- [ ] Build end-to-end pipeline
- [ ] CLI
- [ ] Test with unseen queries

---

## Notebook 06 – Evaluation

- [ ] Classification metrics
- [ ] Clustering evaluation
- [ ] Recommendation metrics
- [ ] Error analysis
- [ ] Future work

---

## README

- [ ] Project overview
- [ ] Installation
- [ ] Results
- [ ] Repository structure
- [ ] References



## Backlog

- [ ] Streamlit demo
- [ ] FastAPI endpoint
- [ ] Docker support
- [ ] Unit tests
- [ ] CI with GitHub Actions
- [ ] Sentence Transformers
- [ ] SHAP explanations
- [ ] Interactive recommendation visualization


```
intent-aware-recommendation-system/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01-data-understanding.ipynb
│   ├── 02-product-classification.ipynb
│   ├── 03-search-intent-modeling.ipynb
│   ├── 04-recommendation-engine.ipynb
│   ├── 05-system-integration.ipynb
│   └── 06-evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── text.py
│   │   └── numerical.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   ├── intent.py
│   │   └── recommender.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   │
│   └── pipeline.py
│
├── models/
│
├── reports/
│   └── figures/
│
├── docs/
│   └── original-challenge.md
│
└── app/
    └── cli.py
```