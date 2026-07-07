# Intent-Aware Recommendation System

An end-to-end machine learning project for intent-aware product recommendation using NLP, product classification, and hybrid recommender systems.

## Overview

Modern e-commerce platforms rely on search and recommendation systems to help users discover relevant products.
While some users know exactly what they are looking for, others are exploring the catalog with broader or less specific queries.
Understanding this search intent can significantly improve the quality of product recommendations.

This project investigates how search intent can be inferred from user queries and leveraged to build an intent-aware recommendation system.
The proposed solution combines supervised learning, unsupervised learning, natural language processing (NLP), and recommendation techniques into a single end-to-end pipeline.

The final system is designed to receive an arbitrary search query as input and:

- Predict the most likely product category for the query.
- Infer the user's search intent.
- Recommend the ten most relevant products.

## Project Origin

This project is inspired by a technical challenge that was part of a Data Scientist recruitment process at **Elo7** in 2021.

Rather than preserving the original submission, this repository revisits the problem using modern machine learning techniques and software engineering practices. The implementation, experiments, evaluation, and project organization have been developed from scratch as a portfolio project.

The original challenge description is included in this repository for historical context and can be found in:

- [Original challange (Brazilian Portuguese)](docs/elo7-ds-challenge-pt.md)
- [English translation](docs/elo7-ds-challenge-en.md)

## Repository Structure

```text
intent-aware-recommendation-system/
│
├── app/
│
├── data/
│   ├── raw/
│   │   └── elo7_recruitment_dataset.csv
│   ├── interim/
│   ├── processed/
│   └── external/
│
├── docs/
│   ├── elo7-ds-challenge-pt.md
│   └── elo7-ds-challenge-en.md
│
├── models/
│
├── notebooks/
│   ├── README.md
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_product_classification.ipynb
│   ├── 03_search_intent_modeling.ipynb
│   ├── 04_recommendation_engine.ipynb
│   ├── 05_system_integration.ipynb
│   └── 06_evaluation.ipynb
│
├── reports/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── utils/
│
├── _quarto.yml
├── index.qmd
│
├── README.md
├── requirements.txt
├── LICENSE
└── TODO.md
```

## Project Roadmap

- [ ] Exploratory Data Analysis
- [ ] Product Classification
- [ ] Search Intent Modeling
- [ ] Recommendation Engine
- [ ] System Integration
- [ ] Evaluation
- [ ] Interactive Demo

## License

This project is licensed under the MIT License.

The original challenge statement contained in `elo7-ds-challenge-pt.md` is provided solely for historical reference and remains the intellectual property of its original authors.

