# Intent-Aware Recommendation System

[![CI](https://github.com/ianlopezdiaz/intent-aware-recommendation-system/actions/workflows/ci.yml/badge.svg)](https://github.com/ianlopezdiaz/intent-aware-recommendation-system/actions/workflows/ci.yml)
[![Publish site](https://github.com/ianlopezdiaz/intent-aware-recommendation-system/actions/workflows/publish.yml/badge.svg)](https://github.com/ianlopezdiaz/intent-aware-recommendation-system/actions/workflows/publish.yml)

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

This project is based on a technical challenge I was given during a Data Scientist recruitment process at **Elo7** in 2021. That process didn't lead to an offer, and my [original submission](https://github.com/ianlopezdiaz/ELO7-recruitment-test) was left unfinished.

The problem stuck with me, so I picked it back up as an independent project — this repository is a from-scratch implementation, not a continuation of that submission.

The original challenge description is included here for context and can be found in:

- [Original challange (Brazilian Portuguese)](docs/elo7-ds-challenge-pt.md)
- [English translation](docs/elo7-ds-challenge-en.md)

## Repository Structure

```text
intent-aware-recommendation-system/
│
├── README.md                                   # Project overview and usage instructions.
├── index.qmd                                   # Landing page for the Quarto website.
├── _quarto.yml                                 # Quarto website configuration.
├── pyproject.toml                              # Project metadata and Python dependencies.
├── LICENSE                                     # Project license.
│
├── .github/
│   └── workflows/                              # CI (lint + notebook execution) and site publishing.
│
├── app/                                        # Application entry points and deployment code.
│
├── data/                                       # Project datasets.
│   ├── raw/                                        # Original, immutable data.
│   │   └── elo7_recruitment_dataset.csv                # Original Elo7 recruitment challenge dataset.
│   ├── interim/                                # Intermediate datasets generated during preprocessing.
│   ├── processed/                              # Final datasets ready for modeling.
│   └── external/                               # External datasets and third-party resources.
│
├── docs/                                       # Project documentation and challenge specification.
│   ├── elo7-ds-challenge-pt.md                     # Original challenge description (Brazilian Portuguese).
│   └── elo7-ds-challenge-en.md                     # English translation of the challenge description.
│
├── models/                                     # Trained models and serialized artifacts.
│
├── notebooks/                                  # Jupyter notebooks documenting the complete workflow.
│   ├── README.md                               # Notebook organization and execution order.
│   ├── 01_exploratory_data_analysis.ipynb          # Exploratory data analysis.
│   ├── 02_product_classification.ipynb             # Supervised product category classification.
│   ├── 03_search_intent_modeling.ipynb             # User search intent modeling.
│   ├── 04_recommendation_engine.ipynb              # Recommendation system development.
│   ├── 05_system_integration.ipynb                 # Integration of all system components.
│   └── 06_evaluation.ipynb                         # Model evaluation and performance analysis.
│
├── reports/                                    # Generated reports, figures, and visualizations.
│
├── src/                                        # Reusable source code.
│   ├── data/                                   # Data loading and preprocessing utilities.
│   ├── features/                               # Feature engineering and transformation.
│   ├── models/                                 # Model training and inference.
│   ├── evaluation/                             # Evaluation metrics and validation utilities.
│   └── utils/                                  # Shared helper functions.
│
└── _site/                                      # Website generated by Quarto.
    └── ...
```

## Installation

Requires Python 3.12+. From the repository root:

```shell
pip install -e ".[notebook]"
```

This installs the project's dependencies and the `src/` package in editable mode, so notebooks and scripts can `import src...` directly. The `notebook` extra adds Jupyter/`nbconvert`, needed to run and execute the notebooks (e.g. via `scripts/publish.sh`); omit it if you only need `src` as a library.

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

