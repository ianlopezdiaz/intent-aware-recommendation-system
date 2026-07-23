# Notebooks

Run in numeric order; each one loads artifacts (`data/processed/*.parquet`, `models/*.joblib`) produced by the ones before it.

| Notebook | Builds | Key output |
|---|---|---|
| [`01_exploratory_data_analysis.ipynb`](01_exploratory_data_analysis.ipynb) | Data understanding, cleaning decisions, feature catalog | `data/processed/01_data.parquet` |
| [`02_product_classification.ipynb`](02_product_classification.ipynb) | Supervised product -> category classifier | `models/02_product_classifier.joblib` |
| [`03_search_intent_modeling.ipynb`](03_search_intent_modeling.ipynb) | Unsupervised search intent classes, query -> intent classifier | `models/03_intent_classifier.joblib` |
| [`04_recommendation_engine.ipynb`](04_recommendation_engine.ipynb) | Content-based and intent-aware hybrid recommender | `data/processed/04_product_vectors.npz` |
| [`05_system_integration.ipynb`](05_system_integration.ipynb) | Query -> category (no numeric features), end-to-end pipeline, CLI | `models/05_query_category_classifier.joblib`, `src/pipeline.py`, `app/cli.py` |
| [`06_evaluation.ipynb`](06_evaluation.ipynb) | Cross-system evaluation, clustering validity, recommendation diversity, error analysis | `reports/06_evaluation_summary.png` |

Each notebook opens with the questions it answers and closes with a "Key findings & handoff" section pointing to what the next one needs; read those two sections first if you're skimming rather than reading start to finish.
