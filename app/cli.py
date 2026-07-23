#!/usr/bin/env python
"""Command-line entry point for the challenge's Part 6 delivery script.

Usage
-----
    python app/cli.py --category "{'title': '...', 'concatenated_tags': '...', \
'price': 29.9, 'minimum_quantity': 1, 'weight': 150}"
    python app/cli.py --intent "anel de prata"
    python app/cli.py --recommendation "anel de prata"

`--category` and `--intent` are thin wrappers around the notebook 02 and 03
classifiers respectively: `--category` expects a real product's features
(numeric fields included), the ordinary product classification problem.
`--recommendation` is the only mode that takes a bare query and runs the
full query -> category/intent/recommendations pipeline from notebook 05,
since that's the case with no numeric features available.
"""

import argparse
import ast
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.load import load_processed_dataset  # noqa: E402
from src.data.preprocess import deduplicate_products  # noqa: E402
from src.features.numerical import price_per_weight as compute_price_per_weight  # noqa: E402
from src.features.text import joined_tokens  # noqa: E402
from src.models.classifier import NUMERIC_FEATURES, TEXT_FEATURE  # noqa: E402
from src.models.recommender import build_catalog  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402

MODELS_DIR = ROOT / "models"

CATEGORY_FEATURE_KEYS = ["title", "concatenated_tags", "price", "minimum_quantity", "weight"]


def _predict_category(features: dict) -> str:
    """Predict category for a real product description (Part 2's classifier)."""
    missing = [key for key in CATEGORY_FEATURE_KEYS if key not in features]
    if missing:
        raise ValueError(
            f"--category is missing required keys {missing}. Expected: {CATEGORY_FEATURE_KEYS}"
        )
    row = pd.DataFrame(
        {
            TEXT_FEATURE: [joined_tokens(features["title"], features["concatenated_tags"])],
            "price": [features["price"]],
            "minimum_quantity": [features["minimum_quantity"]],
            "weight": [features["weight"]],
            "price_per_weight": [
                compute_price_per_weight(pd.Series([features["price"]]), pd.Series([features["weight"]])).iloc[0]
            ],
        }
    )
    pipeline = joblib.load(MODELS_DIR / "02_product_classifier.joblib")
    return pipeline.predict(row[[TEXT_FEATURE, *NUMERIC_FEATURES]])[0]


def _predict_intent(query: str) -> str:
    """Predict search intent for a query (Part 3's classifier)."""
    pipeline = joblib.load(MODELS_DIR / "03_intent_classifier.joblib")
    return pipeline.predict([joined_tokens(query)])[0]


def _build_catalog():
    df = load_processed_dataset("01_data.parquet")
    products = deduplicate_products(df)
    products[TEXT_FEATURE] = [
        joined_tokens(t, tg) for t, tg in zip(products["title"], products["concatenated_tags"])
    ]
    classifier_pipeline = joblib.load(MODELS_DIR / "02_product_classifier.joblib")
    tfidf = classifier_pipeline.named_steps["preprocess"].named_transformers_["text"]
    return build_catalog(products, tfidf)


def _run_recommendation(query: str) -> None:
    """Run the full Part 5 hybrid system and print all three required outputs."""
    category_pipeline = joblib.load(MODELS_DIR / "05_query_category_classifier.joblib")
    intent_pipeline = joblib.load(MODELS_DIR / "03_intent_classifier.joblib")
    catalog = _build_catalog()

    result = run_pipeline(query, category_pipeline, intent_pipeline, catalog, k=10)
    print(result.category)
    print(result.intent)
    for _, row in result.recommendations.iterrows():
        print(f"{row['product_id']},{row['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--category", metavar="FEATURES_DICT", help="Predict category from product features.")
    group.add_argument("--intent", metavar="QUERY", help="Predict search intent from a query.")
    group.add_argument("--recommendation", metavar="QUERY", help="Run the full hybrid system on a query.")
    args = parser.parse_args()

    if args.category is not None:
        try:
            features = ast.literal_eval(args.category)
        except (ValueError, SyntaxError):
            parser.error("--category expects a dict literal, e.g. \"{'title': '...', ...}\"")
        if not isinstance(features, dict):
            parser.error("--category expects a dict literal, e.g. \"{'title': '...', ...}\"")
        try:
            print(_predict_category(features))
        except ValueError as error:
            parser.error(str(error))
    elif args.intent is not None:
        print(_predict_intent(args.intent))
    else:
        _run_recommendation(args.recommendation)


if __name__ == "__main__":
    main()
