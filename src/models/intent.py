"""Unsupervised search intent modeling.

The intent classes, arrived at in notebook 03, are `specific` and `exploratory`,
defined by `category_top_share`: whether a query's clicks concentrate in one
`category` or spread across several. Labels for known queries come from a
two-stage design: behavioral aggregates decide the class for queries with
enough click history (`MIN_CLICKS_FOR_DISCOVERY`), and a text-only classifier
(`build_pipeline`) generalizes those labels to any query, seen or not. See
`intent-methodology.md` (workspace root) and notebook 03 for the full
reasoning and the numbers behind each choice.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

MIN_CLICKS_FOR_DISCOVERY = 3

TOP_SHARE_THRESHOLD = 0.8


def category_top_share(categories: pd.Series) -> float:
    """Share of values equal to the single most common value.

    Parameters
    ----------
    categories : pandas.Series
        A query's `category` values across its clicks.

    Returns
    -------
    float
        Fraction of `categories` equal to its mode, in (0, 1]. 1.0 means
        every click landed in the same category.
    """
    return categories.value_counts(normalize=True).iloc[0]


def build_pipeline(class_weight: str | None = "balanced", random_state: int = 42) -> Pipeline:
    """Build the query classification pipeline: TF-IDF (uni+bigrams) -> LinearSVC.

    Unlike `src.models.classifier.build_pipeline`, this pipeline takes plain
    text (no numeric side input): a query word-count feature was tried during
    notebook 03's development and made no measurable difference, so it was
    left out. Bigrams, unlike in `src.models.classifier`, do measurably help
    here.

    Parameters
    ----------
    class_weight : str or None, default "balanced"
        Passed to `LinearSVC`. `"balanced"` trades some accuracy for better
        recall on the minority `exploratory` class, the right tradeoff given
        this project's use of macro-F1, not accuracy, as its imbalanced
        classification metric (see notebook 02 and notebook 03 §6).
    random_state : int, default 42
        Passed to `LinearSVC` for reproducible fits.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline: `TfidfVectorizer(min_df=2, ngram_range=(1, 2))`
        followed by `LinearSVC`. Expects a 1D iterable of query text (already
        cleaned/tokenized/joined, e.g. via `src.features.text.joined_tokens`).
    """
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
            (
                "classify",
                LinearSVC(class_weight=class_weight, random_state=random_state, dual="auto"),
            ),
        ]
    )
