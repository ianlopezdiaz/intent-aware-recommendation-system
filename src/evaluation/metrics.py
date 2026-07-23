"""Evaluation metrics for classification, clustering, and recommendation quality."""

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd


def evaluate_recommender(
    recommend_fn: Callable[[str], set[int]], query_clicks: dict[str, set[int]]
) -> dict[str, float]:
    """Hit-rate and recall of a recommender against historical query clicks.

    A weak, biased, but real proxy for relevance: no ground-truth "relevant
    products for this query" label exists, but the products actually
    clicked for a historical query are a real (if exposure-biased) signal.

    Parameters
    ----------
    recommend_fn : callable
        Maps a query string to the set of `product_id`s it recommends.
        Callers are responsible for capping this to top-k before calling.
    query_clicks : dict of str to set of int
        Maps a query string to the set of `product_id`s actually clicked
        for it historically.

    Returns
    -------
    dict
        `hit_rate`: fraction of queries where at least one recommended
        product was actually clicked. `recall`: mean, per query, of the
        share of actually-clicked products that were recommended.
    """
    hits, recalls = [], []
    for query, clicked in query_clicks.items():
        if not clicked:
            continue
        recommended = recommend_fn(query)
        overlap = len(recommended & clicked)
        hits.append(1.0 if overlap > 0 else 0.0)
        recalls.append(overlap / len(clicked))
    return {"hit_rate": float(np.mean(hits)), "recall": float(np.mean(recalls))}


def category_diversity(categories: Iterable[str]) -> dict[str, float]:
    """Distinct-category count and category-distribution entropy for a set of recommendations.

    Complements `evaluate_recommender`'s relevance-focused metrics with a
    diversity-focused one: two top-k lists can have identical hit-rate while
    one repeats a single category and the other spans several.

    Parameters
    ----------
    categories : iterable of str
        One `category` value per recommended item, e.g. a top-k result's
        `category` column.

    Returns
    -------
    dict
        `n_distinct`: number of distinct categories present.
        `entropy`: Shannon entropy, in bits, of the category distribution;
        0 if every item shares one category, higher as the split evens out
        across more categories.
    """
    counts = pd.Series(list(categories)).value_counts(normalize=True)
    entropy = float(-(counts * np.log2(counts)).sum()) + 0.0  # avoid -0.0 when every item matches.
    return {"n_distinct": int(counts.shape[0]), "entropy": entropy}
