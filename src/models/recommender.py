"""Hybrid product recommender combining category and intent signals.

Notebook 04 builds the content/popularity/intent-aware half of this module.
Category-awareness (applying a product classifier to raw query text, which
needs the "no numerical features available" handling the challenge spec
calls out as its own problem) is added in notebook 05, once that problem is
actually solved there.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.features.text import joined_tokens

CATEGORY_CAP_EXPLORATORY = 2

OUTPUT_COLUMNS = ["product_id", "title", "category"]


@dataclass
class ProductCatalog:
    """The product universe a query is recommended against.

    Attributes
    ----------
    products : pandas.DataFrame
        One row per product, must include `product_id`, `title`, `category`,
        `view_counts`. Row order must match `text_matrix`'s row order.
    text_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
        Fitted vectorizer, shared with `src.models.classifier`'s product
        classification pipeline rather than refit here.
    text_matrix : scipy.sparse.csr_matrix
        `text_vectorizer.transform(...)` applied to every product's text,
        precomputed so a query only needs one additional `transform` call.
    """

    products: pd.DataFrame
    text_vectorizer: TfidfVectorizer
    text_matrix: sparse.csr_matrix


def build_catalog(
    products: pd.DataFrame, text_vectorizer: TfidfVectorizer, text_column: str = "product_text"
) -> ProductCatalog:
    """Assemble a `ProductCatalog` by vectorizing every product's text once.

    Parameters
    ----------
    products : pandas.DataFrame
        One row per product, including `text_column`, `product_id`, `title`,
        `category`, `view_counts`. Index is reset so it lines up with the
        vectorized matrix's row order.
    text_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
        Fitted vectorizer to transform `products[text_column]` with.
    text_column : str, default "product_text"
        Column holding each product's cleaned, tokenized text (see
        `src.features.text.joined_tokens`).

    Returns
    -------
    ProductCatalog
    """
    products = products.reset_index(drop=True)
    text_matrix = text_vectorizer.transform(products[text_column])
    return ProductCatalog(products=products, text_vectorizer=text_vectorizer, text_matrix=text_matrix)


def content_scores(query: str, catalog: ProductCatalog) -> np.ndarray:
    """Cosine similarity between a query and every product in the catalog.

    Parameters
    ----------
    query : str
        Raw, uncleaned search query text.
    catalog : ProductCatalog

    Returns
    -------
    numpy.ndarray of float
        One similarity score per row of `catalog.products`, in (0, 1]. All
        zero if the query has no words in `catalog.text_vectorizer`'s
        vocabulary.
    """
    query_vector = catalog.text_vectorizer.transform([joined_tokens(query)])
    return cosine_similarity(query_vector, catalog.text_matrix).flatten()


def recommend_popularity(catalog: ProductCatalog, k: int = 10) -> pd.DataFrame:
    """Recommend the k most-viewed products, ignoring query text entirely.

    Parameters
    ----------
    catalog : ProductCatalog
    k : int, default 10

    Returns
    -------
    pandas.DataFrame
        `k` rows with columns `product_id`, `title`, `category`, ranked by
        `view_counts` descending.
    """
    top_idx = np.argsort(catalog.products["view_counts"].to_numpy())[::-1][:k]
    return catalog.products.iloc[top_idx][OUTPUT_COLUMNS].reset_index(drop=True)


def recommend_content(query: str, catalog: ProductCatalog, k: int = 10) -> pd.DataFrame:
    """Recommend the k products most textually similar to a query.

    Falls back to `recommend_popularity` when the query has zero similarity
    to every product (e.g. no vocabulary overlap at all), rather than
    returning an arbitrary top-k from an all-zero score vector.

    Parameters
    ----------
    query : str
        Raw search query text.
    catalog : ProductCatalog
    k : int, default 10

    Returns
    -------
    pandas.DataFrame
        `k` rows with columns `product_id`, `title`, `category`,
        `similarity`, ranked by cosine similarity descending. `similarity`
        is absent when the popularity fallback was used.
    """
    scores = content_scores(query, catalog)
    if scores.max() <= 0:
        return recommend_popularity(catalog, k=k)
    top_idx = np.argsort(scores)[::-1][:k]
    return (
        catalog.products.iloc[top_idx][OUTPUT_COLUMNS]
        .assign(similarity=scores[top_idx])
        .reset_index(drop=True)
    )


def recommend_hybrid(
    query: str,
    catalog: ProductCatalog,
    intent_pipeline,
    k: int = 10,
    cap_per_category: int = CATEGORY_CAP_EXPLORATORY,
) -> pd.DataFrame:
    """Recommend k products, diversifying across categories for exploratory queries.

    Content similarity is always the primary ranking signal. When
    `intent_pipeline` predicts the query is `"exploratory"`, the top-k is
    built with at most `cap_per_category` products from any single
    `category`, so a broad query doesn't return a handful of near-duplicate
    listings from one category. `"specific"` queries get the plain top-k by
    similarity, uncapped: capping a query that's already narrowly and
    correctly targeted only replaces good results with worse, more "diverse"
    ones for no benefit (see notebook 04, `"anel de prata"` example).

    Parameters
    ----------
    query : str
        Raw search query text.
    catalog : ProductCatalog
    intent_pipeline : sklearn estimator
        Fitted pipeline from `src.models.intent.build_pipeline`, or
        equivalent: `.predict([text])` returning `"specific"` or
        `"exploratory"`.
    k : int, default 10
    cap_per_category : int, default `CATEGORY_CAP_EXPLORATORY`
        Maximum products from one `category` in the result, applied only
        for `exploratory` queries.

    Returns
    -------
    pandas.DataFrame
        `k` rows with columns `product_id`, `title`, `category`,
        `similarity`. `similarity` is absent when the popularity fallback
        was used (see `recommend_content`).
    """
    scores = content_scores(query, catalog)
    if scores.max() <= 0:
        return recommend_popularity(catalog, k=k)

    intent = intent_pipeline.predict([joined_tokens(query)])[0]
    if intent != "exploratory":
        top_idx = np.argsort(scores)[::-1][:k]
        return (
            catalog.products.iloc[top_idx][OUTPUT_COLUMNS]
            .assign(similarity=scores[top_idx])
            .reset_index(drop=True)
        )

    order = np.argsort(scores)[::-1]
    categories = catalog.products["category"].to_numpy()
    category_counts: dict[str, int] = {}
    chosen = []
    for idx in order:
        category = categories[idx]
        if category_counts.get(category, 0) >= cap_per_category:
            continue
        chosen.append(idx)
        category_counts[category] = category_counts.get(category, 0) + 1
        if len(chosen) == k:
            break
    chosen_idx = np.array(chosen)
    return (
        catalog.products.iloc[chosen_idx][OUTPUT_COLUMNS]
        .assign(similarity=scores[chosen_idx])
        .reset_index(drop=True)
    )
