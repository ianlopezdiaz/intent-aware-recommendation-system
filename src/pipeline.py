"""End-to-end query -> category, intent, recommendations pipeline.

This is the Part 5 hybrid system: an arbitrary search query in, three
outputs out (predicted category, predicted intent, top-k recommended
products). Assembled entirely from artifacts notebooks 02-05 already
produce; nothing here is fit.
"""

from dataclasses import dataclass

import pandas as pd

from src.features.text import joined_tokens
from src.models.recommender import ProductCatalog, recommend_hybrid


@dataclass
class PipelineResult:
    """The three outputs Part 5 requires for one query.

    Attributes
    ----------
    query : str
        The raw query this result is for.
    category : str
        Predicted product category, from
        `src.models.classifier.build_query_category_pipeline` (text-only:
        no numeric features are available for a bare query).
    intent : str
        Predicted search intent, `"specific"` or `"exploratory"`, from
        `src.models.intent.build_pipeline`.
    recommendations : pandas.DataFrame
        Top-k rows with columns `product_id`, `title`, `category`,
        `similarity` (see `src.models.recommender.recommend_hybrid`).
    """

    query: str
    category: str
    intent: str
    recommendations: pd.DataFrame


def run_pipeline(
    query: str,
    category_pipeline,
    intent_pipeline,
    catalog: ProductCatalog,
    k: int = 10,
) -> PipelineResult:
    """Run the full hybrid system on one query.

    Parameters
    ----------
    query : str
        Raw, uncleaned search query text.
    category_pipeline : sklearn estimator
        Fitted pipeline from
        `src.models.classifier.build_query_category_pipeline`.
        `.predict([text])` returning a category label.
    intent_pipeline : sklearn estimator
        Fitted pipeline from `src.models.intent.build_pipeline`.
        `.predict([text])` returning `"specific"` or `"exploratory"`.
    catalog : src.models.recommender.ProductCatalog
        The product catalog to recommend from.
    k : int, default 10
        Number of products to recommend.

    Returns
    -------
    PipelineResult
    """
    query_text = joined_tokens(query)
    category = category_pipeline.predict([query_text])[0]
    intent = intent_pipeline.predict([query_text])[0]
    recommendations = recommend_hybrid(query, catalog, intent_pipeline, k=k)
    return PipelineResult(query=query, category=category, intent=intent, recommendations=recommendations)
