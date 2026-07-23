"""Cleaning and preprocessing steps for the raw Elo7 dataset."""

import pandas as pd


def deduplicate_products(df: pd.DataFrame, keep: str = "first") -> pd.DataFrame:
    """Collapse click-level rows to one row per `product_id`.

    A row in the cleaned dataset is a click event, not a product: the same
    `product_id` can appear once per search that led to a click on it.
    `category`, `title`, and `concatenated_tags` are constant within a
    `product_id` (verified against the notebook 01 output), so repeat rows
    add no new signal for those columns, only extra weight for
    frequently-clicked products. `price` and `weight` do vary slightly
    across a product's repeat rows for a minority of products; this
    function does not attempt to reconcile that, it simply keeps one
    occurrence per product.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataset with one row per click, containing a `product_id` column.
    keep : str, default "first"
        Which occurrence to keep per `product_id`, passed through to
        `DataFrame.drop_duplicates`.

    Returns
    -------
    pandas.DataFrame
        One row per distinct `product_id`, sorted by `product_id` and with
        the index reset.
    """
    return (
        df.sort_values("product_id")
        .drop_duplicates(subset="product_id", keep=keep)
        .reset_index(drop=True)
    )
