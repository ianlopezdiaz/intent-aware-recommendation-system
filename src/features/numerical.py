"""Numerical feature engineering for price, weight, and engagement counts."""

import pandas as pd


def days_since(dates: pd.Series, reference_date: str | pd.Timestamp) -> pd.Series:
    """Number of days between a fixed reference date and each value in dates.

    Parameters
    ----------
    dates : pandas.Series
        Datetime values (e.g. `creation_date`).
    reference_date : str or pandas.Timestamp
        Fixed anchor date, e.g. the maximum `creation_date` observed in the
        dataset. Must be a fixed value, not `pd.Timestamp.now()`, so the
        feature is reproducible across runs. See notebook 01 for the
        chosen reference date and the reasoning behind it.

    Returns
    -------
    pandas.Series of int
        Days elapsed between `reference_date` and each value in `dates`.
    """
    reference_date = pd.to_datetime(reference_date)
    return (reference_date - dates).dt.days


def price_per_weight(price: pd.Series, weight: pd.Series) -> pd.Series:
    """Price divided by weight (grams) + 1.

    Parameters
    ----------
    price : pandas.Series
        Product price.
    weight : pandas.Series
        Product weight, in grams. Rows with `weight == 0` use the `+1`
        shift to avoid division by zero, at the cost of treating them as
        1-gram items. Rows where `weight` is NaN (missing, not zero)
        propagate NaN here rather than being folded into the same shift:
        a missing weight is not the same claim as a near-zero one.

    Returns
    -------
    pandas.Series of float
        `price / (weight + 1)`.
    """
    return price / (weight + 1)


def order_probability(order_counts: pd.Series, view_counts: pd.Series) -> pd.Series:
    """Order counts divided by view counts.

    Parameters
    ----------
    order_counts : pandas.Series
        Number of orders.
    view_counts : pandas.Series
        Number of views.

    Returns
    -------
    pandas.Series of float
        `order_counts / view_counts`, not clipped to [0, 1]: values above 1
        mean more recorded orders than views, an anomaly worth inspecting
        by the caller rather than something this function should silently
        hide.
    """
    return order_counts / view_counts


def rate_per_day(count: pd.Series, days: pd.Series) -> pd.Series:
    """Count divided by days elapsed + 1, e.g. views or orders per day.

    Parameters
    ----------
    count : pandas.Series
        Cumulative count (e.g. `view_counts`, `order_counts`).
    days : pandas.Series
        Days elapsed (e.g. from `days_since`). The `+1` shift avoids
        division by zero for products created on the same day as
        `reference_date` (`days == 0`), the same convention used by
        `price_per_weight` for `weight == 0`.

    Returns
    -------
    pandas.Series of float
        `count / (days + 1)`.
    """
    return count / (days + 1)
