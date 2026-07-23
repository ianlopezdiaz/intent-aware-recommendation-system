"""Supervised product category classifier.

The chosen model, arrived at in notebook 02, is a linear support vector
classifier over TF-IDF product text (`title` + `concatenated_tags`) combined
with a handful of scaled numeric product features. See notebook 02 for the
comparison against Random Forest and KNN, and against a hashing-trick text
vectorizer, that led to this choice.
"""

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

TEXT_FEATURE = "product_text"

NUMERIC_FEATURES = ["price", "minimum_quantity", "weight", "price_per_weight"]


def build_pipeline(C: float = 1.0, random_state: int = 42) -> Pipeline:
    """Build the product classification pipeline: TF-IDF text + scaled numeric -> LinearSVC.

    The returned pipeline expects a DataFrame with a `product_text` column
    (see `src.features.text.joined_tokens`) and the columns listed in
    `NUMERIC_FEATURES`. Numeric features are median-imputed before scaling,
    since `weight` and `price_per_weight` carry NaN for products with a
    missing `weight` (see notebook 01's `weight_missing` handling).

    Parameters
    ----------
    C : float, default 1.0
        Regularization strength for `LinearSVC`. 1.0 is the value notebook
        02's bounded hyperparameter search (`RandomizedSearchCV` over
        `C in [0.01, ..., 30]`) selected as best on cross-validated
        macro-F1, i.e. the library default was already the right choice.
    random_state : int, default 42
        Passed to `LinearSVC` for reproducible fits.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline: `ColumnTransformer` (TF-IDF on `product_text`,
        median-impute + `StandardScaler` on `NUMERIC_FEATURES`) followed by
        `LinearSVC`.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(min_df=2), TEXT_FEATURE),
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classify", LinearSVC(C=C, random_state=random_state, dual="auto")),
        ]
    )
